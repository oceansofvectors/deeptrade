"""
Train the risk management model.

Flow:
1. Train a trading model on train data (or load existing)
2. Run trading model through validation data → collect trade history
3. Train risk PPO model on that history
4. Evaluate: run trading model + risk model on test data
5. Compare results with and without risk model

Usage:
    python train_risk_model.py [--trading-timesteps 60000] [--risk-timesteps 50000]
"""

import argparse
import logging
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

try:
    from sb3_contrib import RecurrentPPO
    RECURRENT_PPO_AVAILABLE = True
except ImportError:
    RECURRENT_PPO_AVAILABLE = False

from config import config
from get_data import get_data
from environment import TradingEnv
from train import train_agent_iteratively, evaluate_agent
from normalization import scale_window, get_standardized_column_names
from indicators.lstm_features import LSTMFeatureGenerator
from risk_model import (
    RiskManagementEnv, RiskModelWrapper, RiskAction,
    collect_trade_history
)
from utils.seeding import set_global_seed
import money

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy loggers
for name in ['get_data', 'indicators', 'indicators.lstm_features', 'indicators.rsi',
             'indicators.adx', 'indicators.macd', 'indicators.atr', 'indicators.volume',
             'normalization', 'utils.data_utils', 'train']:
    logging.getLogger(name).setLevel(logging.WARNING)


def prepare_data(window_pct=0.6):
    """
    Load and prepare 4-way data split for unbiased risk model training.

    Split: train (50%) | val (15%) | embargo | risk_train (15%) | embargo | test (20%)

    - Trading model: trained on train, validated on val
    - Risk model: trained on risk_train (truly out-of-sample for trading model)
    - Both evaluated on test (unseen by both)
    """
    train_raw, val_raw, test_raw = get_data()
    df = pd.concat([train_raw, val_raw, test_raw])
    start_idx = int(len(df) * (1 - window_pct))
    df = df.iloc[start_idx:].copy()

    n = len(df)

    # Estimate embargo
    if n > 1:
        time_diff = (df.index[1] - df.index[0]).total_seconds()
        bars_per_day = int(24 * 3600 / time_diff) if time_diff > 0 else 288
    else:
        bars_per_day = 288
    embargo_bars = 1 * bars_per_day

    # 4-way split
    train_end = int(n * 0.50)
    val_end = int(n * 0.65)
    risk_train_start = val_end + embargo_bars
    risk_train_end = int(n * 0.80)
    test_start = risk_train_end + embargo_bars

    # Ensure test has enough data
    test_start = min(test_start, n - max(int(n * 0.10), 100))
    risk_train_start = min(risk_train_start, risk_train_end - 100)

    train_data = df.iloc[:train_end].copy()
    val_data = df.iloc[train_end:val_end].copy()
    risk_train_data = df.iloc[risk_train_start:risk_train_end].copy()
    test_data = df.iloc[test_start:].copy()

    return train_data, val_data, risk_train_data, test_data


def add_features(train_data, val_data, risk_train_data, test_data):
    """Add LSTM features and scale. Fits only on train_data to prevent leakage."""
    # LSTM features
    lstm_config = config.get("indicators", {}).get("lstm_features", {})
    if lstm_config.get("enabled", False):
        generator = LSTMFeatureGenerator(
            lookback=lstm_config.get("lookback", 20),
            hidden_size=lstm_config.get("hidden_size", 32),
            num_layers=lstm_config.get("num_layers", 1),
            output_size=lstm_config.get("output_size", 8),
            pretrain_epochs=lstm_config.get("pretrain_epochs", 50),
            pretrain_lr=lstm_config.get("pretrain_lr", 0.001),
            pretrain_batch_size=lstm_config.get("pretrain_batch_size", 64),
            pretrain_patience=lstm_config.get("pretrain_patience", 10)
        )
        generator.fit(train_data)
        train_data = generator.transform(train_data)
        val_data = generator.transform(val_data)
        risk_train_data = generator.transform(risk_train_data)
        test_data = generator.transform(test_data)

    # Scale — fit on train_data only, transform all splits
    # scale_window expects 3 splits, so we scale risk_train and test using the same scaler
    cols_to_scale = get_standardized_column_names(train_data)
    scaler_type = config.get("normalization", {}).get("scaler_type", "robust")

    # First scale train/val/risk_train (risk_train acts as "test" for scaler)
    _, train_data, val_data, risk_train_data = scale_window(
        train_data=train_data, val_data=val_data, test_data=risk_train_data,
        cols_to_scale=cols_to_scale, feature_range=(-1, 1), scaler_type=scaler_type
    )

    # Scale test data using same approach (train stats)
    # Re-run scale_window with test as the "test" split
    _, _, _, test_data = scale_window(
        train_data=train_data, val_data=val_data, test_data=test_data,
        cols_to_scale=cols_to_scale, feature_range=(-1, 1), scaler_type=scaler_type
    )

    # Drop raw columns (keep close for env price lookups)
    cols_to_drop = ['volume', 'Volume', 'SMA', 'EMA', 'VWAP', 'PSAR', 'OBV',
                    'VOLUME_NORM', 'DOW', 'position',
                    'open', 'Open', 'OPEN', 'high', 'low', 'High', 'Low', 'HIGH', 'LOW']
    for df in [train_data, val_data, risk_train_data, test_data]:
        cols_present = [c for c in cols_to_drop if c in df.columns]
        if cols_present:
            df.drop(columns=cols_present, inplace=True)

    return train_data, val_data, risk_train_data, test_data


def evaluate_with_risk_model(trading_model, risk_wrapper, test_data, label=""):
    """Evaluate trading model with risk model wrapper applied."""
    env = TradingEnv(
        test_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 2.50),
        position_size=config["environment"].get("position_size", 1)
    )

    obs, _ = env.reset()
    is_recurrent = hasattr(trading_model, 'policy') and hasattr(trading_model.policy, 'lstm')
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    portfolio_history = [float(env.net_worth)]
    trade_count = 0
    trades_allowed = 0
    trades_blocked = 0
    winning_trades = 0
    done = False

    while not done:
        # Get trading model prediction
        if is_recurrent:
            action, lstm_states = trading_model.predict(obs, state=lstm_states,
                                                         episode_start=episode_starts, deterministic=True)
            episode_starts = np.array([done])
        else:
            action, _ = trading_model.predict(obs, deterministic=True)

        action_int = int(action)

        # Check if this is a position change
        would_change = (action_int == 0 and env.position != 1) or \
                       (action_int == 1 and env.position != -1)

        if would_change and risk_wrapper.enabled:
            # Build portfolio state for risk model
            max_nw = float(env.max_net_worth)
            nw = float(env.net_worth)
            portfolio_state = {
                "position": env.position,
                "drawdown_pct": (max_nw - nw) / max_nw if max_nw > 0 else 0,
                "unrealized_pnl": float(env._calculate_unrealized_pnl_normalized()) * 500,
                "time_in_position": env.time_in_position,
                "net_worth": nw,
                "trade_count": trade_count,
            }

            allowed = risk_wrapper.should_allow(action_int, portfolio_state)

            if not allowed:
                # Block: use redundant action (stay in current position)
                if env.position == 1:
                    action = np.array(0)  # Stay long
                elif env.position == -1:
                    action = np.array(1)  # Stay short
                else:
                    action = np.array(action_int)  # No position, allow entry
                trades_blocked += 1
            else:
                trades_allowed += 1
        elif would_change:
            trades_allowed += 1

        old_nw = float(env.net_worth)
        old_pos = env.position
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio_history.append(float(env.net_worth))

        # Track trade results for risk model
        if info.get("position_changed", False):
            trade_count += 1
            step_pnl = float(env.net_worth) - old_nw
            if step_pnl > 0:
                winning_trades += 1
            if risk_wrapper.enabled:
                risk_wrapper.record_trade_result(step_pnl)

    # Calculate metrics
    final_nw = float(env.net_worth)
    initial = config["environment"]["initial_balance"]
    return_pct = (final_nw - initial) / initial * 100

    portfolio_array = np.array(portfolio_history)
    running_max = np.maximum.accumulate(portfolio_array)
    drawdowns = (portfolio_array - running_max) / running_max * 100
    max_drawdown = float(np.min(drawdowns))

    hit_rate = winning_trades / max(trade_count, 1) * 100

    if max_drawdown == 0:
        calmar = return_pct if return_pct > 0 else 0.0
    else:
        calmar = return_pct / abs(max_drawdown)

    risk_stats = risk_wrapper.get_stats() if risk_wrapper.enabled else {}

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Portfolio:     ${final_nw:>12,.2f}")
    print(f"  Return:        {return_pct:>+11.2f}%")
    print(f"  Calmar Ratio:  {calmar:>11.2f}")
    print(f"  Max Drawdown:  {max_drawdown:>11.2f}%")
    print(f"  Hit Rate:      {hit_rate:>11.1f}%")
    print(f"  Trade Count:   {trade_count:>11d}")
    if risk_stats:
        print(f"  Trades Allowed:{risk_stats['allowed']:>7d}")
        print(f"  Trades Blocked:{risk_stats['blocked']:>7d}")
        print(f"  Block Rate:    {risk_stats['block_rate']:>10.1f}%")
    print(f"  Min Portfolio: ${np.min(portfolio_array):>12,.2f}")
    print(f"  Max Portfolio: ${np.max(portfolio_array):>12,.2f}")
    print(f"{'=' * 60}")

    return {
        "return_pct": return_pct,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "trade_count": trade_count,
        "risk_stats": risk_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Train risk management model")
    parser.add_argument("--trading-timesteps", type=int, default=60000)
    parser.add_argument("--trading-iterations", type=int, default=5)
    parser.add_argument("--risk-timesteps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-pct", type=float, default=0.6)
    args = parser.parse_args()

    set_global_seed(args.seed)
    config["seed"] = args.seed

    output_dir = f"models/risk_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  RISK MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Prepare data — 4-way split for unbiased risk model training
    print("\n[1/6] Loading data (4-way split)...")
    train_data, val_data, risk_train_data, test_data = prepare_data(args.window_pct)
    print(f"  Train:      {len(train_data):>6} bars  (trading model training)")
    print(f"  Val:        {len(val_data):>6} bars  (trading model validation)")
    print(f"  Risk Train: {len(risk_train_data):>6} bars  (out-of-sample for risk model)")
    print(f"  Test:       {len(test_data):>6} bars  (final evaluation)")

    # 2. Add features
    print("\n[2/6] Adding features...")
    train_data, val_data, risk_train_data, test_data = add_features(
        train_data, val_data, risk_train_data, test_data
    )

    # 3. Train trading model on train, validate on val
    print(f"\n[3/6] Training trading model ({args.trading_timesteps} timesteps)...")
    t0 = time.time()
    eval_metric = config["training"]["evaluation"].get("metric", "sortino")
    trading_model, _, _ = train_agent_iteratively(
        train_data=train_data,
        validation_data=val_data,
        initial_timesteps=args.trading_timesteps,
        max_iterations=args.trading_iterations,
        n_stagnant_loops=3,
        additional_timesteps=15000,
        evaluation_metric=eval_metric,
        window_folder=output_dir
    )
    print(f"  Trading model trained in {time.time() - t0:.1f}s")

    # 4. Collect trade history from RISK_TRAIN data (truly out-of-sample)
    # The trading model has never seen this data — its behavior here is realistic
    print("\n[4/6] Collecting out-of-sample trade history...")
    trade_history = collect_trade_history(trading_model, risk_train_data)
    n_trades = len(trade_history["trades"])
    winning = sum(1 for t in trade_history["trades"] if t["pnl"] > 0)
    losing = n_trades - winning
    print(f"  {n_trades} out-of-sample trades ({winning} wins, {losing} losses)")
    print(f"  (Trading model never saw this data during training)")

    if n_trades < 10:
        print("  WARNING: Very few trades. Risk model may not learn well.")

    # 5. Train risk model
    print(f"\n[5/6] Training risk model ({args.risk_timesteps} timesteps)...")
    t0 = time.time()

    risk_env = RiskManagementEnv(trade_history)
    check_env(risk_env, skip_render_check=True)

    n_decisions = len(risk_env.decision_points)
    risk_model = PPO(
        "MlpPolicy",
        risk_env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=min(256, n_decisions),
        batch_size=min(64, n_decisions),
        gamma=0.95,  # Shorter horizon for risk decisions
        ent_coef=0.5,  # Very high entropy to prevent collapse to all-block
        seed=args.seed,
        policy_kwargs={"net_arch": [32, 16]},  # Small network
    )
    risk_model.learn(total_timesteps=args.risk_timesteps, progress_bar=True)

    risk_model_path = os.path.join(output_dir, "risk_model.zip")
    risk_model.save(risk_model_path)
    print(f"  Risk model trained in {time.time() - t0:.1f}s")
    print(f"  Saved to {risk_model_path}")

    # 6. Evaluate on test data
    print("\n[6/6] Evaluating on test data...")

    # Without risk model — explicitly disable
    no_risk = RiskModelWrapper(mode="rules")
    no_risk.enabled = False
    result_no_risk = evaluate_with_risk_model(
        trading_model, no_risk, test_data, "TEST — WITHOUT RISK MODEL"
    )

    # With rule-based risk model — loose thresholds, catastrophic-only
    rules_risk = RiskModelWrapper(mode="rules", rules_config={
        "max_drawdown_pct": 0.40,        # Only block at 40% DD (catastrophic)
        "max_consecutive_losses": 6,      # Allow up to 6 consecutive losses
        "cooldown_bars_after_losses": 10, # Short cooldown
        "max_daily_loss_pct": 0.25,       # 25% daily loss limit
        "min_bars_between_trades": 1,     # Almost no trade throttling
    })
    result_rules = evaluate_with_risk_model(
        trading_model, rules_risk, test_data, "TEST — RULE-BASED RISK MODEL"
    )

    # With PPO risk model
    with_risk = RiskModelWrapper(risk_model_path, mode="ppo")
    result_with_risk = evaluate_with_risk_model(
        trading_model, with_risk, test_data, "TEST — PPO RISK MODEL"
    )

    # Comparison
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON vs BASELINE (no risk)")
    print(f"{'=' * 60}")
    for name, result in [("Rules", result_rules), ("PPO", result_with_risk)]:
        dd_change = result["max_drawdown"] - result_no_risk["max_drawdown"]
        ret_change = result["return_pct"] - result_no_risk["return_pct"]
        print(f"  {name:>6}: Return {ret_change:>+.2f}%, DD {dd_change:>+.2f}% ({'better' if dd_change > 0 else 'worse'}), Calmar {result['calmar'] - result_no_risk['calmar']:>+.2f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
