"""
Quick single-window training & evaluation for fast iteration.
Trains on the most recent window only, prints detailed diagnostics.

Usage: python quick_test.py [--timesteps 20000] [--iterations 5]
"""

import argparse
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

from config import config
from get_data import get_data
from environment import TradingEnv
from train import train_agent_iteratively, evaluate_agent
from normalization import scale_window, get_standardized_column_names
from indicators.lstm_features import LSTMFeatureGenerator
from utils.seeding import set_global_seed
import money

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress noisy loggers
for name in ['get_data', 'indicators', 'indicators.lstm_features', 'indicators.rsi',
             'indicators.adx', 'indicators.macd', 'indicators.atr', 'indicators.volume',
             'normalization', 'utils.data_utils', 'train']:
    logging.getLogger(name).setLevel(logging.WARNING)


def prepare_single_window(df, train_ratio=0.6, val_ratio=0.2, embargo_days=1):
    """Split data into train/val/test for a single window."""
    n = len(df)
    train_idx = int(n * train_ratio)
    val_idx = train_idx + int(n * val_ratio)

    # Estimate bars per day
    if n > 1:
        time_diff = (df.index[1] - df.index[0]).total_seconds()
        bars_per_day = int(24 * 3600 / time_diff) if time_diff > 0 else 288
    else:
        bars_per_day = 288
    embargo_bars = embargo_days * bars_per_day

    train_data = df.iloc[:train_idx].copy()
    val_data = df.iloc[train_idx:val_idx].copy()
    test_start = min(val_idx + embargo_bars, n - max(int(n * 0.05), 10))
    test_data = df.iloc[test_start:].copy()

    return train_data, val_data, test_data


def add_lstm_features(train_data, val_data, test_data):
    """Add LSTM features if enabled in config."""
    lstm_config = config.get("indicators", {}).get("lstm_features", {})
    if not lstm_config.get("enabled", False):
        return train_data, val_data, test_data

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
    test_data = generator.transform(test_data)
    return train_data, val_data, test_data


def scale_and_clean(train_data, val_data, test_data):
    """Scale indicators and drop raw columns."""
    cols_to_scale = get_standardized_column_names(train_data)
    scaler_type = config.get("normalization", {}).get("scaler_type", "robust")
    scaler, train_data, val_data, test_data = scale_window(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        cols_to_scale=cols_to_scale,
        feature_range=(-1, 1),
        scaler_type=scaler_type
    )

    # Drop raw OHLCV columns (same as walk_forward), but keep 'close' for environment price lookups
    cols_to_drop = ['volume', 'Volume', 'SMA', 'EMA', 'VWAP', 'PSAR', 'OBV',
                    'VOLUME_NORM', 'DOW', 'position',
                    'open', 'Open', 'OPEN', 'high', 'low', 'High', 'Low', 'HIGH', 'LOW']
    for df in [train_data, val_data, test_data]:
        cols_present = [c for c in cols_to_drop if c in df.columns]
        if cols_present:
            df.drop(columns=cols_present, inplace=True)

    return train_data, val_data, test_data


def print_diagnostics(results, label=""):
    """Print detailed diagnostics from evaluation results."""
    header = f"{'=' * 60}"
    print(f"\n{header}")
    print(f"  {label} RESULTS")
    print(header)

    print(f"  Portfolio:     ${results['final_portfolio_value']:>12,.2f}  (started $100,000)")
    print(f"  Return:        {results['total_return_pct']:>+11.2f}%")
    print(f"  Calmar Ratio:  {results['calmar_ratio']:>11.2f}")
    print(f"  Sortino Ratio: {results.get('sortino_ratio', 0):>11.4f}")
    print(f"  Max Drawdown:  {results['max_drawdown']:>11.2f}%")
    print(f"  Hit Rate:      {results['hit_rate']:>11.1f}%")
    print(f"  Trade Count:   {results['trade_count']:>11d}")
    print(f"  Profitable:    {results.get('profitable_trades', 0):>11d}")

    # Action distribution
    ac = results.get('action_counts', {})
    total_actions = sum(ac.values()) if ac else 0
    if total_actions > 0:
        long_pct = ac.get(0, 0) / total_actions * 100
        short_pct = ac.get(1, 0) / total_actions * 100
        flat_pct = ac.get(2, 0) / total_actions * 100
        print(f"  Long Actions:  {ac.get(0, 0):>7d} ({long_pct:.1f}%)")
        print(f"  Short Actions: {ac.get(1, 0):>7d} ({short_pct:.1f}%)")
        if ac.get(2, 0) > 0:
            print(f"  Flat Actions:  {ac.get(2, 0):>7d} ({flat_pct:.1f}%)")

    # Trade P&L analysis
    trades = results.get('trade_history', [])
    if trades:
        winning = [t for t in trades if t.get('profitable', False)]
        losing = [t for t in trades if not t.get('profitable', False)]
        print(f"  Winning Trades:{len(winning):>7d}")
        print(f"  Losing Trades: {len(losing):>7d}")

    # Portfolio trajectory summary
    ph = results.get('portfolio_history', [])
    if ph:
        ph_arr = np.array(ph)
        print(f"  Min Portfolio: ${np.min(ph_arr):>12,.2f}")
        print(f"  Max Portfolio: ${np.max(ph_arr):>12,.2f}")

    print(header)


def main():
    parser = argparse.ArgumentParser(description="Quick single-window training test")
    parser.add_argument("--timesteps", type=int, default=20000, help="Initial training timesteps")
    parser.add_argument("--iterations", type=int, default=5, help="Max iterative training iterations")
    parser.add_argument("--additional", type=int, default=5000, help="Additional timesteps per iteration")
    parser.add_argument("--stagnant", type=int, default=3, help="Stagnant loops before stopping")
    parser.add_argument("--window-pct", type=float, default=0.4, help="Use last N%% of data (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_global_seed(seed)
    if args.seed is not None:
        config["seed"] = seed

    print("=" * 60)
    print("  QUICK TRAINING TEST")
    print(f"  Timesteps: {args.timesteps}, Iterations: {args.iterations}")
    print(f"  Additional per iter: {args.additional}, Stagnant limit: {args.stagnant}")
    print("=" * 60)

    # Load data — get_data returns (train, val, test) already split
    # We re-concat and take the tail for our single-window test
    t0 = time.time()
    print("\n[1/5] Loading data...")
    train_raw, val_raw, test_raw = get_data()
    df = pd.concat([train_raw, val_raw, test_raw])
    # Use only the last portion of data for speed
    start_idx = int(len(df) * (1 - args.window_pct))
    df = df.iloc[start_idx:].copy()
    print(f"  Using {len(df)} bars ({args.window_pct*100:.0f}% of data)")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # Split
    print("\n[2/5] Splitting into train/val/test...")
    train_data, val_data, test_data = prepare_single_window(df)
    print(f"  Train: {len(train_data)} bars ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"  Val:   {len(val_data)} bars ({val_data.index[0]} to {val_data.index[-1]})")
    print(f"  Test:  {len(test_data)} bars ({test_data.index[0]} to {test_data.index[-1]})")

    # LSTM features
    print("\n[3/5] Adding LSTM features...")
    train_data, val_data, test_data = add_lstm_features(train_data, val_data, test_data)

    # Scale
    print("\n[4/5] Scaling features...")
    train_data, val_data, test_data = scale_and_clean(train_data, val_data, test_data)
    print(f"  Feature columns ({len(train_data.columns)}): {list(train_data.columns)}")
    print(f"  Close range (raw): {train_data['close'].min():.2f} - {train_data['close'].max():.2f}")
    print(f"  close_norm range:  {train_data['close_norm'].min():.4f} - {train_data['close_norm'].max():.4f}")

    # Train
    print(f"\n[5/5] Training model ({args.timesteps} initial + up to {args.iterations}x{args.additional} additional)...")
    eval_metric = config["training"]["evaluation"].get("metric", "calmar")
    t_train = time.time()

    # Create output folder for this run
    window_folder = f"models/quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(window_folder, exist_ok=True)

    best_model, best_results, all_results = train_agent_iteratively(
        train_data=train_data,
        validation_data=val_data,
        initial_timesteps=args.timesteps,
        max_iterations=args.iterations,
        n_stagnant_loops=args.stagnant,
        additional_timesteps=args.additional,
        evaluation_metric=eval_metric,
        window_folder=window_folder
    )
    train_time = time.time() - t_train

    # Print iteration-by-iteration summary
    print(f"\n  {'Iter':>4}  {'Val Calmar':>10}  {'Portfolio':>12}  {'Trades':>6}  {'Hit Rate':>8}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*12}  {'-'*6}  {'-'*8}")
    for i, r in enumerate(all_results):
        calmar = r.get('calmar_ratio', 0)
        port = r.get('final_portfolio_value', 0)
        trades = r.get('trade_count', 0)
        hr = r.get('hit_rate', 0)
        best_mark = " *" if r.get('is_best', False) else ""
        print(f"  {i:>4}  {calmar:>+10.2f}  ${port:>11,.2f}  {trades:>6}  {hr:>7.1f}%{best_mark}")

    # Evaluate on validation
    print_diagnostics(best_results, "VALIDATION (best iteration)")

    # Evaluate on test
    print("\nEvaluating on TEST set...")
    test_results = evaluate_agent(best_model, test_data, verbose=0, deterministic=True)
    print_diagnostics(test_results, "TEST (out-of-sample)")

    # Summary
    total_time = time.time() - t0
    print(f"\n  Training time: {train_time:.1f}s")
    print(f"  Total time:    {total_time:.1f}s")

    # Overfitting check
    val_ret = best_results.get("total_return_pct", 0)
    test_ret = test_results.get("total_return_pct", 0)
    if val_ret > 0 and test_ret < 0:
        print("\n  ⚠  OVERFITTING SIGNAL: Positive validation, negative test return")
    elif val_ret > 0 and test_ret > 0:
        print(f"\n  ✓  Val and test both positive. Generalization ratio: {test_ret/val_ret:.2f}")


if __name__ == "__main__":
    main()
