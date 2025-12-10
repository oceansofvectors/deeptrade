"""
Backtesting engine for World Model agent.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..models.vae import VAE
from ..models.mdn_rnn import MDNRNN
from ..models.controller import Controller
from ..models.world_model import FullAgent
from ..envs.trading_env import TradingEnv
from .metrics import calculate_metrics, TradeMetrics, print_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    metrics: TradeMetrics
    net_worths: np.ndarray
    positions: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    timestamps: Optional[np.ndarray] = None


def backtest(
    vae: VAE,
    mdn_rnn: MDNRNN,
    controller: Controller,
    data: pd.DataFrame,
    initial_balance: float = 10000.0,
    position_size: int = 1,
    observation_columns: Optional[List[str]] = None,
    deterministic: bool = True,
    stop_loss_pct: Optional[float] = None,
    take_profit_pct: Optional[float] = None,
    close_position_eod: bool = True,
    point_value: float = 20.0,
    transaction_cost: float = 0.0,
    device: str = 'cpu',
    verbose: bool = True,
    risk_controller=None
) -> BacktestResult:
    """
    Run backtest on historical data.

    Args:
        vae: Trained VAE
        mdn_rnn: Trained MDN-RNN
        controller: Trained controller
        data: DataFrame with OHLCV and indicator data
        initial_balance: Starting balance
        position_size: Number of contracts (base size, scaled by risk controller)
        observation_columns: Columns to use as observations
        deterministic: Use deterministic actions
        stop_loss_pct: Stop loss as percentage of portfolio (overridden by risk controller if present)
        take_profit_pct: Take profit as percentage of portfolio (overridden by risk controller if present)
        close_position_eod: If True and data has day boundaries, close positions at end of day
        point_value: Dollar value per point (default $20 for NQ)
        transaction_cost: Cost per trade in dollars
        device: Device for inference
        verbose: Show progress
        risk_controller: Optional trained RiskController for dynamic position sizing and stops

    Returns:
        BacktestResult with performance data
    """
    device = torch.device(device)

    # Move models to device and eval mode
    vae = vae.to(device).eval()
    mdn_rnn = mdn_rnn.to(device).eval()
    controller = controller.to(device).eval()

    if risk_controller is not None:
        risk_controller = risk_controller.to(device).eval()
        logger.info("Using Risk Controller for dynamic position sizing and stops")

    # Check if data has day boundary info
    has_day_info = 'is_last_bar_of_day' in data.columns

    # Create environment
    env = TradingEnv(
        data=data.copy(),
        initial_balance=initial_balance,
        position_size=position_size,
        point_value=point_value,
        observation_columns=observation_columns,
        close_position_eod=close_position_eod and has_day_info,
        transaction_cost=transaction_cost
    )

    # Run backtest
    obs, _ = env.reset()
    done = False

    # Initialize tracking arrays
    net_worths = [initial_balance]
    positions = [0]
    actions_taken = []
    rewards_received = []

    # Initialize RNN hidden state
    hidden = mdn_rnn.initial_hidden(1, device)

    # Risk management state
    entry_price = None
    entry_net_worth = initial_balance  # Track net worth when position was entered
    current_position = 0  # 0=flat, 1=long, -1=short
    current_net_worth = initial_balance
    sl_hits = 0
    tp_hits = 0
    eod_closes = 0
    risk_controller_stops = 0  # Stops triggered by risk controller

    # Current risk params (updated by risk controller)
    current_pos_size_mult = 1.0
    current_sl_atr = 2.0
    current_tp_atr = 3.0

    step = 0
    with torch.no_grad():
        while not done:
            # Get current price for risk management
            current_price = data['close'].iloc[min(step, len(data) - 1)]

            # Check stop loss / take profit
            force_close = False
            if current_position != 0 and entry_price is not None:
                # Calculate unrealized P&L
                price_diff = current_price - entry_price
                unrealized_pnl = price_diff * current_position * position_size * point_value

                # Calculate P&L as percentage of portfolio at entry
                pnl_pct = (unrealized_pnl / float(entry_net_worth)) * 100

                # Check stop loss (percentage based)
                if stop_loss_pct is not None and pnl_pct <= -stop_loss_pct:
                    force_close = True
                    sl_hits += 1
                    if verbose:
                        logger.info(f"Step {step}: Stop loss hit ({pnl_pct:.2f}% of portfolio, ${unrealized_pnl:.2f})")

                # Check take profit (percentage based)
                if take_profit_pct is not None and pnl_pct >= take_profit_pct:
                    force_close = True
                    tp_hits += 1
                    if verbose:
                        logger.info(f"Step {step}: Take profit hit ({pnl_pct:.2f}% of portfolio, ${unrealized_pnl:.2f})")

            # Convert observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            # Encode observation
            z = vae.encode(obs_tensor, deterministic=True)

            # Get hidden state for controller
            h = mdn_rnn.get_hidden_state(hidden)

            # Get action from controller
            action = controller.get_action(z, h, deterministic=deterministic)
            action_int = action.item()

            # Get risk params from risk controller if available
            if risk_controller is not None:
                current_pos_size_mult, current_sl_atr, current_tp_atr = risk_controller.get_risk_params(z, h)

                # Check ATR-based stops if in position
                if current_position != 0 and entry_price is not None:
                    # Get current ATR
                    current_atr = 1.0
                    if 'ATR' in data.columns:
                        atr_val = data['ATR'].iloc[min(step, len(data) - 1)]
                        if atr_val > 0:
                            current_atr = float(atr_val)

                    price_diff = current_price - entry_price
                    if current_position == -1:  # Short position
                        price_diff = -price_diff

                    # ATR-based stop loss
                    if price_diff < -current_sl_atr * current_atr:
                        force_close = True
                        risk_controller_stops += 1
                        if verbose:
                            logger.debug(f"Step {step}: Risk controller stop loss ({price_diff:.2f} < -{current_sl_atr:.1f}*ATR)")

                    # ATR-based take profit
                    if price_diff > current_tp_atr * current_atr:
                        force_close = True
                        risk_controller_stops += 1
                        if verbose:
                            logger.debug(f"Step {step}: Risk controller take profit ({price_diff:.2f} > {current_tp_atr:.1f}*ATR)")

            # Override action if stop loss or take profit triggered
            if force_close:
                action_int = 2  # Hold/flat - this will close the position in the env

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated

            # Track position changes for risk management
            new_position = info['position']
            current_net_worth = info['net_worth']

            # Track EOD closes
            if info.get('eod_closed', False):
                eod_closes += 1
                if verbose:
                    logger.debug(f"Step {step}: EOD position close")

            if new_position != current_position:
                if new_position != 0:
                    # Entered new position
                    entry_price = current_price
                    entry_net_worth = current_net_worth
                else:
                    # Closed position
                    entry_price = None
                    entry_net_worth = current_net_worth
                current_position = new_position

            # Update RNN hidden state
            action_tensor = torch.tensor([action_int], dtype=torch.long, device=device)
            _, _, _, hidden = mdn_rnn(z, action_tensor, hidden)

            # Track results
            net_worths.append(info['net_worth'])
            positions.append(info['position'])
            actions_taken.append(action_int)
            rewards_received.append(reward)

            obs = next_obs
            step += 1

            if verbose and step % 10000 == 0:
                logger.info(f"Step {step}: Net Worth = ${info['net_worth']:.2f}")

    if verbose and (stop_loss_pct or take_profit_pct or close_position_eod or risk_controller is not None):
        logger.info(f"Risk management: {sl_hits} stop losses, {tp_hits} take profits, {eod_closes} EOD closes")
        if risk_controller is not None:
            logger.info(f"Risk controller stops: {risk_controller_stops}")

    # Convert to numpy
    net_worths = np.array(net_worths)
    positions = np.array(positions)
    actions_taken = np.array(actions_taken)
    rewards_received = np.array(rewards_received)

    # Log action distribution for debugging
    if verbose:
        unique, counts = np.unique(actions_taken, return_counts=True)
        action_names = {0: 'Long', 1: 'Short', 2: 'Flat/Close'}
        logger.info("Action distribution:")
        for action, count in zip(unique, counts):
            pct = count / len(actions_taken) * 100
            logger.info(f"  {action_names.get(action, action)}: {count} ({pct:.1f}%)")

        # Log position changes
        position_changes = np.sum(np.diff(positions) != 0)
        logger.info(f"Position changes: {position_changes}")

    # Calculate metrics
    metrics = calculate_metrics(net_worths, positions)

    # Get timestamps if available
    timestamps = None
    if hasattr(data, 'index'):
        timestamps = data.index.values[:len(net_worths)]

    if verbose:
        print_metrics(metrics)

    return BacktestResult(
        metrics=metrics,
        net_worths=net_worths,
        positions=positions,
        actions=actions_taken,
        rewards=rewards_received,
        timestamps=timestamps
    )


def compare_with_baseline(
    backtest_result: BacktestResult,
    data: pd.DataFrame,
    initial_balance: float = 10000.0
) -> Dict[str, TradeMetrics]:
    """
    Compare agent performance with baseline strategies.

    Args:
        backtest_result: Agent backtest result
        data: DataFrame with price data
        initial_balance: Starting balance

    Returns:
        Dictionary of strategy -> metrics
    """
    results = {'agent': backtest_result.metrics}

    # Buy and hold
    prices = data['close'].values
    bh_net_worths = initial_balance * (prices / prices[0])
    bh_positions = np.ones(len(prices))
    results['buy_hold'] = calculate_metrics(bh_net_worths, bh_positions)

    # Always short
    short_net_worths = initial_balance * (2 - prices / prices[0])
    short_positions = -np.ones(len(prices))
    results['always_short'] = calculate_metrics(short_net_worths, short_positions)

    return results


def print_comparison(results: Dict[str, TradeMetrics]):
    """Print comparison of strategies."""
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    print(f"{'Strategy':<15} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>8}")
    print("-" * 70)

    for name, metrics in results.items():
        print(f"{name:<15} {metrics.total_return*100:>9.2f}% {metrics.sharpe_ratio:>8.2f} "
              f"{metrics.max_drawdown*100:>7.2f}% {metrics.win_rate*100:>7.1f}% {metrics.num_trades:>8}")

    print("=" * 70 + "\n")
