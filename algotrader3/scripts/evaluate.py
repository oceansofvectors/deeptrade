#!/usr/bin/env python3
"""
Evaluation Script for World Model.

Runs backtest on test data and compares with baselines.

Usage:
    python -m algotrader3.scripts.evaluate --checkpoint ./checkpoints
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algotrader3.data.data_loader import prepare_training_data, get_observation_columns
from algotrader3.training.train_vae import load_vae
from algotrader3.training.train_rnn import load_mdnrnn
from algotrader3.training.train_controller import load_controller
from algotrader3.training.train_risk_controller import load_risk_controller
from algotrader3.evaluation.backtest import backtest, compare_with_baseline, print_comparison

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_results(result, eval_df, save_path: str = None):
    """Plot backtest results with comprehensive charts."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    initial_balance = result.net_worths[0]

    # 1. Equity curve with buy & hold comparison
    ax1 = axes[0]
    steps = range(len(result.net_worths))
    ax1.plot(steps, result.net_worths, label='Agent', color='blue', linewidth=1.5)

    # Buy & hold comparison
    prices = eval_df['close'].values[:len(result.net_worths)]
    buy_hold = initial_balance * (prices / prices[0])
    ax1.plot(steps, buy_hold, label='Buy & Hold', color='gray', linestyle='--', alpha=0.7)

    ax1.axhline(y=initial_balance, color='black', linestyle=':', alpha=0.5, label='Initial')
    ax1.set_ylabel('Net Worth ($)')
    ax1.set_title('Equity Curve: Agent vs Buy & Hold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(result.net_worths))

    # 2. Drawdown chart
    ax2 = axes[1]
    running_max = np.maximum.accumulate(result.net_worths)
    drawdown = (result.net_worths - running_max) / running_max * 100
    ax2.fill_between(steps, drawdown, 0, color='red', alpha=0.4)
    ax2.plot(steps, drawdown, color='darkred', linewidth=0.5)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title(f'Drawdown (Max: {min(drawdown):.1f}%)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(result.net_worths))

    # 3. Position over time
    ax3 = axes[2]
    positions = result.positions[:len(result.net_worths)]
    ax3.fill_between(
        steps[:len(positions)], positions, 0,
        where=np.array(positions) > 0,
        color='green', alpha=0.6, label='Long'
    )
    ax3.fill_between(
        steps[:len(positions)], positions, 0,
        where=np.array(positions) < 0,
        color='red', alpha=0.6, label='Short'
    )
    ax3.set_ylabel('Position')
    ax3.set_title('Position Over Time')
    ax3.legend(loc='upper right')
    ax3.set_ylim(-1.5, 1.5)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, len(result.net_worths))

    # 4. Daily returns distribution
    ax4 = axes[3]
    # Calculate daily returns (approximate by grouping every ~78 bars for 5-min data)
    bars_per_day = 78
    daily_returns = []
    for i in range(0, len(result.net_worths) - bars_per_day, bars_per_day):
        start_val = result.net_worths[i]
        end_val = result.net_worths[i + bars_per_day]
        if start_val > 0:
            daily_returns.append((end_val - start_val) / start_val * 100)

    if daily_returns:
        colors = ['green' if r > 0 else 'red' for r in daily_returns]
        ax4.bar(range(len(daily_returns)), daily_returns, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.axhline(y=np.mean(daily_returns), color='blue', linestyle='--',
                    label=f'Avg: {np.mean(daily_returns):.2f}%')
        ax4.set_ylabel('Return (%)')
        ax4.set_xlabel('Trading Day')
        ax4.set_title(f'Daily Returns ({len([r for r in daily_returns if r > 0])}/{len(daily_returns)} winning days)')
        ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate World Model on test data")
    parser.add_argument('--checkpoint', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--data', type=str, default='../data/NQ_2024_unix.csv',
                       help='Path to data file')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split to evaluate on')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for inference')
    parser.add_argument('--stop-loss', type=float, default=None,
                       help='Stop loss as percentage of portfolio (e.g., 5 for 5%%)')
    parser.add_argument('--take-profit', type=float, default=None,
                       help='Take profit as percentage of portfolio (e.g., 10 for 10%%)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting (plots shown by default)')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Path to save plot')
    parser.add_argument('--market-hours-only', action='store_true',
                       help='Filter to market hours only (9:30-16:00 ET)')
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    env_config = config.get('environment', {})
    initial_balance = env_config.get('initial_balance', 10000.0)
    position_size = env_config.get('position_size', 1)
    point_value = env_config.get('point_value', 20.0)
    close_position_eod = env_config.get('close_position_eod', False)
    transaction_cost = env_config.get('transaction_cost', 0.0)

    logger.info(f"Using point_value: ${point_value}/point (position_size: {position_size}, transaction_cost: ${transaction_cost})")

    # Load models
    logger.info("Loading trained models...")

    vae_path = os.path.join(args.checkpoint, 'vae.pt')
    rnn_path = os.path.join(args.checkpoint, 'mdn_rnn.pt')
    controller_path = os.path.join(args.checkpoint, 'controller.pt')

    for path, name in [(vae_path, 'VAE'), (rnn_path, 'MDN-RNN'), (controller_path, 'Controller')]:
        if not os.path.exists(path):
            logger.error(f"{name} not found at {path}")
            return

    vae = load_vae(vae_path, device=args.device)
    mdn_rnn = load_mdnrnn(rnn_path, device=args.device)
    controller = load_controller(controller_path, device=args.device)

    # Load risk controller if available
    risk_controller = None
    risk_controller_path = os.path.join(args.checkpoint, 'risk_controller.pt')
    risk_config = config.get('risk_controller', {})

    if risk_config.get('enabled', False) and os.path.exists(risk_controller_path):
        risk_controller = load_risk_controller(risk_controller_path, device=args.device)
        logger.info(f"Risk Controller loaded from {risk_controller_path}")
    elif risk_config.get('enabled', False):
        logger.warning("Risk Controller enabled but not found - running without risk management")

    # Load data
    logger.info(f"Loading data... (market_hours_only={args.market_hours_only})")
    train_df, val_df, test_df, _ = prepare_training_data(
        file_path=args.data,
        market_hours_only=args.market_hours_only
    )

    # Select split
    if args.split == 'train':
        eval_df = train_df
    elif args.split == 'val':
        eval_df = val_df
    else:
        eval_df = test_df

    logger.info(f"Evaluating on {args.split} set: {len(eval_df)} samples")

    # Get observation columns
    obs_columns = get_observation_columns()
    obs_columns = [c for c in obs_columns if c in eval_df.columns]

    # Run backtest
    logger.info("Running backtest...")
    if args.stop_loss:
        logger.info(f"Stop loss: {args.stop_loss}% of portfolio")
    if args.take_profit:
        logger.info(f"Take profit: {args.take_profit}% of portfolio")

    result = backtest(
        vae=vae,
        mdn_rnn=mdn_rnn,
        controller=controller,
        data=eval_df,
        initial_balance=initial_balance,
        position_size=position_size,
        observation_columns=obs_columns,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        close_position_eod=close_position_eod,
        point_value=point_value,
        transaction_cost=transaction_cost,
        device=args.device,
        verbose=True,
        risk_controller=risk_controller
    )

    # Compare with baselines
    logger.info("Comparing with baseline strategies...")
    comparison = compare_with_baseline(result, eval_df)
    print_comparison(comparison)

    # Plot results (default: show plots)
    if not args.no_plot or args.save_plot:
        plot_results(result, eval_df, args.save_plot)


if __name__ == "__main__":
    main()
