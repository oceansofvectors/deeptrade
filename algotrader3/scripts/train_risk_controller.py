#!/usr/bin/env python3
"""
Risk Controller Training Script.

Trains a separate risk controller after the main World Model is trained.
The risk controller manages:
- Position sizing
- Stop loss levels (ATR-based)
- Take profit levels (ATR-based)

Optimized with Sharpe Ratio as the fitness function.

Usage:
    python -m algotrader3.scripts.train_risk_controller

Prerequisites:
    - VAE, MDN-RNN, and Controller must already be trained
    - Run train_world_model.py first
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algotrader3.data.data_loader import prepare_training_data, get_observation_columns
from algotrader3.envs.trading_env import TradingEnv
from algotrader3.training.train_vae import load_vae
from algotrader3.training.train_rnn import load_mdnrnn
from algotrader3.training.train_controller import load_controller, pre_encode_all_observations
from algotrader3.training.train_risk_controller import train_risk_controller

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Risk Controller")
    parser.add_argument('--checkpoint', type=str, default='./checkpoints',
                       help='Checkpoint directory with trained models')
    parser.add_argument('--data', type=str, default='../data/NQ_2024_unix.csv',
                       help='Path to data file')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda, mps)')
    parser.add_argument('--generations', type=int, default=100,
                       help='Number of CMA-ES generations')
    parser.add_argument('--population', type=int, default=32,
                       help='Population size')
    parser.add_argument('--episodes', type=int, default=4,
                       help='Episodes per evaluation')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Load config
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Check for trained models
    vae_path = os.path.join(args.checkpoint, 'vae.pt')
    rnn_path = os.path.join(args.checkpoint, 'mdn_rnn.pt')
    controller_path = os.path.join(args.checkpoint, 'controller.pt')

    for path, name in [(vae_path, 'VAE'), (rnn_path, 'MDN-RNN'), (controller_path, 'Controller')]:
        if not os.path.exists(path):
            logger.error(f"{name} not found at {path}")
            logger.error("Train the main World Model first: python -m algotrader3.scripts.train_world_model")
            return

    # Load trained models
    logger.info("Loading trained models...")
    vae = load_vae(vae_path, device=device)
    mdn_rnn = load_mdnrnn(rnn_path, device=device)
    trade_controller = load_controller(controller_path, device=device)

    latent_dim = vae.latent_dim
    hidden_dim = mdn_rnn.hidden_dim

    logger.info(f"VAE latent_dim: {latent_dim}")
    logger.info(f"RNN hidden_dim: {hidden_dim}")

    # Load data
    logger.info("Loading training data...")
    data_config = config.get('data', {})
    train_df, val_df, test_df, _ = prepare_training_data(
        file_path=args.data,
        market_hours_only=data_config.get('market_hours_only', False)
    )

    # Get observation columns
    obs_columns = get_observation_columns()
    obs_columns = [c for c in obs_columns if c in train_df.columns]

    # Create environment
    env_config = config.get('environment', {})
    env = TradingEnv(
        data=train_df.copy(),
        initial_balance=env_config.get('initial_balance', 10000.0),
        position_size=env_config.get('position_size', 1),
        point_value=env_config.get('point_value', 2.0),
        observation_columns=obs_columns
    )

    # Pre-encode observations
    logger.info("Pre-encoding observations...")
    encoded_obs = pre_encode_all_observations(
        vae=vae,
        data=train_df,
        observation_columns=obs_columns,
        device=device
    )

    # Train risk controller
    logger.info("=" * 50)
    logger.info("TRAINING RISK CONTROLLER")
    logger.info("=" * 50)

    cmaes_config = config.get('cmaes', {})
    risk_controller_path = os.path.join(args.checkpoint, 'risk_controller.pt')

    risk_controller, history = train_risk_controller(
        vae=vae,
        mdn_rnn=mdn_rnn,
        trade_controller=trade_controller,
        env=env,
        encoded_obs=encoded_obs,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        population_size=args.population,
        sigma=cmaes_config.get('sigma', 0.5),
        generations=args.generations,
        eval_episodes=args.episodes,
        max_steps_per_episode=cmaes_config.get('max_steps_per_episode', 10000),
        patience=args.patience,
        save_path=risk_controller_path
    )

    logger.info("=" * 50)
    logger.info("RISK CONTROLLER TRAINING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Saved to: {risk_controller_path}")
    logger.info(f"Best Sharpe: {history['best_sharpe'][-1]:.4f}")


if __name__ == "__main__":
    main()
