#!/usr/bin/env python3
"""
Main Training Script for World Model.

This script implements the full training pipeline:
1. Phase 1: Collect random rollouts
2. Phase 2: Train VAE
3. Phase 3: Train MDN-RNN
4. Phase 4: Train Controller with CMA-ES (with parallel evaluation)

Usage:
    python -m algotrader3.scripts.train_world_model

Based on Ha & Schmidhuber's World Models (2018).
"""

import argparse
import logging
import multiprocessing as mp
import os
import sys
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

# Required for macOS multiprocessing with PyTorch
if sys.platform == 'darwin':
    mp.set_start_method('spawn', force=True)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algotrader3.data.data_loader import prepare_training_data, get_observation_columns, add_day_boundaries
from algotrader3.data.replay_buffer import ReplayBuffer
from algotrader3.envs.trading_env import TradingEnv
from algotrader3.models.vae import create_vae
from algotrader3.models.mdn_rnn import create_mdn_rnn
from algotrader3.models.controller import create_controller, Controller, MLPController
from algotrader3.training.train_vae import train_vae, encode_observations
from algotrader3.training.train_rnn import train_mdnrnn
from algotrader3.training.train_controller import train_controller

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def phase1_collect_rollouts(
    env: TradingEnv,
    num_episodes: int = 1,
    random_policy: bool = True
) -> ReplayBuffer:
    """
    Phase 1: Collect rollouts from the environment.

    For trading data, we typically just need 1 pass through the data
    since each step is a unique market state.

    Args:
        env: Trading environment
        num_episodes: Number of episodes to collect
        random_policy: Use random actions

    Returns:
        ReplayBuffer with collected transitions
    """
    logger.info(f"Phase 1: Collecting rollouts with random policy ({num_episodes} episode(s))")

    buffer = ReplayBuffer(capacity=1000000)
    total_steps = 0

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # Random action
            action = random.randint(0, 2)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.push(obs, action, reward, next_obs, done)

            obs = next_obs
            episode_reward += reward
            steps += 1

            # Progress logging every 50K steps
            if steps % 50000 == 0:
                logger.info(f"  Collected {steps} steps...")

        total_steps += steps
        logger.info(f"Episode {episode + 1}/{num_episodes}: "
                   f"Reward={episode_reward:.4f}, Steps={steps}")

    logger.info(f"Phase 1 complete: Collected {len(buffer)} transitions")
    return buffer


def main():
    parser = argparse.ArgumentParser(description="Train World Model for NQ Futures Trading")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--data', type=str, default='../data/NQ_2024_unix.csv',
                       help='Path to data file')
    parser.add_argument('--output', type=str, default='./checkpoints',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda, mps)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--skip-vae', action='store_true',
                       help='Skip VAE training (load existing)')
    parser.add_argument('--skip-rnn', action='store_true',
                       help='Skip RNN training (load existing)')
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # Determine device
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
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = {}

    # Config
    vae_config = config.get('vae', {})
    rnn_config = config.get('mdnrnn', {})
    latent_dim = vae_config.get('latent_dim', 32)
    hidden_dim = rnn_config.get('hidden_dim', 256)

    vae_path = os.path.join(args.output, 'vae.pt')
    rnn_path = os.path.join(args.output, 'mdn_rnn.pt')

    # Check if we can skip phases 1-3
    skip_vae = args.skip_vae and os.path.exists(vae_path)
    skip_rnn = args.skip_rnn and os.path.exists(rnn_path)

    # Cache path for processed data (avoids recalculating indicators)
    cache_path = os.path.join(args.output, 'train_data_cache.pkl')

    # Check if we can load cached data
    cache_valid = False
    if skip_vae and skip_rnn and os.path.exists(cache_path):
        import pickle
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        train_df = cache['train_df']
        obs_columns = cache['obs_columns']

        # Check if cache has day boundaries (required for daily episodes)
        if 'day_index' in train_df.columns and 'is_last_bar_of_day' in train_df.columns:
            cache_valid = True
        else:
            logger.warning("Cache is outdated (missing day boundaries). Regenerating...")
            os.remove(cache_path)

    if cache_valid:
        # Fast path: load cached data and models
        logger.info("=" * 50)
        logger.info("SKIPPING PHASES 1-3: Loading cached data and models")
        logger.info("=" * 50)

        from algotrader3.training.train_vae import load_vae
        from algotrader3.training.train_rnn import load_mdnrnn

        vae = load_vae(vae_path, device=device)
        mdn_rnn = load_mdnrnn(rnn_path, device=device)

        logger.info(f"Loaded cached data from {cache_path}")
        logger.info(f"Loaded VAE from {vae_path}")
        logger.info(f"Loaded MDN-RNN from {rnn_path}")

    if not cache_valid:
        # Need to load and process data
        logger.info("Loading and preparing data...")
        data_config = config.get('data', {})
        train_df, val_df, test_df, norm_params = prepare_training_data(
            file_path=args.data,
            train_ratio=data_config.get('train_ratio', 0.6),
            val_ratio=data_config.get('validation_ratio', 0.2),
            market_hours_only=data_config.get('market_hours_only', True),
            add_day_info=True  # Always add day info for daily episode support
        )

        # Get observation columns
        obs_columns = get_observation_columns()
        obs_columns = [c for c in obs_columns if c in train_df.columns]
        obs_dim = len(obs_columns)
        logger.info(f"Observation dimension: {obs_dim}")

        # Cache the processed data for future runs
        import pickle
        cache = {'train_df': train_df, 'obs_columns': obs_columns}
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        logger.info(f"Cached processed data to {cache_path}")

        if skip_vae and skip_rnn:
            # Models exist but we needed to regenerate cache
            logger.info("=" * 50)
            logger.info("SKIPPING PHASES 1-3: Loading existing VAE and MDN-RNN")
            logger.info("=" * 50)

            from algotrader3.training.train_vae import load_vae
            from algotrader3.training.train_rnn import load_mdnrnn

            vae = load_vae(vae_path, device=device)
            mdn_rnn = load_mdnrnn(rnn_path, device=device)

            logger.info(f"Loaded VAE from {vae_path}")
            logger.info(f"Loaded MDN-RNN from {rnn_path}")

        else:
            # Need to train VAE and/or RNN

            # Create environment
            logger.info("Creating trading environment...")
            env = TradingEnv(
                data=train_df.copy(),
                initial_balance=config.get('environment', {}).get('initial_balance', 10000.0),
                position_size=config.get('environment', {}).get('position_size', 1),
                observation_columns=obs_columns
            )

            # Phase 1: Collect rollouts
            logger.info("=" * 50)
            logger.info("PHASE 1: Collecting Random Rollouts")
            logger.info("=" * 50)

            buffer = phase1_collect_rollouts(env, num_episodes=1)
            all_data = buffer.get_all_data()

            # Phase 2: Train VAE
            logger.info("=" * 50)
            logger.info("PHASE 2: Training VAE")
            logger.info("=" * 50)

            if skip_vae:
                logger.info("Loading existing VAE...")
                from algotrader3.training.train_vae import load_vae
                vae = load_vae(vae_path, device=device)
            else:
                vae, vae_history = train_vae(
                    observations=all_data['obs'],
                    obs_dim=obs_dim,
                    latent_dim=latent_dim,
                    hidden_dims=vae_config.get('hidden_dims', [256, 128]),
                    beta=vae_config.get('beta', 0.001),
                    learning_rate=vae_config.get('learning_rate', 1e-3),
                    batch_size=vae_config.get('batch_size', 256),
                    epochs=vae_config.get('epochs', 10),
                    device=device,
                    save_path=vae_path
                )

            # Encode all observations
            logger.info("Encoding observations to latent space...")
            latents = encode_observations(vae, all_data['obs'], device=device)
            logger.info(f"Encoded {len(latents)} observations to {latent_dim}D latent space")

            # Phase 3: Train MDN-RNN
            logger.info("=" * 50)
            logger.info("PHASE 3: Training MDN-RNN")
            logger.info("=" * 50)

            if skip_rnn:
                logger.info("Loading existing MDN-RNN...")
                from algotrader3.training.train_rnn import load_mdnrnn
                mdn_rnn = load_mdnrnn(rnn_path, device=device)
            else:
                mdn_rnn, rnn_history = train_mdnrnn(
                    latents=latents,
                    actions=all_data['actions'],
                    latent_dim=latent_dim,
                    action_dim=3,
                    hidden_dim=hidden_dim,
                    num_layers=rnn_config.get('num_layers', 1),
                    num_mixtures=rnn_config.get('num_mixtures', 5),
                    learning_rate=rnn_config.get('learning_rate', 1e-3),
                    batch_size=rnn_config.get('batch_size', 32),
                    sequence_length=rnn_config.get('sequence_length', 50),
                    epochs=rnn_config.get('epochs', 20),
                    device=device,
                    save_path=rnn_path
                )

    # Phase 4: Train Controller with CMA-ES
    logger.info("=" * 50)
    logger.info("PHASE 4: Training Controller with CMA-ES")
    logger.info("=" * 50)

    cmaes_config = config.get('cmaes', {})
    controller_config = config.get('controller', {})
    env_config = config.get('environment', {})
    daily_episodes = cmaes_config.get('daily_episodes', True)
    close_position_eod = cmaes_config.get('close_position_eod', True)
    transaction_cost = env_config.get('transaction_cost', 0.0)
    use_mlp = controller_config.get('use_mlp', False)
    mlp_hidden = controller_config.get('mlp_hidden', 64)

    # Create fresh environment for controller training
    controller_env = TradingEnv(
        data=train_df.copy(),
        initial_balance=env_config.get('initial_balance', 10000.0),
        observation_columns=obs_columns,
        daily_episodes=daily_episodes,
        close_position_eod=close_position_eod,
        transaction_cost=transaction_cost
    )

    controller_path = os.path.join(args.output, 'controller.pt')

    controller, controller_history = train_controller(
        vae=vae,
        mdn_rnn=mdn_rnn,
        env=controller_env,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        action_dim=3,
        use_mlp=use_mlp,
        mlp_hidden=mlp_hidden,
        population_size=cmaes_config.get('population_size', 32),
        sigma=cmaes_config.get('sigma', 0.5),
        generations=cmaes_config.get('generations', 100),
        eval_episodes=cmaes_config.get('eval_episodes', 4),
        max_steps_per_episode=cmaes_config.get('max_steps_per_episode', 100),
        daily_episodes=daily_episodes,
        close_position_eod=close_position_eod,
        transaction_cost=transaction_cost,
        patience=cmaes_config.get('patience', 10),
        device=device,
        save_path=controller_path
    )

    logger.info("=" * 50)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"VAE saved to: {vae_path}")
    logger.info(f"MDN-RNN saved to: {rnn_path}")
    logger.info(f"Controller saved to: {controller_path}")
    logger.info(f"Best controller Profit: {controller_history['best_profit'][-1]:.4f}")


if __name__ == "__main__":
    main()
