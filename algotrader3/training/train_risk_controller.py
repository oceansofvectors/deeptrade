"""
Risk Controller Training with CMA-ES.

Trains a separate risk controller that outputs:
- Position size multiplier
- Stop loss (ATR multiples)
- Take profit (ATR multiples)

Uses Calmar Ratio (Return / Max Drawdown) as fitness.
Better suited for high-frequency data than Sharpe Ratio.
"""

import logging
import os
import time
from typing import Tuple, Optional, Dict
import multiprocessing as mp

import numpy as np
import torch
import cma
from tqdm import tqdm

from ..models.vae import VAE
from ..models.mdn_rnn import MDNRNN, create_mdn_rnn
from ..models.controller import Controller, create_controller
from ..models.risk_controller import RiskController, create_risk_controller
from ..envs.trading_env import TradingEnv

logger = logging.getLogger(__name__)

# Global variables for worker processes
_worker_mdn_rnn = None
_worker_trade_controller = None
_worker_risk_controller = None
_worker_env = None
_worker_config = None
_worker_encoded_obs = None


def _init_risk_worker(
    encoded_obs: np.ndarray,
    rnn_state: dict,
    rnn_config: dict,
    trade_controller_state: dict,
    trade_controller_config: dict,
    risk_controller_config: dict,
    env_data,
    env_config: dict
):
    """Initialize worker process with models."""
    global _worker_mdn_rnn, _worker_trade_controller, _worker_risk_controller
    global _worker_env, _worker_config, _worker_encoded_obs

    _worker_encoded_obs = encoded_obs

    # Create MDN-RNN
    _worker_mdn_rnn = create_mdn_rnn(
        latent_dim=rnn_config['latent_dim'],
        action_dim=rnn_config['action_dim'],
        hidden_dim=rnn_config['hidden_dim'],
        num_layers=rnn_config.get('num_layers', 1),
        num_mixtures=rnn_config.get('num_mixtures', 5)
    )
    _worker_mdn_rnn.load_state_dict(rnn_state)
    _worker_mdn_rnn.eval()

    # Create Trade Controller (frozen - we use this for trade decisions)
    _worker_trade_controller = create_controller(
        latent_dim=trade_controller_config['latent_dim'],
        hidden_dim=trade_controller_config['hidden_dim'],
        action_dim=trade_controller_config['action_dim']
    )
    _worker_trade_controller.load_state_dict(trade_controller_state)
    _worker_trade_controller.eval()

    # Create Risk Controller (this is what we're training)
    _worker_risk_controller = create_risk_controller(
        latent_dim=risk_controller_config['latent_dim'],
        hidden_dim=risk_controller_config['hidden_dim']
    )
    _worker_risk_controller.eval()

    # Create Environment
    _worker_env = TradingEnv(
        data=env_data.copy(),
        initial_balance=env_config.get('initial_balance', 10000.0),
        position_size=env_config.get('position_size', 1),
        point_value=env_config.get('point_value', 2.0),
        observation_columns=env_config.get('observation_columns')
    )

    _worker_config = {
        'num_episodes': env_config.get('num_episodes', 4),
        'max_steps_per_episode': env_config.get('max_steps_per_episode', 10000),
        'base_position_size': env_config.get('position_size', 1),
        'point_value': env_config.get('point_value', 2.0)
    }


def _calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """Calculate maximum drawdown from equity curve."""
    if len(equity_curve) < 2:
        return 0.0

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max
    return float(np.max(drawdowns))


def _evaluate_risk_worker(params: np.ndarray) -> float:
    """
    Evaluate risk controller with given parameters.

    Uses Calmar Ratio (Return / Max Drawdown) as fitness.
    Better suited for high-frequency data than Sharpe.
    Trade decisions come from frozen trade controller.
    Risk controller only manages position sizing and stops.

    Returns negative Calmar (CMA-ES minimizes).
    """
    global _worker_mdn_rnn, _worker_trade_controller, _worker_risk_controller
    global _worker_env, _worker_config, _worker_encoded_obs

    # Set risk controller parameters
    _worker_risk_controller.set_params(params)

    all_equity_curves = []
    num_episodes = _worker_config['num_episodes']
    max_steps = _worker_config['max_steps_per_episode']
    base_position = _worker_config['base_position_size']
    point_value = _worker_config['point_value']
    initial_balance = _worker_env.initial_balance
    device = torch.device('cpu')

    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = _worker_env.reset(options={'random_start': True, 'max_steps': max_steps})
            done = False
            current_step = _worker_env.current_step
            current_position = _worker_env.position
            steps_taken = 0

            hidden = _worker_mdn_rnn.initial_hidden(1, device)

            # Track for stop loss / take profit
            entry_price = None
            current_atr = 1.0  # Default ATR

            # Track equity curve for this episode (ensure float)
            episode_equity = [float(initial_balance)]
            last_env_net_worth = float(initial_balance)

            while not done and steps_taken < max_steps:
                # Get latent vector
                pos_idx = current_position + 1
                z_np = _worker_encoded_obs[current_step, pos_idx]
                z = torch.tensor(z_np, dtype=torch.float32, device=device).unsqueeze(0)

                # Get hidden state
                h = _worker_mdn_rnn.get_hidden_state(hidden)

                # Get trade action from frozen trade controller
                action = _worker_trade_controller.get_action(z, h, deterministic=True)
                action_int = action.item()

                # Get risk params from risk controller
                pos_size_mult, sl_atr, tp_atr = _worker_risk_controller.get_risk_params(z, h)

                # Get current ATR from environment data
                if 'ATR' in _worker_env.data.columns:
                    current_atr = float(_worker_env.data.loc[current_step, 'ATR'])
                    if current_atr <= 0:
                        current_atr = 1.0

                # Get current price
                current_price = float(_worker_env.data.loc[current_step, 'close'])

                # Check stop loss / take profit if in position
                force_close = False
                if current_position != 0 and entry_price is not None:
                    price_diff = current_price - entry_price
                    if current_position == -1:  # Short
                        price_diff = -price_diff

                    # Stop loss check (in ATR)
                    if price_diff < -sl_atr * current_atr:
                        force_close = True

                    # Take profit check (in ATR)
                    if price_diff > tp_atr * current_atr:
                        force_close = True

                if force_close:
                    action_int = 2  # Close position

                # Step environment
                next_obs, reward, terminated, truncated, info = _worker_env.step(action_int)
                done = terminated or truncated
                steps_taken += 1

                # Track equity (apply position size multiplier)
                # Scale the change in net worth by position size
                # Compare env net worth to previous env net worth (not our scaled equity)
                base_net_worth = float(info['net_worth'])
                pnl_this_step = base_net_worth - last_env_net_worth
                adjusted_net_worth = episode_equity[-1] + (pnl_this_step * pos_size_mult)
                episode_equity.append(adjusted_net_worth)
                last_env_net_worth = base_net_worth

                # Track entry price for stop/take profit
                new_position = info['position']
                if new_position != current_position:
                    if new_position != 0:
                        entry_price = current_price
                    else:
                        entry_price = None

                current_step = _worker_env.current_step
                current_position = new_position

                # Update RNN
                action_tensor = torch.tensor([action_int], dtype=torch.long, device=device)
                _, _, _, hidden = _worker_mdn_rnn(z, action_tensor, hidden)

            all_equity_curves.append(np.array(episode_equity))

    # Calculate Calmar Ratio across all episodes
    if not all_equity_curves:
        return 0.0

    # Concatenate all equity curves (normalized to start at 1.0)
    total_return = 0.0
    max_drawdown = 0.0

    for equity in all_equity_curves:
        if len(equity) < 2:
            continue

        # Calculate return for this episode
        episode_return = (equity[-1] - equity[0]) / equity[0]
        total_return += episode_return

        # Calculate max drawdown for this episode
        episode_dd = _calculate_max_drawdown(equity)
        max_drawdown = max(max_drawdown, episode_dd)

    # Average return across episodes
    avg_return = total_return / len(all_equity_curves)

    # Calmar Ratio = Return / Max Drawdown
    # Add small epsilon to avoid division by zero
    calmar = avg_return / max(max_drawdown, 0.01)

    # Return negative because CMA-ES minimizes
    return -calmar


def train_risk_controller(
    vae: VAE,
    mdn_rnn: MDNRNN,
    trade_controller: Controller,
    env: TradingEnv,
    encoded_obs: np.ndarray,
    latent_dim: int = 12,
    hidden_dim: int = 256,
    population_size: int = 32,
    sigma: float = 0.5,
    generations: int = 100,
    eval_episodes: int = 4,
    max_steps_per_episode: int = 10000,
    patience: int = 15,
    num_workers: Optional[int] = None,
    save_path: Optional[str] = None
) -> Tuple[RiskController, Dict]:
    """
    Train risk controller using CMA-ES.

    Args:
        vae: Trained VAE
        mdn_rnn: Trained MDN-RNN
        trade_controller: Trained trade controller (frozen)
        env: Trading environment
        encoded_obs: Pre-encoded observations
        latent_dim: Latent dimension
        hidden_dim: RNN hidden dimension
        population_size: CMA-ES population size
        sigma: Initial step size
        generations: Maximum generations
        eval_episodes: Episodes per evaluation
        max_steps_per_episode: Max steps per episode
        patience: Early stopping patience
        num_workers: Number of parallel workers
        save_path: Path to save controller

    Returns:
        Tuple of (trained risk controller, training history)
    """
    logger.info("Starting Risk Controller CMA-ES training")
    logger.info("Trade Controller is FROZEN - only training risk parameters")

    if num_workers is None:
        num_workers = mp.cpu_count()
    logger.info(f"Using {num_workers} parallel workers")

    # Create risk controller
    risk_controller = create_risk_controller(latent_dim=latent_dim, hidden_dim=hidden_dim)
    num_params = risk_controller.get_param_count()
    logger.info(f"Risk Controller has {num_params} parameters")

    # Prepare configs for workers
    env_data = env.data.copy()
    env_config = {
        'initial_balance': float(env.initial_balance),
        'position_size': env.position_size,
        'point_value': float(env.point_value),
        'observation_columns': env.observation_columns,
        'num_episodes': eval_episodes,
        'max_steps_per_episode': max_steps_per_episode
    }

    rnn_state = {k: v.cpu() for k, v in mdn_rnn.state_dict().items()}
    rnn_config = {
        'latent_dim': mdn_rnn.latent_dim,
        'action_dim': mdn_rnn.action_dim,
        'hidden_dim': mdn_rnn.hidden_dim,
        'num_layers': mdn_rnn.num_layers,
        'num_mixtures': mdn_rnn.num_mixtures
    }

    trade_controller_state = {k: v.cpu() for k, v in trade_controller.state_dict().items()}
    trade_controller_config = {
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'action_dim': 3
    }

    risk_controller_config = {
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim
    }

    # Initialize CMA-ES
    initial_params = risk_controller.get_params()
    es = cma.CMAEvolutionStrategy(
        initial_params,
        sigma,
        {'popsize': population_size, 'verbose': -9}
    )

    # Training history
    history = {
        'generation': [],
        'best_calmar': [],
        'mean_calmar': [],
        'sigma': []
    }

    generation = 0
    best_calmar = float('-inf')
    best_params = None
    no_improvement_count = 0
    start_time = time.time()

    # Create worker pool
    logger.info("Initializing worker pool...")
    pool = mp.Pool(
        processes=num_workers,
        initializer=_init_risk_worker,
        initargs=(encoded_obs, rnn_state, rnn_config,
                  trade_controller_state, trade_controller_config,
                  risk_controller_config, env_data, env_config)
    )

    try:
        pbar = tqdm(
            total=generations * population_size,
            desc="Risk Controller Training",
            unit="eval"
        )

        while not es.stop() and generation < generations:
            solutions = es.ask()
            fitnesses = pool.map(_evaluate_risk_worker, solutions)

            es.tell(solutions, fitnesses)

            pbar.update(population_size)

            # Track best
            min_fitness = min(fitnesses)
            current_best = -min_fitness

            if current_best > best_calmar + 1e-6:
                best_calmar = current_best
                best_idx = fitnesses.index(min_fitness)
                best_params = solutions[best_idx].copy()
                no_improvement_count = 0
                improved = "↑"
            else:
                no_improvement_count += 1
                improved = " "

            mean_calmar = -np.mean(fitnesses)
            history['generation'].append(generation)
            history['best_calmar'].append(best_calmar)
            history['mean_calmar'].append(mean_calmar)
            history['sigma'].append(es.sigma)

            pbar.set_postfix_str(
                f"Gen {generation+1}/{generations} | "
                f"Calmar: {best_calmar:.3f}{improved} | "
                f"Mean: {mean_calmar:.3f} | "
                f"Stall: {no_improvement_count}/{patience}"
            )

            if no_improvement_count >= patience:
                pbar.write(f"Early stopping: no improvement for {patience} generations")
                break

            generation += 1

        pbar.close()

    finally:
        pool.close()
        pool.join()

    total_time = time.time() - start_time
    logger.info(f"Training time: {total_time/60:.1f} minutes")

    if best_params is not None:
        risk_controller.set_params(best_params)

    logger.info(f"Training complete. Best Calmar: {best_calmar:.4f}")

    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': risk_controller.state_dict(),
            'params': best_params,
            'config': {
                'latent_dim': latent_dim,
                'hidden_dim': hidden_dim
            },
            'history': history,
            'best_calmar': best_calmar
        }, save_path)
        logger.info(f"Risk Controller saved to {save_path}")

    return risk_controller, history


def load_risk_controller(path: str, device: str = 'cpu') -> RiskController:
    """Load a trained risk controller from disk."""
    device = torch.device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = checkpoint['config']
    risk_controller = create_risk_controller(
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim']
    )

    if 'model_state_dict' in checkpoint:
        risk_controller.load_state_dict(checkpoint['model_state_dict'])
    elif 'params' in checkpoint:
        risk_controller.set_params(checkpoint['params'])

    risk_controller.to(device)
    risk_controller.eval()

    logger.info(f"Loaded Risk Controller from {path}")
    return risk_controller
