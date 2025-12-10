"""
Controller Training with CMA-ES.

Phase 4 of World Model training:
Train the controller using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

The controller is trained by optimizing Sortino Ratio (risk-adjusted returns),
using the world model (VAE + MDN-RNN) for state representation.

Features:
- PRE-ENCODED observations: VAE encoding done once upfront, not per-step
  This provides ~10-50x speedup vs encoding in the training loop
- Parallel evaluation using multiprocessing for ~8x speedup
- Sortino Ratio fitness: optimizes risk-adjusted returns (penalizes downside only)
- Early stopping when Sortino plateaus
"""

import logging
import os
import time
from typing import Tuple, Optional, Dict, List
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import cma
from tqdm import tqdm

from ..models.vae import VAE
from ..models.mdn_rnn import MDNRNN, create_mdn_rnn
from ..models.controller import Controller, create_controller
from ..envs.trading_env import TradingEnv

logger = logging.getLogger(__name__)


# Global variables for worker processes (avoids pickling large objects)
_worker_mdn_rnn = None
_worker_controller = None
_worker_env = None
_worker_config = None
_worker_encoded_obs = None  # Pre-encoded latent vectors: [num_steps, 3, latent_dim]


def pre_encode_all_observations(
    vae: VAE,
    data: pd.DataFrame,
    observation_columns: list,
    device: str = 'cpu',
    batch_size: int = 1024
) -> np.ndarray:
    """
    Pre-encode all observations for all 3 position states.

    This eliminates VAE inference from the training loop, providing
    massive speedup (VAE is called once per step instead of millions of times).

    Args:
        vae: Trained VAE model
        data: DataFrame with observation data
        observation_columns: List of columns to use as observations
        device: Device for encoding
        batch_size: Batch size for encoding

    Returns:
        encoded_obs: Array of shape [num_steps, 3, latent_dim]
                     Index with encoded_obs[step, position+1] where position is -1, 0, 1
    """
    logger.info("Pre-encoding all observations (this eliminates VAE from training loop)...")

    vae = vae.to(device).eval()
    num_steps = len(data)
    latent_dim = vae.latent_dim

    # Prepare base observations (without position)
    base_obs_cols = [c for c in observation_columns if c != 'position']
    base_obs = data[base_obs_cols].values.astype(np.float32)

    # Clip to reasonable bounds (same as TradingEnv)
    base_obs = np.clip(base_obs, -10.0, 10.0)

    # Pre-allocate output array: [num_steps, 3 positions, latent_dim]
    # Position mapping: -1 -> index 0, 0 -> index 1, 1 -> index 2
    encoded_obs = np.zeros((num_steps, 3, latent_dim), dtype=np.float32)

    with torch.no_grad():
        for pos_idx, position in enumerate([-1, 0, 1]):
            logger.info(f"  Encoding position {position}...")

            # Add position column
            full_obs = np.column_stack([base_obs, np.full(num_steps, position, dtype=np.float32)])

            # Encode in batches
            for i in range(0, num_steps, batch_size):
                batch = full_obs[i:i+batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
                z = vae.encode(batch_tensor, deterministic=True)
                encoded_obs[i:i+batch_size, pos_idx] = z.cpu().numpy()

    logger.info(f"Pre-encoded {num_steps} steps x 3 positions = {num_steps * 3} latent vectors")
    logger.info(f"Memory usage: {encoded_obs.nbytes / 1024 / 1024:.1f} MB")

    return encoded_obs


def _init_worker(encoded_obs: np.ndarray, rnn_state: dict, rnn_config: dict,
                 controller_config: dict, env_data: pd.DataFrame, env_config: dict):
    """
    Initialize worker process with pre-encoded observations and models.

    Called once per worker to set up global state.
    No VAE needed - we use pre-encoded latent vectors for massive speedup.
    """
    global _worker_mdn_rnn, _worker_controller, _worker_env, _worker_config, _worker_encoded_obs

    # Store pre-encoded observations (no VAE inference needed!)
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

    # Create Controller
    _worker_controller = create_controller(
        latent_dim=controller_config['latent_dim'],
        hidden_dim=controller_config['hidden_dim'],
        action_dim=controller_config['action_dim'],
        use_mlp=controller_config.get('use_mlp', False),
        mlp_hidden=controller_config.get('mlp_hidden', 64)
    )
    _worker_controller.eval()

    # Create Environment with daily episode support
    _worker_env = TradingEnv(
        data=env_data.copy(),
        initial_balance=env_config.get('initial_balance', 10000.0),
        position_size=env_config.get('position_size', 1),
        observation_columns=env_config.get('observation_columns'),
        daily_episodes=env_config.get('daily_episodes', False),
        close_position_eod=env_config.get('close_position_eod', True),
        transaction_cost=env_config.get('transaction_cost', 0.0)
    )

    _worker_config = {
        'num_episodes': env_config.get('num_episodes', 4),
        'deterministic': env_config.get('deterministic', True),
        'max_steps_per_episode': env_config.get('max_steps_per_episode', 10000),
        'daily_episodes': env_config.get('daily_episodes', False)
    }


def _evaluate_worker_indexed(args: tuple) -> tuple:
    """Wrapper that returns index along with fitness for imap_unordered."""
    idx, params = args
    fitness = _evaluate_worker(params)
    return idx, fitness


def _evaluate_worker(params: np.ndarray) -> float:
    """
    Evaluate controller with given parameters in worker process.

    Uses total profit as fitness metric to maximize returns.

    Uses pre-encoded latent vectors for massive speedup (no VAE inference).

    Returns negative total reward (CMA-ES minimizes).
    """
    global _worker_mdn_rnn, _worker_controller, _worker_env, _worker_config, _worker_encoded_obs

    # Set controller parameters
    _worker_controller.set_params(params)

    total_reward = 0.0  # Sum of all rewards (log returns)
    num_episodes = _worker_config['num_episodes']
    deterministic = _worker_config['deterministic']
    daily_episodes = _worker_config.get('daily_episodes', False)
    max_steps = _worker_config.get('max_steps_per_episode', 10000)
    device = torch.device('cpu')  # Workers always use CPU

    with torch.no_grad():
        for _ in range(num_episodes):
            # Random start: random day if daily_episodes, else random position
            obs, info = _worker_env.reset(options={'random_start': True, 'max_steps': max_steps})
            done = False
            current_step = _worker_env.current_step
            current_position = _worker_env.position  # -1, 0, or 1
            steps_taken = 0

            # In daily mode, max_steps is determined by the day length
            # Otherwise use configured max_steps
            episode_max_steps = info.get('bars_remaining', max_steps) + 1 if daily_episodes else max_steps

            # Initialize RNN hidden state
            hidden = _worker_mdn_rnn.initial_hidden(1, device)

            while not done and steps_taken < episode_max_steps:
                # Lookup pre-encoded latent vector (FAST - no VAE inference!)
                # Position mapping: -1 -> 0, 0 -> 1, 1 -> 2
                pos_idx = current_position + 1
                z_np = _worker_encoded_obs[current_step, pos_idx]
                z = torch.tensor(z_np, dtype=torch.float32, device=device).unsqueeze(0)

                # Get hidden state for controller
                h = _worker_mdn_rnn.get_hidden_state(hidden)

                # Get action
                action = _worker_controller.get_action(z, h, deterministic=deterministic)
                action_int = action.item()

                # Step environment
                next_obs, reward, terminated, truncated, info = _worker_env.step(action_int)
                done = terminated or truncated
                steps_taken += 1

                # Update step and position for next lookup
                current_step = _worker_env.current_step
                current_position = info['position']

                # Accumulate reward (log returns)
                total_reward += reward

                # Update RNN hidden state
                action_tensor = torch.tensor([action_int], dtype=torch.long, device=device)
                _, _, _, hidden = _worker_mdn_rnn(z, action_tensor, hidden)

    # Return negative because CMA-ES minimizes
    # Higher profit = better, so we negate it
    return -total_reward


class ControllerEvaluator:
    """
    Evaluates controller performance in the trading environment.
    """

    def __init__(
        self,
        vae: VAE,
        mdn_rnn: MDNRNN,
        controller: Controller,
        env: TradingEnv,
        num_episodes: int = 16,
        deterministic: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize evaluator.

        Args:
            vae: Trained VAE
            mdn_rnn: Trained MDN-RNN
            controller: Controller to evaluate
            env: Trading environment
            num_episodes: Number of episodes to average
            deterministic: Use deterministic actions
            device: Device for inference
        """
        self.vae = vae
        self.mdn_rnn = mdn_rnn
        self.controller = controller
        self.env = env
        self.num_episodes = num_episodes
        self.deterministic = deterministic
        self.device = torch.device(device)

        # Move models to device
        self.vae.to(self.device)
        self.mdn_rnn.to(self.device)
        self.controller.to(self.device)

        # Set to eval mode
        self.vae.eval()
        self.mdn_rnn.eval()

    def evaluate(self, params: np.ndarray) -> float:
        """
        Evaluate controller with given parameters.

        Args:
            params: Flat parameter array for controller

        Returns:
            Negative total reward (CMA-ES minimizes)
        """
        # Set controller parameters
        self.controller.set_params(params)
        self.controller.eval()

        total_reward = 0.0

        with torch.no_grad():
            for _ in range(self.num_episodes):
                episode_reward = self._run_episode()
                total_reward += episode_reward

        avg_reward = total_reward / self.num_episodes

        # Return negative because CMA-ES minimizes
        return -avg_reward

    def _run_episode(self) -> float:
        """Run single episode and return total reward."""
        obs, _ = self.env.reset()
        done = False
        episode_reward = 0.0

        # Initialize RNN hidden state
        hidden = self.mdn_rnn.initial_hidden(1, self.device)

        while not done:
            # Convert observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Encode observation
            z = self.vae.encode(obs_tensor, deterministic=True)

            # Get hidden state for controller
            h = self.mdn_rnn.get_hidden_state(hidden)

            # Get action
            action = self.controller.get_action(z, h, deterministic=self.deterministic)
            action_int = action.item()

            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_int)
            done = terminated or truncated
            episode_reward += reward

            # Update RNN hidden state
            action_tensor = torch.tensor([action_int], dtype=torch.long, device=self.device)
            _, _, _, hidden = self.mdn_rnn(z, action_tensor, hidden)

            obs = next_obs

        return episode_reward


def train_controller(
    vae: VAE,
    mdn_rnn: MDNRNN,
    env: TradingEnv,
    latent_dim: int = 32,
    hidden_dim: int = 256,
    action_dim: int = 3,
    use_mlp: bool = False,
    mlp_hidden: int = 64,
    population_size: int = 32,
    sigma: float = 0.5,
    generations: int = 100,
    eval_episodes: int = 4,
    max_steps_per_episode: int = 5000,
    daily_episodes: bool = False,
    close_position_eod: bool = True,
    transaction_cost: float = 0.0,
    target_return: Optional[float] = None,
    patience: int = 10,
    num_workers: Optional[int] = None,
    device: str = 'cpu',
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[Controller, Dict]:
    """
    Train controller using CMA-ES with parallel evaluation.

    Args:
        vae: Trained VAE
        mdn_rnn: Trained MDN-RNN
        env: Trading environment
        latent_dim: Latent dimension
        hidden_dim: RNN hidden dimension
        action_dim: Number of actions
        use_mlp: If True, use MLP controller with hidden layer (more params)
        mlp_hidden: Hidden layer size for MLP controller
        population_size: CMA-ES population size
        sigma: Initial step size
        generations: Maximum generations
        eval_episodes: Episodes per evaluation (each episode = 1 day if daily_episodes=True)
        max_steps_per_episode: Max steps per episode (ignored if daily_episodes=True)
        daily_episodes: If True, each episode is one trading day (~78 bars for 5-min data)
        close_position_eod: If True, force close positions at end of each day
        transaction_cost: Cost per trade in dollars (penalizes excessive trading)
        target_return: Early stopping target (Sortino ratio)
        patience: Early stopping patience (generations without improvement)
        num_workers: Number of parallel workers (default: CPU count)
        device: Device for inference (workers use CPU)
        save_path: Path to save controller
        verbose: Show progress

    Returns:
        Tuple of (trained controller, training history)
    """
    logger.info("Starting CMA-ES controller training")
    if use_mlp:
        logger.info(f"Using MLP controller with {mlp_hidden} hidden units")

    # Determine number of workers
    if num_workers is None:
        num_workers = mp.cpu_count()
    logger.info(f"Using {num_workers} parallel workers")

    # Create controller
    controller = create_controller(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        use_mlp=use_mlp,
        mlp_hidden=mlp_hidden
    )

    num_params = controller.get_param_count()
    logger.info(f"Controller has {num_params} parameters")
    logger.info(f"CMA-ES config: pop={population_size}, gens={generations}, eps={eval_episodes}")

    # Environment config
    env_data = env.data.copy()
    env_config = {
        'initial_balance': env.initial_balance,
        'position_size': env.position_size,
        'observation_columns': env.observation_columns,
        'num_episodes': eval_episodes,
        'max_steps_per_episode': max_steps_per_episode,
        'daily_episodes': daily_episodes,
        'close_position_eod': close_position_eod,
        'transaction_cost': transaction_cost,
        'deterministic': True
    }

    if daily_episodes:
        # Count trading days in data
        if 'day_index' in env_data.columns:
            num_days = env_data['day_index'].nunique()
            avg_bars_per_day = len(env_data) / num_days
            logger.info(f"Daily episodes enabled: {num_days} trading days, ~{avg_bars_per_day:.0f} bars/day")
        else:
            logger.warning("Daily episodes enabled but data lacks day_index column")
    else:
        logger.info(f"Max steps per episode: {max_steps_per_episode} (of {len(env_data)} total)")

    # PRE-ENCODE all observations (MASSIVE SPEEDUP!)
    # This eliminates VAE inference from the training loop entirely
    encoded_obs = pre_encode_all_observations(
        vae=vae,
        data=env_data,
        observation_columns=env.observation_columns,
        device=device
    )

    # Prepare RNN state for workers (serialize to CPU)
    rnn_state = {k: v.cpu() for k, v in mdn_rnn.state_dict().items()}

    # Get RNN config from model
    rnn_config = {
        'latent_dim': mdn_rnn.latent_dim,
        'action_dim': mdn_rnn.action_dim,
        'hidden_dim': mdn_rnn.hidden_dim,
        'num_layers': mdn_rnn.num_layers,
        'num_mixtures': mdn_rnn.num_mixtures
    }

    # Controller config
    controller_config = {
        'latent_dim': latent_dim,
        'hidden_dim': hidden_dim,
        'action_dim': action_dim,
        'use_mlp': use_mlp,
        'mlp_hidden': mlp_hidden
    }

    # Initialize CMA-ES
    initial_params = controller.get_params()
    es = cma.CMAEvolutionStrategy(
        initial_params,
        sigma,
        {'popsize': population_size, 'verbose': -9}  # Suppress CMA-ES output
    )

    # Training history
    history = {
        'generation': [],
        'best_profit': [],
        'mean_profit': [],
        'sigma': []
    }

    # Training loop
    generation = 0
    best_profit = float('-inf')
    best_params = None
    no_improvement_count = 0
    start_time = time.time()

    # Create process pool with initializer
    # Workers receive pre-encoded observations (no VAE needed!)
    logger.info("Initializing worker pool with pre-encoded latents...")
    pool = mp.Pool(
        processes=num_workers,
        initializer=_init_worker,
        initargs=(encoded_obs, rnn_state, rnn_config,
                  controller_config, env_data, env_config)
    )

    try:
        # Calculate total evaluations for progress
        total_evals = generations * population_size
        eval_count = 0

        # Create progress bar for overall evaluations
        pbar = tqdm(
            total=total_evals,
            desc="CMA-ES Training",
            unit="eval",
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} evals [{elapsed}<{remaining}]  {postfix}"
        )

        while not es.stop() and generation < generations:
            gen_start = time.time()

            # Ask for new solutions
            solutions = es.ask()

            # Evaluate in parallel
            # Use async to allow progress updates while waiting
            async_result = pool.map_async(_evaluate_worker, solutions)

            # Poll for completion with progress updates
            dots = 0
            while not async_result.ready():
                async_result.wait(timeout=5)  # Check every 5 seconds
                dots = (dots + 1) % 4
                elapsed_gen = time.time() - gen_start
                pbar.set_postfix_str(
                    f"Gen {generation+1}/{generations} | "
                    f"Evaluating{'.' * (dots + 1):<4} | "
                    f"Time: {elapsed_gen:.0f}s"
                )

            fitnesses = async_result.get()

            # Update progress bar for completed generation
            pbar.update(population_size)

            # Tell CMA-ES the fitnesses
            es.tell(solutions, fitnesses)

            # Track best (fitness is negative profit, so min fitness = best profit)
            min_fitness = min(fitnesses)
            current_best_profit = -min_fitness

            if current_best_profit > best_profit + 1e-6:  # Small threshold for improvement
                best_profit = current_best_profit
                best_idx = fitnesses.index(min_fitness)
                best_params = solutions[best_idx].copy()
                no_improvement_count = 0
                improved = "↑"
            else:
                no_improvement_count += 1
                improved = " "

            # Log progress
            mean_profit = -np.mean(fitnesses)
            history['generation'].append(generation)
            history['best_profit'].append(best_profit)
            history['mean_profit'].append(mean_profit)
            history['sigma'].append(es.sigma)

            gen_time = time.time() - gen_start
            elapsed = time.time() - start_time

            # Update progress bar postfix with current stats
            pbar.set_postfix_str(
                f"Gen {generation+1}/{generations} | "
                f"Profit: {best_profit:.4f}{improved} | "
                f"Mean: {mean_profit:.4f} | "
                f"Stall: {no_improvement_count}/{patience}"
            )

            # Check early stopping - target profit
            if target_return is not None and best_profit >= target_return:
                pbar.write(f"Reached target profit {target_return}")
                break

            # Check early stopping - plateau
            if no_improvement_count >= patience:
                pbar.write(f"Early stopping: no improvement for {patience} generations")
                break

            generation += 1

        pbar.close()

    finally:
        # Clean up pool
        pool.close()
        pool.join()

    total_time = time.time() - start_time
    logger.info(f"Training time: {total_time/60:.1f} minutes")

    # Set controller to best params
    if best_params is not None:
        controller.set_params(best_params)

    logger.info(f"Training complete. Best Profit: {best_profit:.4f}")
    logger.info(f"Completed {generation + 1} generations")

    # Save controller
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': controller.state_dict(),
            'params': best_params,
            'config': {
                'latent_dim': latent_dim,
                'hidden_dim': hidden_dim,
                'action_dim': action_dim,
                'use_mlp': use_mlp,
                'mlp_hidden': mlp_hidden
            },
            'history': history,
            'best_profit': best_profit
        }, save_path)
        logger.info(f"Controller saved to {save_path}")

    return controller, history


def load_controller(path: str, device: str = 'cpu') -> Controller:
    """
    Load a trained controller from disk.

    Args:
        path: Path to saved model
        device: Device to load on

    Returns:
        Loaded controller
    """
    device = torch.device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = checkpoint['config']
    controller = create_controller(
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        action_dim=config['action_dim'],
        use_mlp=config.get('use_mlp', False),
        mlp_hidden=config.get('mlp_hidden', 64)
    )

    if 'model_state_dict' in checkpoint:
        controller.load_state_dict(checkpoint['model_state_dict'])
    elif 'params' in checkpoint:
        controller.set_params(checkpoint['params'])

    controller.to(device)
    controller.eval()

    logger.info(f"Loaded controller from {path}")
    return controller
