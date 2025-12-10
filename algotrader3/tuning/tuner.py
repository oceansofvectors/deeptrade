"""
World Model Hyperparameter Tuner.

Orchestrates VAE and MDN-RNN hyperparameter tuning with Optuna.
Uses hierarchical tuning: VAE first, then RNN with best VAE frozen.
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
import torch
import yaml

from ..data.data_loader import prepare_training_data, get_observation_columns
from ..data.replay_buffer import ReplayBuffer
from ..envs.trading_env import TradingEnv
from ..training.train_vae import encode_observations
from .vae_tuning import vae_objective, train_vae_with_params
from .rnn_tuning import rnn_objective, train_rnn_with_params

logger = logging.getLogger(__name__)


class WorldModelTuner:
    """
    Hyperparameter tuner for World Model components.

    Tunes VAE and MDN-RNN using Optuna with pruning.
    Controller uses default CMA-ES parameters (no tuning needed).
    """

    def __init__(
        self,
        data_path: str,
        output_dir: str = './checkpoints/tuning',
        device: str = 'auto',
        seed: int = 42
    ):
        """
        Initialize tuner.

        Args:
            data_path: Path to training data CSV
            output_dir: Directory for saving models and study database
            device: Device for training (auto, cpu, cuda, mps)
            seed: Random seed
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.seed = seed
        self._set_seed()

        # Data storage
        self.train_obs: Optional[np.ndarray] = None
        self.val_obs: Optional[np.ndarray] = None
        self.train_actions: Optional[np.ndarray] = None
        self.val_actions: Optional[np.ndarray] = None
        self.obs_dim: Optional[int] = None

        # Best results
        self.best_vae_params: Optional[dict] = None
        self.best_rnn_params: Optional[dict] = None
        self.best_vae = None
        self.best_mdn_rnn = None

        logger.info(f"WorldModelTuner initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Output: {self.output_dir}")

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def prepare_data(self, train_ratio: float = 0.6, val_ratio: float = 0.2):
        """
        Load and prepare data for tuning.

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
        """
        logger.info("Preparing data...")

        # Load data
        train_df, val_df, test_df, norm_params = prepare_training_data(
            file_path=self.data_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )

        # Get observation columns
        obs_columns = get_observation_columns()
        obs_columns = [c for c in obs_columns if c in train_df.columns]
        self.obs_dim = len(obs_columns)

        logger.info(f"Observation dimension: {self.obs_dim}")

        # Collect observations and actions from training data
        logger.info("Collecting rollouts from training data...")
        train_env = TradingEnv(
            data=train_df.copy(),
            initial_balance=10000.0,
            position_size=1,
            observation_columns=obs_columns
        )

        train_buffer = self._collect_rollouts(train_env)
        train_data = train_buffer.get_all_data()
        self.train_obs = train_data['obs']
        self.train_actions = train_data['actions']

        logger.info(f"Training samples: {len(self.train_obs)}")

        # Collect from validation data
        logger.info("Collecting rollouts from validation data...")
        val_env = TradingEnv(
            data=val_df.copy(),
            initial_balance=10000.0,
            position_size=1,
            observation_columns=obs_columns
        )

        val_buffer = self._collect_rollouts(val_env)
        val_data = val_buffer.get_all_data()
        self.val_obs = val_data['obs']
        self.val_actions = val_data['actions']

        logger.info(f"Validation samples: {len(self.val_obs)}")
        logger.info("Data preparation complete")

    def _collect_rollouts(self, env: TradingEnv) -> ReplayBuffer:
        """Collect rollouts with random policy."""
        buffer = ReplayBuffer(capacity=1000000)
        obs, _ = env.reset()
        done = False

        while not done:
            action = random.randint(0, 2)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs

        return buffer

    def tune_vae(
        self,
        n_trials: int = 10,
        timeout: Optional[int] = None,
        study_name: str = 'vae_tuning'
    ) -> Tuple[dict, float]:
        """
        Tune VAE hyperparameters.

        Args:
            n_trials: Number of Optuna trials
            timeout: Optional timeout in seconds
            study_name: Name for Optuna study

        Returns:
            Tuple of (best_params, best_value)
        """
        if self.train_obs is None:
            raise RuntimeError("Call prepare_data() first")

        logger.info("=" * 50)
        logger.info("VAE HYPERPARAMETER TUNING")
        logger.info("=" * 50)

        # Create study with SQLite storage for resumption
        storage = f"sqlite:///{self.output_dir}/optuna.db"
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)

        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner
        )

        # Objective wrapper
        def objective(trial):
            return vae_objective(
                trial=trial,
                train_obs=self.train_obs,
                val_obs=self.val_obs,
                obs_dim=self.obs_dim,
                device=self.device
            )

        # Run optimization
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        # Store best params
        self.best_vae_params = study.best_params
        best_value = study.best_value

        logger.info(f"Best VAE params: {self.best_vae_params}")
        logger.info(f"Best VAE loss: {best_value:.6f}")

        # Train final VAE with best params
        logger.info("Training final VAE with best parameters...")
        vae_path = str(self.output_dir / 'best_vae.pt')
        self.best_vae = train_vae_with_params(
            params=self.best_vae_params,
            train_obs=self.train_obs,
            val_obs=self.val_obs,
            obs_dim=self.obs_dim,
            device=self.device,
            save_path=vae_path
        )

        return self.best_vae_params, best_value

    def tune_rnn(
        self,
        n_trials: int = 10,
        timeout: Optional[int] = None,
        study_name: str = 'rnn_tuning'
    ) -> Tuple[dict, float]:
        """
        Tune MDN-RNN hyperparameters with best VAE frozen.

        Args:
            n_trials: Number of Optuna trials
            timeout: Optional timeout in seconds
            study_name: Name for Optuna study

        Returns:
            Tuple of (best_params, best_value)
        """
        if self.best_vae is None:
            raise RuntimeError("Call tune_vae() first")

        logger.info("=" * 50)
        logger.info("MDN-RNN HYPERPARAMETER TUNING")
        logger.info("=" * 50)

        # Pre-encode data with best VAE
        logger.info("Pre-encoding data with best VAE...")
        train_latents = encode_observations(self.best_vae, self.train_obs, device=self.device)
        val_latents = encode_observations(self.best_vae, self.val_obs, device=self.device)

        latent_dim = self.best_vae_params['latent_dim']
        logger.info(f"Encoded to {latent_dim}D latent space")

        # Create study
        storage = f"sqlite:///{self.output_dir}/optuna.db"
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)

        study = optuna.create_study(
            direction='minimize',
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner
        )

        # Objective wrapper
        def objective(trial):
            return rnn_objective(
                trial=trial,
                train_latents=train_latents,
                train_actions=self.train_actions,
                val_latents=val_latents,
                val_actions=self.val_actions,
                latent_dim=latent_dim,
                action_dim=3,
                device=self.device
            )

        # Run optimization
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        # Store best params
        self.best_rnn_params = study.best_params
        best_value = study.best_value

        logger.info(f"Best RNN params: {self.best_rnn_params}")
        logger.info(f"Best RNN NLL: {best_value:.6f}")

        # Train final RNN with best params
        logger.info("Training final MDN-RNN with best parameters...")
        rnn_path = str(self.output_dir / 'best_rnn.pt')
        self.best_mdn_rnn = train_rnn_with_params(
            params=self.best_rnn_params,
            train_latents=train_latents,
            train_actions=self.train_actions,
            latent_dim=latent_dim,
            action_dim=3,
            device=self.device,
            save_path=rnn_path
        )

        return self.best_rnn_params, best_value

    def run_full_tuning(
        self,
        vae_trials: int = 10,
        rnn_trials: int = 10,
        timeout_per_phase: Optional[int] = None
    ) -> Dict[str, dict]:
        """
        Run full VAE + RNN tuning pipeline.

        Args:
            vae_trials: Number of VAE trials
            rnn_trials: Number of RNN trials
            timeout_per_phase: Optional timeout per phase in seconds

        Returns:
            Dictionary with best hyperparameters for each component
        """
        logger.info("=" * 50)
        logger.info("STARTING FULL HYPERPARAMETER TUNING")
        logger.info(f"VAE trials: {vae_trials}")
        logger.info(f"RNN trials: {rnn_trials}")
        logger.info("=" * 50)

        # Prepare data if not done
        if self.train_obs is None:
            self.prepare_data()

        # Phase 1: Tune VAE
        vae_params, vae_loss = self.tune_vae(
            n_trials=vae_trials,
            timeout=timeout_per_phase
        )

        # Phase 2: Tune RNN
        rnn_params, rnn_loss = self.tune_rnn(
            n_trials=rnn_trials,
            timeout=timeout_per_phase
        )

        # Save best hyperparameters
        best_params = {
            'vae': vae_params,
            'rnn': rnn_params,
            'results': {
                'vae_loss': float(vae_loss),
                'rnn_nll': float(rnn_loss)
            }
        }

        params_path = self.output_dir / 'best_hyperparams.yaml'
        with open(params_path, 'w') as f:
            yaml.dump(best_params, f, default_flow_style=False)

        logger.info("=" * 50)
        logger.info("TUNING COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Best VAE saved to: {self.output_dir / 'best_vae.pt'}")
        logger.info(f"Best RNN saved to: {self.output_dir / 'best_rnn.pt'}")
        logger.info(f"Best params saved to: {params_path}")

        return best_params

    def get_tuned_config(self) -> dict:
        """
        Get configuration dict with tuned hyperparameters.

        Returns:
            Config dict ready for use with train_world_model.py
        """
        if self.best_vae_params is None or self.best_rnn_params is None:
            raise RuntimeError("Run tuning first")

        # Parse hidden_dims back to list
        hidden_dims = [int(x) for x in self.best_vae_params['hidden_dims'].split('_')]

        return {
            'vae': {
                'latent_dim': self.best_vae_params['latent_dim'],
                'hidden_dims': hidden_dims,
                'beta': self.best_vae_params['beta'],
                'learning_rate': self.best_vae_params['lr'],
                'epochs': self.best_vae_params['epochs'],
                'batch_size': self.best_vae_params['batch_size']
            },
            'mdnrnn': {
                'hidden_dim': self.best_rnn_params['hidden_dim'],
                'num_layers': self.best_rnn_params['num_layers'],
                'num_mixtures': self.best_rnn_params['num_mixtures'],
                'learning_rate': self.best_rnn_params['lr'],
                'sequence_length': self.best_rnn_params['sequence_length'],
                'epochs': self.best_rnn_params['epochs'],
                'batch_size': self.best_rnn_params['batch_size']
            }
        }
