"""
Dream Environment for training in imagination.

Uses the MDN-RNN to simulate environment dynamics,
allowing the controller to be trained entirely in "dreams".

Based on Ha & Schmidhuber's World Models (2018).
"""

import logging
from typing import Tuple, Optional, Dict, Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from ..models.vae import VAE
from ..models.mdn_rnn import MDNRNN
from ..constants import NUM_ACTIONS

logger = logging.getLogger(__name__)


class DreamEnv(gym.Env):
    """
    Virtual environment that simulates dynamics using the world model.

    Instead of interacting with the real environment, this env:
    1. Uses MDN-RNN to predict next latent state
    2. Uses a learned or heuristic reward model
    3. Allows training controller purely in imagination
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        vae: VAE,
        mdn_rnn: MDNRNN,
        initial_latents: np.ndarray,
        max_steps: int = 1000,
        temperature: float = 1.0,
        reward_mode: str = 'latent_change',
        device: str = 'cpu'
    ):
        """
        Initialize dream environment.

        Args:
            vae: Trained VAE (for decoding if needed)
            mdn_rnn: Trained MDN-RNN for dynamics
            initial_latents: Pool of initial latent states to sample from [N, latent_dim]
            max_steps: Maximum steps per episode
            temperature: Sampling temperature for MDN (higher = more stochastic)
            reward_mode: How to compute rewards:
                - 'latent_change': Reward based on latent space changes
                - 'position': Simple position-based reward
            device: Device for computation
        """
        super().__init__()

        self.vae = vae
        self.mdn_rnn = mdn_rnn
        self.initial_latents = torch.tensor(initial_latents, dtype=torch.float32)
        self.max_steps = max_steps
        self.temperature = temperature
        self.reward_mode = reward_mode
        self.device = torch.device(device)

        # Move models to device
        self.vae.to(self.device)
        self.mdn_rnn.to(self.device)
        self.initial_latents = self.initial_latents.to(self.device)

        # Set to eval mode
        self.vae.eval()
        self.mdn_rnn.eval()

        # Dimensions
        self.latent_dim = mdn_rnn.latent_dim
        self.hidden_dim = mdn_rnn.hidden_dim
        self.action_dim = mdn_rnn.action_dim

        # Observation space: [z, h] concatenated for controller
        obs_dim = self.latent_dim + self.hidden_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action space
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # State variables
        self.z = None
        self.hidden = None
        self.current_step = 0
        self.position = 0  # Track simulated position

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset dream environment.

        Samples a random initial latent state from the pool.
        """
        super().reset(seed=seed)

        # Sample random initial latent
        idx = np.random.randint(len(self.initial_latents))
        self.z = self.initial_latents[idx:idx+1].clone()

        # Initialize RNN hidden state
        self.hidden = self.mdn_rnn.initial_hidden(1, self.device)

        # Reset counters
        self.current_step = 0
        self.position = 0

        obs = self._get_obs()
        info = {'step': 0, 'position': 0}

        return obs, info

    def _get_obs(self) -> np.ndarray:
        """Get observation: concatenation of z and h."""
        h = self.mdn_rnn.get_hidden_state(self.hidden)
        obs = torch.cat([self.z, h], dim=-1)
        return obs.squeeze(0).cpu().numpy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step in dream environment.

        Args:
            action: Action to take (0=Long, 1=Short, 2=Hold)

        Returns:
            Tuple of (obs, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # Convert action to tensor
        action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)

        # Store previous z for reward calculation
        z_prev = self.z.clone()

        # Predict next latent distribution
        with torch.no_grad():
            pi, mu, sigma, new_hidden = self.mdn_rnn(self.z, action_tensor, self.hidden)

            # Sample next latent state
            self.z = self.mdn_rnn.sample(pi, mu, sigma, self.temperature)

            # Update hidden state
            self.hidden = new_hidden

        # Update simulated position
        old_position = self.position
        if action == 0:  # Long
            self.position = 1
        elif action == 1:  # Short
            self.position = -1
        # Hold keeps position

        # Calculate reward
        reward = self._compute_reward(z_prev, self.z, action, old_position)

        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_obs()
        info = {
            'step': self.current_step,
            'position': self.position
        }

        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self,
        z_prev: torch.Tensor,
        z_next: torch.Tensor,
        action: int,
        old_position: int
    ) -> float:
        """
        Compute reward based on latent space dynamics.

        This is a heuristic since we don't have true rewards in dreams.
        """
        if self.reward_mode == 'latent_change':
            # Reward based on latent space movement
            # Assumes first dimension of z correlates with price movement
            z_change = (z_next[0, 0] - z_prev[0, 0]).item()

            if self.position == 1:  # Long
                reward = z_change * 0.01
            elif self.position == -1:  # Short
                reward = -z_change * 0.01
            else:
                reward = 0.0

            return reward

        elif self.reward_mode == 'position':
            # Simple position-based reward (for testing)
            # Small reward for being in a position
            if self.position != 0:
                return 0.001
            return 0.0

        else:
            return 0.0

    def get_latent(self) -> np.ndarray:
        """Get current latent state."""
        return self.z.squeeze(0).cpu().numpy()

    def get_hidden(self) -> np.ndarray:
        """Get current RNN hidden state."""
        h = self.mdn_rnn.get_hidden_state(self.hidden)
        return h.squeeze(0).cpu().numpy()


def create_dream_env(
    vae: VAE,
    mdn_rnn: MDNRNN,
    initial_latents: np.ndarray,
    max_steps: int = 1000,
    temperature: float = 1.0,
    device: str = 'cpu'
) -> DreamEnv:
    """
    Create a dream environment.

    Args:
        vae: Trained VAE
        mdn_rnn: Trained MDN-RNN
        initial_latents: Pool of initial latent states
        max_steps: Max episode length
        temperature: Sampling temperature
        device: Device

    Returns:
        DreamEnv instance
    """
    return DreamEnv(
        vae=vae,
        mdn_rnn=mdn_rnn,
        initial_latents=initial_latents,
        max_steps=max_steps,
        temperature=temperature,
        device=device
    )
