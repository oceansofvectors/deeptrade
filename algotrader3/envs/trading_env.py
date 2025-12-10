"""
Trading environment for World Model training.
Adapted from algotrader2's environment.py
"""

import logging
from decimal import Decimal
from typing import Tuple, Dict, Any, Optional, List

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..constants import (
    NQ_POINT_VALUE,
    ACTION_LONG,
    ACTION_SHORT,
    ACTION_HOLD,
    NUM_ACTIONS,
    POSITION_FLAT,
    POSITION_LONG,
    POSITION_SHORT,
    MIN_BALANCE_PERCENTAGE
)

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Trading environment for NQ futures.

    Observation Space: Technical indicators + position
    Action Space: 0 (Long), 1 (Short), 2 (Hold)

    The environment simulates trading with:
    - $20 per point for NQ futures
    - Log returns as reward
    - Position tracking (long/short/flat)

    Supports daily episode mode where:
    - Each episode is one trading day
    - Positions are automatically closed at end of day
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        position_size: int = 1,
        point_value: float = NQ_POINT_VALUE,
        observation_columns: Optional[List[str]] = None,
        daily_episodes: bool = False,
        close_position_eod: bool = True,
        transaction_cost: float = 0.0
    ):
        """
        Initialize the trading environment.

        Args:
            data: DataFrame with OHLCV and indicator data
            initial_balance: Starting portfolio balance
            position_size: Number of contracts to trade
            point_value: Dollar value per point (default $20 for NQ)
            observation_columns: List of columns to use as observations
            daily_episodes: If True, each episode is one trading day
            close_position_eod: If True, automatically close positions at end of day
            transaction_cost: Cost per trade in dollars (applied on position change)
        """
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.total_steps = len(self.data) - 1

        # Trading parameters
        self.initial_balance = Decimal(str(initial_balance))
        self.position_size = position_size
        self.point_value = Decimal(str(point_value))
        self.transaction_cost = Decimal(str(transaction_cost))

        # Daily episode settings
        self.daily_episodes = daily_episodes
        self.close_position_eod = close_position_eod

        # Check if day boundary columns exist
        self.has_day_info = 'is_last_bar_of_day' in self.data.columns

        # Build day start indices for random day selection
        if self.has_day_info and 'day_index' in self.data.columns:
            self._build_day_indices()
        else:
            self.day_start_indices = None
            self.num_days = 0

        # Define observation columns
        if observation_columns is None:
            # Default observation columns
            self.observation_columns = self._get_default_observation_columns()
        else:
            self.observation_columns = observation_columns

        # Filter to columns that exist in data
        self.observation_columns = [
            col for col in self.observation_columns
            if col in self.data.columns
        ]

        # Observation space
        obs_dim = len(self.observation_columns)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action space: Long (0), Short (1), Hold (2)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # State variables (will be set in reset)
        self.current_step = 0
        self.position = POSITION_FLAT
        self.entry_price = Decimal('0')
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance
        self.current_day_index = 0
        self.episode_end_step = self.total_steps  # Where current episode ends

        logger.info(f"TradingEnv initialized with {obs_dim} observation features")
        if self.daily_episodes:
            if self.num_days > 0:
                logger.info(f"  Daily episodes enabled: {self.num_days} trading days available")
            else:
                logger.warning(f"  Daily episodes requested but data lacks day boundaries!")
                logger.warning(f"  Delete cached data and re-run to add day boundaries.")
                self.daily_episodes = False  # Disable since we can't use it
        if self.close_position_eod and self.has_day_info:
            logger.info(f"  End-of-day position close: enabled")

    def _build_day_indices(self):
        """Build lookup table for day start indices."""
        # Find the first row of each trading day
        day_changes = self.data['day_index'].diff().fillna(1) != 0
        self.day_start_indices = self.data.index[day_changes].tolist()
        self.num_days = len(self.day_start_indices)

        # Also store day end indices (last bar of each day)
        if 'is_last_bar_of_day' in self.data.columns:
            self.day_end_indices = self.data.index[self.data['is_last_bar_of_day']].tolist()
        else:
            # Approximate: day ends at start of next day - 1
            self.day_end_indices = [idx - 1 for idx in self.day_start_indices[1:]]
            self.day_end_indices.append(self.total_steps)

    def _get_default_observation_columns(self) -> List[str]:
        """Get default list of observation columns."""
        return [
            'close_norm',
            'supertrend',
            'RSI',
            'CCI',
            'ADX',
            'ADX_POS',
            'ADX_NEG',
            'STOCH_K',
            'STOCH_D',
            'MACD',
            'MACD_SIGNAL',
            'MACD_HIST',
            'ROC',
            'WILLIAMS_R',
            'SMA',
            'EMA',
            'DISPARITY',
            'ATR',
            'OBV',
            'CMF',
            'PSAR_DIR',
            'DOW_SIN',
            'DOW_COS',
            'MSO_SIN',
            'MSO_COS',
            'position'
        ]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Args:
            options: Optional dict with:
                - 'random_start': If True, start at random position (default: False)
                - 'max_steps': If set, limit episode length (ignored in daily mode)
                - 'day_index': Specific day to start on (for daily_episodes mode)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        options = options or {}

        if self.daily_episodes and self.day_start_indices:
            # Daily episode mode: each episode is one trading day
            if 'day_index' in options:
                # Start on specific day
                day_idx = options['day_index'] % self.num_days
            elif options.get('random_start', False):
                # Random day selection
                day_idx = self.np_random.integers(0, self.num_days)
            else:
                # Sequential day (start from beginning)
                day_idx = 0

            self.current_day_index = day_idx
            self.current_step = self.day_start_indices[day_idx]
            self.episode_end_step = self.day_end_indices[day_idx]
        else:
            # Original behavior: random or sequential start
            if options.get('random_start', False):
                max_start = max(0, self.total_steps - options.get('max_steps', 5000))
                self.current_step = self.np_random.integers(0, max(1, max_start))
            else:
                self.current_step = 0

            self.episode_end_step = self.total_steps
            self.current_day_index = 0

        self.position = POSITION_FLAT
        self.entry_price = Decimal('0')
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance

        # Update position in data
        self.data.loc[self.current_step, 'position'] = self.position

        obs = self._get_obs()
        info = {
            'net_worth': float(self.net_worth),
            'position': self.position,
            'day_index': self.current_day_index,
            'bars_remaining': self.episode_end_step - self.current_step
        }

        return obs, info

    def _get_obs(self) -> np.ndarray:
        """Get current observation vector."""
        obs_values = []

        for col in self.observation_columns:
            if col == 'position':
                obs_values.append(float(self.position))
            else:
                value = self.data.loc[self.current_step, col]
                obs_values.append(float(value))

        obs = np.array(obs_values, dtype=np.float32)

        # Clip to reasonable bounds
        obs = np.clip(obs, -10.0, 10.0)

        return obs

    def _get_current_price(self) -> Decimal:
        """Get current close price."""
        price = self.data.loc[self.current_step, 'close']
        return Decimal(str(float(price)))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: 0 (Long), 1 (Short), 2 (Hold)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Store previous net worth for reward calculation
        self.previous_net_worth = self.net_worth
        previous_position = self.position

        # Get current price
        current_price = self._get_current_price()

        # Check if this is the last bar of the day (for EOD close)
        is_last_bar = False
        if self.has_day_info and self.close_position_eod:
            is_last_bar = bool(self.data.loc[self.current_step, 'is_last_bar_of_day'])

        # Update position based on action
        if action == ACTION_LONG:  # Go long
            if self.position != POSITION_LONG:
                self.position = POSITION_LONG
                self.entry_price = current_price

        elif action == ACTION_SHORT:  # Go short
            if self.position != POSITION_SHORT:
                self.position = POSITION_SHORT
                self.entry_price = current_price

        elif action == ACTION_HOLD:  # Close position / go flat
            # IMPORTANT: Hold means "close any open position"
            # This enables the model to exit trades, not just switch sides
            if self.position != POSITION_FLAT:
                self.position = POSITION_FLAT
                self.entry_price = Decimal('0')

        # Apply transaction cost on position change
        if self.position != previous_position and self.transaction_cost > 0:
            self.net_worth -= self.transaction_cost

        # Advance to next step
        self.current_step += 1

        # Calculate P&L based on price change
        if self.current_step <= self.total_steps and self.position != POSITION_FLAT:
            next_price = self._get_current_price()

            if self.position == POSITION_LONG:
                price_change = next_price - current_price
            else:  # SHORT
                price_change = current_price - next_price

            dollar_change = price_change * self.point_value * Decimal(str(self.position_size))
            self.net_worth += dollar_change

        # Force close position at end of day if enabled
        eod_closed = False
        if is_last_bar and self.close_position_eod and self.position != POSITION_FLAT:
            # Close position at current price (realize any remaining P&L)
            self.position = POSITION_FLAT
            self.entry_price = Decimal('0')
            eod_closed = True

        # Ensure minimum balance
        min_balance = self.initial_balance * Decimal(str(MIN_BALANCE_PERCENTAGE))
        if self.net_worth < min_balance:
            self.net_worth = min_balance

        # Check termination - use episode_end_step for daily episodes
        terminated = self.current_step >= self.episode_end_step or self.current_step >= self.total_steps
        truncated = False

        if self.current_step > self.total_steps:
            self.current_step = self.total_steps

        # Calculate reward (log return)
        if self.previous_net_worth > Decimal('0'):
            reward = float(np.log(float(self.net_worth / self.previous_net_worth)))
        else:
            reward = 0.0

        # Update position in data for next observation
        if not terminated and self.current_step <= self.total_steps:
            self.data.loc[self.current_step, 'position'] = self.position

        obs = self._get_obs()

        info = {
            'position': self.position,
            'net_worth': float(self.net_worth),
            'reward': reward,
            'step': self.current_step,
            'day_index': self.current_day_index,
            'eod_closed': eod_closed
        }

        return obs, reward, terminated, truncated, info

    def get_obs_dim(self) -> int:
        """Get observation space dimension."""
        return len(self.observation_columns)

    def get_action_dim(self) -> int:
        """Get action space dimension."""
        return NUM_ACTIONS

    def render(self, mode: str = "human"):
        """Render the environment (optional)."""
        if mode == "human":
            print(f"Step: {self.current_step}, Position: {self.position}, "
                  f"Net Worth: ${float(self.net_worth):.2f}")


def create_env_from_data(
    data: pd.DataFrame,
    initial_balance: float = 10000.0,
    position_size: int = 1
) -> TradingEnv:
    """
    Create a trading environment from a DataFrame.

    Args:
        data: DataFrame with OHLCV and indicator data
        initial_balance: Starting balance
        position_size: Number of contracts

    Returns:
        TradingEnv instance
    """
    return TradingEnv(
        data=data,
        initial_balance=initial_balance,
        position_size=position_size
    )
