# Standard library imports
import logging
from decimal import Decimal

# Third-party imports
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.env_util import make_vec_env

# Local application imports
import constants
import money
from config import config

class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for trading with multiple technical indicators.

    Observation Space: [close_norm, technical_indicators..., position, unrealized_pnl_norm, time_in_position_norm]
    Action Space:
        0: Long (Buy)
        1: Short (Sell)
        2: Hold (Stay in current position or do nothing)
        3: Flat (Close position and stay out of market)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, data: pd.DataFrame, initial_balance: float = 1.0, transaction_cost: float = 0.0, position_size: int = 1, enabled_indicators: list = None, dsr_eta: float = 0.01):
        """
        Initialize the trading environment.

        Args:
            data (pd.DataFrame): DataFrame containing price and indicator data with columns:
                - close: Actual closing prices
                - close_norm: Normalized closing prices (between 0 and 1)
                - supertrend: Supertrend indicator direction
                - Various technical indicators (RSI, CCI, ADX, etc.)
            initial_balance (float): Starting portfolio balance.
            transaction_cost (float): Cost per trade as a fraction of the trade value (e.g., 0.0005 = 0.05%).
                                     Default is 0.0 (no transaction costs).
            position_size (int): Number of contracts to trade (default: 1).
            enabled_indicators (list): List of enabled indicators to use in the observation space.
        """
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.total_steps = len(self.data) - 1  # Last valid index

        # Action space: 0 = long (buy), 1 = short (sell), 2 = hold (stay in current position), 3 = flat (close position)
        self.action_space = spaces.Discrete(4)

        # Define the technical indicators to include in the observation space
        self.technical_indicators = []
        
        # If enabled_indicators is provided, use those
        if enabled_indicators is not None:
            self.technical_indicators = enabled_indicators
        else:
            if 'supertrend' in self.data.columns:
                self.technical_indicators.append('supertrend')
            if 'RSI' in self.data.columns:
                self.technical_indicators.append('RSI')
            if 'CCI' in self.data.columns:
                self.technical_indicators.append('CCI')
            if 'ADX' in self.data.columns:
                self.technical_indicators.append('ADX')
            if 'ADX_POS' in self.data.columns:
                self.technical_indicators.append('ADX_POS')
            if 'ADX_NEG' in self.data.columns:
                self.technical_indicators.append('ADX_NEG')
            if 'STOCH_K' in self.data.columns:
                self.technical_indicators.append('STOCH_K')
            if 'STOCH_D' in self.data.columns:
                self.technical_indicators.append('STOCH_D')
            if 'MACD' in self.data.columns:
                self.technical_indicators.append('MACD')
            if 'MACD_SIGNAL' in self.data.columns:
                self.technical_indicators.append('MACD_SIGNAL')
            if 'MACD_HIST' in self.data.columns:
                self.technical_indicators.append('MACD_HIST')
            if 'ROC' in self.data.columns:
                self.technical_indicators.append('ROC')
            if 'WILLIAMS_R' in self.data.columns:
                self.technical_indicators.append('WILLIAMS_R')
            if 'SMA_NORM' in self.data.columns:
                self.technical_indicators.append('SMA_NORM')
            if 'EMA_NORM' in self.data.columns:
                self.technical_indicators.append('EMA_NORM')
            if 'DISPARITY' in self.data.columns:
                self.technical_indicators.append('DISPARITY')
            if 'ATR' in self.data.columns:
                self.technical_indicators.append('ATR')
            if 'OBV_NORM' in self.data.columns:
                self.technical_indicators.append('OBV_NORM')
            if 'CMF' in self.data.columns:
                self.technical_indicators.append('CMF')
            if 'PSAR_NORM' in self.data.columns:
                self.technical_indicators.append('PSAR_NORM')
            if 'PSAR_DIR' in self.data.columns:
                self.technical_indicators.append('PSAR_DIR')
            if 'VOLUME_MA' in self.data.columns:
                self.technical_indicators.append('VOLUME_MA')
            if 'VWAP_NORM' in self.data.columns:
                self.technical_indicators.append('VWAP_NORM')
            if 'DOW_SIN' in self.data.columns:
                self.technical_indicators.append('DOW_SIN')
            if 'DOW_COS' in self.data.columns:
                self.technical_indicators.append('DOW_COS')
            if 'MSO_SIN' in self.data.columns:
                self.technical_indicators.append('MSO_SIN')
            if 'MSO_COS' in self.data.columns:
                self.technical_indicators.append('MSO_COS')
            if 'ZScore' in self.data.columns:
                self.technical_indicators.append('ZScore')
            
        # Calculate observation space size: close_norm + all technical indicators + position + unrealized_pnl + time_in_position
        obs_size = 1 + len(self.technical_indicators) + 3  # +3 for position, unrealized_pnl, time_in_position

        # Create observation space with appropriate bounds
        # Most indicators are normalized between -1 and 1 or 0 and 1
        low_obs = np.array([-1.0] * obs_size, dtype=np.float32)
        high_obs = np.array([1.0] * obs_size, dtype=np.float32)

        # Ensure close_norm is between 0 and 1
        low_obs[0] = 0.0

        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Internal state variables
        self.current_step = 0
        self.position = 0  # 0 = flat, 1 = long, -1 = short
        self.entry_price = Decimal('0.0')  # Actual price when position was opened
        self.initial_index = 0
        self.time_in_position = 0  # Number of steps since position was opened
        self.trade_count = 0  # Total number of trades made in episode

        # Portfolio tracking with Decimal for precision
        self.initial_balance = money.to_decimal(initial_balance)
        self.net_worth = self.initial_balance
        self.transaction_cost = money.to_decimal(transaction_cost)
        self.previous_net_worth = self.initial_balance  # Track previous net worth for reward calculation

        # Risk-adjusted reward tracking
        self.returns_history = []  # Store returns for volatility calculation
        self.max_net_worth = self.initial_balance  # Track peak for drawdown calculation

        # Position sizing
        self.position_size = position_size

        # Differential Sharpe Ratio decay rate
        self.dsr_eta = dsr_eta

    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment state and portfolio to start a new episode.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            tuple: (observation, info)
        """
        # Set random seed if provided
        if seed is not None:
            self.seed(seed)

        self.current_step = self.initial_index
        self.position = 0  # Start flat (no position)
        self.entry_price = Decimal('0.0')
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance  # Reset previous net worth for reward calculation
        self.time_in_position = 0  # Reset time in position
        self.trade_count = 0  # Reset trade count
        self.returns_history = []  # Reset returns history for volatility calculation
        self.max_net_worth = self.initial_balance  # Reset peak for drawdown
        # Reset Differential Sharpe Ratio tracking
        self._dsr_A = 0.0
        self._dsr_B = 0.0001
        self._prev_sharpe = 0.0
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        """
        Constructs the observation at the current time step.

        Returns:
            np.ndarray: Array with [close_norm, technical_indicators..., position, unrealized_pnl_norm, time_in_position_norm].
        """
        # Get close_norm value - check for any capitalization variant
        if "CLOSE_NORM" in self.data.columns:
            close_norm = self.data.loc[self.current_step, "CLOSE_NORM"]
        elif "Close_norm" in self.data.columns:
            close_norm = self.data.loc[self.current_step, "Close_norm"]
        else:
            close_norm = self.data.loc[self.current_step, "close_norm"]

        # Add all technical indicators
        indicators = []
        for indicator in self.technical_indicators:
            # For each indicator, try to find it in any capitalization format
            indicator_value = None

            # Try different capitalization formats
            possible_names = [
                indicator,
                indicator.lower(),
                indicator.upper(),
                indicator.capitalize()
            ]

            for name in possible_names:
                if name in self.data.columns:
                    indicator_value = self.data.loc[self.current_step, name]
                    break

            # Special handling for supertrend indicator
            if indicator.lower() == 'supertrend' and indicator_value is None:
                # Try additional variations specific to supertrend
                for name in ['trend_direction', 'TREND_DIRECTION', 'Trend_Direction']:
                    if name in self.data.columns:
                        indicator_value = self.data.loc[self.current_step, name]
                        break

            # If we found a value, add it, otherwise use a default
            if indicator_value is not None:
                indicators.append(indicator_value)
            else:
                logger = logging.getLogger(__name__)
                logger.warning(f"Could not find indicator {indicator} in data columns: {self.data.columns}")
                indicators.append(0.0)  # Default value if indicator not found

        # Add position (normalized to -1, 0, 1)
        position = float(self.position)

        # Calculate unrealized P&L normalized to [-1, 1]
        unrealized_pnl_norm = self._calculate_unrealized_pnl_normalized()

        # Normalize time in position (using sigmoid-like scaling, capped at ~100 steps)
        # This maps 0 -> 0, 50 -> ~0.76, 100 -> ~0.96
        time_in_position_norm = float(np.tanh(self.time_in_position / 50.0))

        # Combine all features
        obs = np.array([close_norm] + indicators + [position, unrealized_pnl_norm, time_in_position_norm], dtype=np.float32)

        # Ensure observation is within bounds
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        return obs

    def _get_current_price(self) -> Decimal:
        """Get the current price from the data."""
        if 'CLOSE' in self.data.columns:
            return money.to_decimal(self.data.loc[self.current_step, "CLOSE"])
        elif 'Close' in self.data.columns:
            return money.to_decimal(self.data.loc[self.current_step, "Close"])
        else:
            return money.to_decimal(self.data.loc[self.current_step, "close"])

    def _calculate_unrealized_pnl_normalized(self) -> float:
        """
        Calculate the unrealized P&L normalized to [-1, 1].

        Uses tanh scaling to map P&L to a bounded range.
        A P&L of ~$500 maps to ~0.76, $1000 maps to ~0.96.
        """
        if self.position == 0 or self.entry_price == Decimal('0.0'):
            return 0.0

        current_price = self._get_current_price()

        # Calculate unrealized P&L
        if self.position == 1:  # Long
            price_change = current_price - self.entry_price
        else:  # Short
            price_change = self.entry_price - current_price

        point_value = money.to_decimal(constants.NQ_POINT_VALUE)
        unrealized_pnl = float(price_change * point_value * money.to_decimal(self.position_size))

        # Normalize using tanh with scaling factor of $500
        # This maps $500 -> ~0.76, $1000 -> ~0.96, -$500 -> ~-0.76
        normalized = float(np.tanh(unrealized_pnl / 500.0))

        return normalized

    def step(self, action: int):
        """
        Apply an action, update the portfolio, and return the next observation.

        Args:
            action (int): Action to execute (0: long/buy, 1: short/sell, 2: hold, 3: flat/close position).

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        info = {}
        info["action_masked"] = False

        # Store the previous net worth for reward calculation
        self.previous_net_worth = self.net_worth

        # Get current price
        current_price = self._get_current_price()

        # Track old position to detect changes for transaction cost
        old_position = self.position
        position_changed = False
        is_redundant_action = False

        # ACTION MASKING: Prevent redundant actions
        # If action would result in no change, treat it as hold
        if action == 0 and self.position == 1:  # Already long, trying to go long
            is_redundant_action = True
            info["action_masked"] = True
        elif action == 1 and self.position == -1:  # Already short, trying to go short
            is_redundant_action = True
            info["action_masked"] = True
        elif action == 3 and self.position == 0:  # Already flat, trying to go flat
            is_redundant_action = True
            info["action_masked"] = True

        # Update position based on action (unless redundant)
        if not is_redundant_action:
            if action == 0:  # Go long
                if self.position != 1:
                    self.position = 1
                    self.entry_price = current_price
                    self.time_in_position = 0
                    self.trade_count += 1
                    position_changed = True

            elif action == 1:  # Go short
                if self.position != -1:
                    self.position = -1
                    self.entry_price = current_price
                    self.time_in_position = 0
                    self.trade_count += 1
                    position_changed = True

            elif action == 2:  # Hold current position
                pass  # No change to position

            elif action == 3:  # Go flat (close position)
                if self.position != 0:
                    self.position = 0
                    self.entry_price = Decimal('0.0')
                    self.time_in_position = 0
                    self.trade_count += 1
                    position_changed = True

        # Apply transaction cost if position changed (realistic cost model)
        transaction_cost_applied = Decimal('0.0')
        if position_changed and self.transaction_cost > Decimal('0.0'):
            # Transaction cost per contract (round trip if reversing position)
            cost_multiplier = 2 if old_position != 0 and self.position != 0 and old_position != self.position else 1
            transaction_cost_applied = self.transaction_cost * money.to_decimal(self.position_size) * Decimal(str(cost_multiplier))
            self.net_worth -= transaction_cost_applied

        # Increment time in position if holding
        if self.position != 0 and not position_changed:
            self.time_in_position += 1

        # Advance to next step
        self.current_step += 1

        # Calculate portfolio value change based on position and price change
        # IMPORTANT: Use old_position for P&L calculation, not the new position!
        # This handles the case where we exit a position (P&L should be calculated on the old position)
        if self.current_step < self.total_steps:
            next_price = self._get_current_price()

            # Calculate P&L if we HAD an active position (use old_position)
            if old_position != 0:
                # Calculate price change based on the position we WERE in
                if old_position == 1:  # Was long position
                    price_change = next_price - current_price
                else:  # Was short position (old_position == -1)
                    price_change = current_price - next_price

                # For NQ futures, each point is $20
                point_value = money.to_decimal(constants.NQ_POINT_VALUE)

                # Calculate dollar change directly
                dollar_change = price_change * point_value * money.to_decimal(self.position_size)

                # Update portfolio value
                self.net_worth += dollar_change

        # Update max net worth for drawdown tracking
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        # Ensure net_worth doesn't go below a minimum threshold (e.g., 1% of initial balance)
        min_balance = self.initial_balance * Decimal(str(constants.MIN_BALANCE_PERCENTAGE))
        if self.net_worth < min_balance:
            self.net_worth = min_balance
            info["hit_minimum_balance"] = True

        terminated = self.current_step >= self.total_steps
        truncated = False

        if terminated:
            self.current_step = self.total_steps  # Clamp to last valid index

        # Calculate RISK-ADJUSTED REWARD
        reward = self._calculate_risk_adjusted_reward(
            position_changed=position_changed,
            transaction_cost_applied=float(transaction_cost_applied),
            is_redundant_action=is_redundant_action
        )

        obs = self._get_obs()

        # Add info about current state
        info["position"] = self.position
        info["old_position"] = old_position
        info["position_changed"] = position_changed
        info["reward"] = float(reward)
        info["net_worth"] = float(self.net_worth)
        info["position_size"] = self.position_size
        info["time_in_position"] = self.time_in_position
        info["trade_count"] = self.trade_count
        info["transaction_cost"] = float(transaction_cost_applied)
        info["max_net_worth"] = float(self.max_net_worth)

        return obs, float(reward), terminated, truncated, info

    def _calculate_risk_adjusted_reward(self, position_changed: bool, transaction_cost_applied: float, is_redundant_action: bool) -> float:
        """
        Calculate reward using Differential Sharpe Ratio.

        Rewards improvement in risk-adjusted performance, not just raw returns.
        This encourages consistent returns over volatile swings.

        Returns:
            float: Differential Sharpe reward
        """
        if self.previous_net_worth > Decimal('0.0'):
            base_return = float(np.log(float(self.net_worth / self.previous_net_worth)))
        else:
            base_return = 0.0

        # Track returns for history
        self.returns_history.append(base_return)

        # Differential Sharpe Ratio calculation
        # Uses exponential moving averages for A (mean return) and B (mean squared return)
        eta = self.dsr_eta  # Decay rate for EMA (smaller = longer memory)

        # Initialize tracking variables on first call
        if not hasattr(self, '_dsr_A'):
            self._dsr_A = 0.0  # EMA of returns
            self._dsr_B = 0.0001  # EMA of squared returns (small init to avoid div by 0)
            self._prev_sharpe = 0.0

        # Update EMAs
        self._dsr_A = self._dsr_A + eta * (base_return - self._dsr_A)
        self._dsr_B = self._dsr_B + eta * (base_return ** 2 - self._dsr_B)

        # Calculate current Sharpe-like ratio: A / sqrt(B - A²)
        variance = self._dsr_B - self._dsr_A ** 2
        if variance > 1e-10:
            current_sharpe = self._dsr_A / np.sqrt(variance)
        else:
            current_sharpe = self._dsr_A * 100  # High Sharpe if no variance

        # Differential Sharpe = change in Sharpe ratio
        diff_sharpe = current_sharpe - self._prev_sharpe
        self._prev_sharpe = current_sharpe

        # Scale for stronger learning signal
        return diff_sharpe * 100.0
