import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for BTC trading with multiple technical indicators.

    Observation Space: [close_norm, trend_direction, technical_indicators..., position]
    Action Space:
        0: Long (Buy)
        1: Short (Sell)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, data: pd.DataFrame, initial_balance: float = 1.0, transaction_cost: float = 0.0, position_size: int = 1):
        """
        Initialize the trading environment.

        Args:
            data (pd.DataFrame): DataFrame containing price and indicator data with columns:
                - close: Actual closing prices
                - close_norm: Normalized closing prices (between 0 and 1)
                - trend_direction: Supertrend indicator direction
                - Various technical indicators (RSI, CCI, ADX, etc.)
            initial_balance (float): Starting portfolio balance.
            transaction_cost (float): Cost per trade as a fraction of the trade value (e.g., 0.0005 = 0.05%).
                                     Default is 0.0 (no transaction costs).
            position_size (int): Number of contracts to trade (default: 1).
        """
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.total_steps = len(self.data) - 1  # Last valid index

        # Action space: 0 = long (buy), 1 = short (sell)
        self.action_space = spaces.Discrete(2)

        # Define the technical indicators to include in the observation space
        self.technical_indicators = []
        
        # Add available indicators to the list
        if 'trend_direction' in self.data.columns:
            self.technical_indicators.append('trend_direction')
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
            
        # Calculate observation space size: close_norm + all technical indicators + position
        obs_size = 1 + len(self.technical_indicators) + 1
        
        # Create observation space with appropriate bounds
        # Most indicators are normalized between -1 and 1 or 0 and 1
        low_obs = np.array([-1.0] * obs_size, dtype=np.float32)
        high_obs = np.array([1.0] * obs_size, dtype=np.float32)
        
        # Ensure close_norm is between 0 and 1
        low_obs[0] = 0.0
        
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Internal state variables
        self.current_step = 0
        self.position = 0  # Will be immediately updated to either 1 (long) or -1 (short)
        self.entry_price = 0.0  # Actual price when position was opened
        self.initial_index = 0

        # Portfolio tracking
        self.initial_balance = initial_balance
        self.net_worth = self.initial_balance
        self.transaction_cost = transaction_cost
        
        # Position sizing
        self.position_size = position_size
        
        # Reward normalization parameters
        self.correct_prediction_reward = 1.0
        self.incorrect_prediction_reward = -1.0
        
        # Normalize all indicators in the dataset to ensure they're within bounds
        self._normalize_indicators()

    def _normalize_indicators(self):
        """
        Ensure all indicators are properly normalized within the bounds of the observation space.
        """
        # Check each indicator and normalize if needed
        for indicator in self.technical_indicators:
            # Skip indicators that are already guaranteed to be within bounds
            if indicator in ['trend_direction', 'RSI', 'PSAR_DIR']:
                continue
                
            # Get min and max values
            min_val = self.data[indicator].min()
            max_val = self.data[indicator].max()
            
            # Check if values exceed bounds
            if min_val < -1.0 or max_val > 1.0:
                # Normalize to [-1, 1] range
                if min_val != max_val:  # Avoid division by zero
                    self.data[indicator] = 2.0 * (self.data[indicator] - min_val) / (max_val - min_val) - 1.0
                else:
                    self.data[indicator] = 0.0  # If all values are the same, set to 0
                    
                # Clip to ensure within bounds
                self.data[indicator] = np.clip(self.data[indicator], -1.0, 1.0)

    def reset(self, seed=None, options=None):
        """
        Reset the environment state and portfolio to start a new episode.

        Returns:
            tuple: (observation, info)
        """
        self.current_step = self.initial_index
        self.position = 0  # Will be immediately set by first action
        self.entry_price = 0.0
        self.net_worth = self.initial_balance
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        """
        Constructs the observation at the current time step.

        Returns:
            np.ndarray: Array with [close_norm, technical_indicators..., position].
        """
        # Start with close_norm
        close_norm = self.data.loc[self.current_step, "close_norm"]
        
        # Add all technical indicators
        indicators = [self.data.loc[self.current_step, indicator] for indicator in self.technical_indicators]
        
        # Add position
        position = float(self.position)
        
        # Combine all features
        obs = np.array([close_norm] + indicators + [position], dtype=np.float32)
        
        # Ensure observation is within bounds
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        
        return obs

    def step(self, action: int):
        """
        Apply an action, update the portfolio, and return the next observation.

        Args:
            action (int): Action to execute (0: long/buy, 1: short/sell).

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        info = {}
        reward = 0.0

        # Current price based on actual 'Close'
        current_price = self.data.loc[self.current_step, "Close"]
        
        # Track old position to detect changes for transaction cost
        old_position = self.position

        # Update position based on action
        # Only change position if the action is different from current position
        if action == 0:  # Go long
            # If already long (position == 1), maintain position
            # Otherwise, set position to long
            if self.position != 1:
                self.position = 1
                # Add a small reward for changing position to encourage exploration
                if old_position != 0:  # Only if changing from an existing position, not from neutral
                    reward += 0.1
            else:
                pass  # No position duration tracking needed
        else:  # Go short
            # If already short (position == -1), maintain position
            # Otherwise, set position to short
            if self.position != -1:
                self.position = -1
                # Add a small reward for changing position to encourage exploration
                if old_position != 0:  # Only if changing from an existing position, not from neutral
                    reward += 0.1
            else:
                pass  # No position duration tracking needed
            
        # Calculate reward based on correctness of prediction about next candle
        if self.current_step < self.total_steps:
            next_price = self.data.loc[self.current_step + 1, "Close"]
            price_change = next_price - current_price
            
            # If long and price goes up OR short and price goes down, prediction is correct
            if (self.position == 1 and price_change > 0) or (self.position == -1 and price_change < 0):
                reward += self.correct_prediction_reward
            else:
                reward += self.incorrect_prediction_reward
                
            # Normalize reward based on the magnitude of price change
            price_change_pct = abs(price_change / current_price)
            reward *= min(1.0, price_change_pct * 100)  # Scale by percentage change, capped at 1.0

        # Advance to next step
        self.current_step += 1
        
        # Calculate portfolio value change based on position and price change
        if self.current_step < self.total_steps:
            next_price = self.data.loc[self.current_step, "Close"]
            price_change = next_price - current_price
            
            # Calculate dollar value of the price change per contract
            # For NQ futures, each point is worth $20
            point_value = 20.0  # $20 per point for NQ futures
            dollar_change = price_change * point_value * self.position_size
            
            # Update net worth based on position and price change
            if self.position == 1:  # Long position
                # Add the dollar change to the portfolio
                self.net_worth += dollar_change
            elif self.position == -1:  # Short position
                # Subtract the dollar change from the portfolio (profit when price goes down)
                self.net_worth -= dollar_change
        
        # Ensure net_worth doesn't go below a minimum threshold (e.g., 1% of initial balance)
        min_balance = self.initial_balance * 0.01
        if self.net_worth < min_balance:
            self.net_worth = min_balance
            info["hit_minimum_balance"] = True

        terminated = self.current_step >= self.total_steps
        truncated = False

        if terminated:
            self.current_step = self.total_steps  # Clamp to last valid index
        obs = self._get_obs()
        
        # Add info about current state
        info["position"] = self.position
        info["reward"] = reward
        info["net_worth"] = self.net_worth
        info["position_size"] = self.position_size
        
        return obs, float(reward), terminated, truncated, info