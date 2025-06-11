import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pandas as pd
from decimal import Decimal
from collections import deque
import money  # Import the new money module
import logging
from config import config
from stable_baselines3.common.env_util import make_vec_env

logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for trading with multiple technical indicators.

    Observation Space: [close_norm, technical_indicators..., position, unrealized_profit_norm, steps_since_last_trade_norm]
    Action Space:
        0: Buy (Open long if neutral, Close short if short, Hold if already long)
        1: Sell (Open short if neutral, Close long if long, Hold if already short)
        2: Hold (Stay in current position or do nothing)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, data: pd.DataFrame, initial_balance: float = 1.0, transaction_cost: float = 0.0, position_size: int = 1, enabled_indicators: list = None, returns_window: int = 30, reward_type: str = "hybrid"):
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
            returns_window (int): Window size for calculating Sharpe ratio (default: 30).
            reward_type (str): Type of reward function to use: "returns", "sharpe", "hybrid", "risk_adjusted"
        """
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.total_steps = len(self.data) - 1  # Last valid index

        # Action space: 0 = buy, 1 = sell, 2 = hold (stay in current position)
        self.action_space = spaces.Discrete(3)

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
            
        # Calculate observation space size: close_norm + all technical indicators + position + unrealized_profit_norm + steps_since_last_trade_norm
        obs_size = 1 + len(self.technical_indicators) + 1 + 1 + 1
        
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
        self.entry_price = Decimal('0.0')  # Actual price when position was opened
        self.initial_index = 0

        # Portfolio tracking with Decimal for precision
        self.initial_balance = money.to_decimal(initial_balance)
        self.net_worth = self.initial_balance
        self.transaction_cost = money.to_decimal(transaction_cost)
        self.previous_net_worth = self.initial_balance  # Track previous net worth for reward calculation
        
        # Position sizing
        self.position_size = position_size
        
        # Reward and Sharpe ratio calculation
        self.reward_type = reward_type
        self.returns_window = returns_window
        self.returns_history = deque(maxlen=returns_window)
        self.risk_free_rate = 0.02 / 252  # Assume 2% annual risk-free rate, daily
        
        # Enhanced tracking for better Sharpe calculation
        self.portfolio_history = deque(maxlen=returns_window * 2)  # Keep more history
        self.running_sharpe = 0.0
        self.sharpe_ema_alpha = 0.1  # Exponential moving average weight for Sharpe
        
        # Track unrealized profit normalization bounds
        self.max_unrealized_profit = Decimal('0.0')
        self.min_unrealized_profit = Decimal('0.0')
        
        # Trade timing and overtrading tracking
        self.steps_since_last_trade = 0
        self.total_trades = 0
        self.trade_frequency_penalty = 0.001  # Penalty per trade above optimal frequency
        self.min_steps_between_trades = 5  # Minimum recommended steps between trades

    def _calculate_unrealized_profit(self) -> Decimal:
        """Calculate current unrealized profit/loss from the open position."""
        if self.position == 0 or self.entry_price == Decimal('0.0'):
            return Decimal('0.0')
        
        current_price = money.to_decimal(self.data.loc[self.current_step, "close"])
        
        # Calculate price change based on position direction
        if self.position == 1:  # Long position
            price_change = current_price - self.entry_price
        else:  # Short position (position == -1)
            price_change = self.entry_price - current_price
        
        # For NQ futures, each point is $20
        point_value = money.to_decimal(20.0)
        unrealized_profit = price_change * point_value * money.to_decimal(self.position_size)
        
        # Update bounds for normalization
        self.max_unrealized_profit = max(self.max_unrealized_profit, unrealized_profit)
        self.min_unrealized_profit = min(self.min_unrealized_profit, unrealized_profit)
        
        return unrealized_profit
    
    def _normalize_unrealized_profit(self, unrealized_profit: Decimal) -> float:
        """Normalize unrealized profit to [-1, 1] range."""
        if self.max_unrealized_profit == self.min_unrealized_profit:
            return 0.0
        
        # Convert to float for normalization calculation
        unrealized_float = float(unrealized_profit)
        min_float = float(self.min_unrealized_profit)
        max_float = float(self.max_unrealized_profit)
        
        # Normalize to [-1, 1] range
        range_size = max_float - min_float
        normalized = 2.0 * (unrealized_float - min_float) / range_size - 1.0
        
        return float(np.clip(normalized, -1.0, 1.0))
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from returns history with improved stability."""
        min_observations = max(5, self.returns_window // 6)  # Require at least 5 or 1/6 of window
        
        if len(self.returns_history) < min_observations:
            return 0.0
        
        returns_array = np.array(self.returns_history)
        
        # Handle edge cases
        if len(returns_array) == 0:
            return 0.0
        
        # Remove any NaN or infinite values
        returns_array = returns_array[np.isfinite(returns_array)]
        
        if len(returns_array) < min_observations:
            return 0.0
        
        # Use robust statistics for better stability
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)  # Use sample standard deviation
        
        # Check for NaN values in calculations
        if not np.isfinite(mean_return) or not np.isfinite(std_return):
            return -10.0  # Large negative reward for invalid calculations
        
        # Handle zero volatility case
        if std_return == 0:
            if mean_return > self.risk_free_rate:
                return 5.0  # Positive Sharpe for positive returns with no volatility
            elif mean_return < self.risk_free_rate:
                return -5.0  # Negative Sharpe for negative returns with no volatility
            else:
                return 0.0
        
        # Calculate Sharpe ratio
        sharpe = (mean_return - self.risk_free_rate) / std_return
        
        # Ensure the result is finite
        if not np.isfinite(sharpe):
            return -10.0
        
        # Apply reasonable bounds to prevent extreme values
        sharpe_clipped = np.clip(sharpe, -15.0, 15.0)
        
        return float(sharpe_clipped)
    
    def _calculate_rolling_sharpe_ratio(self) -> float:
        """Calculate rolling Sharpe ratio using exponential moving average for smoother updates."""
        current_sharpe = self._calculate_sharpe_ratio()
        
        # Update running Sharpe using exponential moving average
        if len(self.returns_history) >= 5:
            self.running_sharpe = (1 - self.sharpe_ema_alpha) * self.running_sharpe + self.sharpe_ema_alpha * current_sharpe
        else:
            self.running_sharpe = current_sharpe
            
        return self.running_sharpe
    
    def _calculate_risk_adjusted_return(self) -> float:
        """Calculate risk-adjusted return incorporating both return and volatility."""
        if len(self.returns_history) < 5:
            return 0.0
            
        returns_array = np.array(self.returns_history)
        returns_array = returns_array[np.isfinite(returns_array)]
        
        if len(returns_array) == 0:
            return 0.0
            
        # Calculate return and volatility
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        
        if not np.isfinite(mean_return) or not np.isfinite(std_return):
            return -10.0
        
        # Risk-adjusted return: return - penalty for volatility
        volatility_penalty = 0.5 * std_return  # Penalty factor for high volatility
        risk_adjusted = mean_return - volatility_penalty
        
        return float(np.clip(risk_adjusted, -10.0, 10.0))
    
    def _calculate_overtrading_penalty(self) -> float:
        """
        Calculate penalty for overtrading based on trade frequency.
        
        Returns:
            float: Penalty value (always non-positive)
        """
        if self.total_trades == 0:
            return 0.0
            
        # Calculate optimal trade frequency (trades per step)
        # For most strategies, trading every 10-20 steps is reasonable
        current_steps = max(1, self.current_step - self.initial_index)
        optimal_trades = current_steps / 15.0  # Assume optimal frequency is 1 trade per 15 steps
        
        # Calculate overtrading ratio
        excess_trades = max(0, self.total_trades - optimal_trades)
        
        # Apply progressive penalty for excess trades
        if excess_trades > 0:
            penalty = -self.trade_frequency_penalty * (excess_trades ** 1.2)  # Exponential penalty
            return max(penalty, -0.1)  # Cap maximum penalty per step
        
        return 0.0
    
    def _calculate_hybrid_reward(self, period_return: float) -> float:
        """Calculate hybrid reward combining returns and Sharpe ratio."""
        # Weight factors
        return_weight = 0.7
        sharpe_weight = 0.3
        
        # Get current Sharpe ratio (normalized)
        current_sharpe = self._calculate_rolling_sharpe_ratio()
        sharpe_normalized = np.tanh(current_sharpe / 5.0)  # Normalize Sharpe to [-1, 1] range
        
        # Combine return and Sharpe components
        hybrid_reward = return_weight * period_return + sharpe_weight * sharpe_normalized
        
        return float(np.clip(hybrid_reward, -10.0, 10.0))

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
        self.position = 0  # Will be immediately set by first action
        self.entry_price = Decimal('0.0')
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance  # Reset previous net worth for reward calculation
        
        # Reset Sharpe ratio tracking
        self.returns_history.clear()
        self.portfolio_history.clear()
        self.portfolio_history.append(float(self.net_worth))
        self.running_sharpe = 0.0
        
        # Reset unrealized profit bounds
        self.max_unrealized_profit = Decimal('0.0')
        self.min_unrealized_profit = Decimal('0.0')
        
        # Reset trade timing tracking
        self.steps_since_last_trade = 0
        self.total_trades = 0
        
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        """
        Constructs the observation at the current time step.

        Returns:
            np.ndarray: Array with [close_norm, technical_indicators..., position, unrealized_profit_norm, steps_since_last_trade_norm].
        """
        # Get close_norm value
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
                logger.warning(f"Could not find indicator {indicator} in data columns: {self.data.columns}")
                indicators.append(0.0)  # Default value if indicator not found
        
        # Add position
        position = float(self.position)
        
        # Calculate and normalize unrealized profit
        unrealized_profit = self._calculate_unrealized_profit()
        unrealized_profit_norm = self._normalize_unrealized_profit(unrealized_profit)
        
        # Normalize steps since last trade (cap at 50 steps for normalization)
        steps_since_last_trade_norm = min(self.steps_since_last_trade / 50.0, 1.0)
        
        # DEBUGGING: Print details about observation vector construction
        logger = logging.getLogger(__name__)
    
        
        observation_elements = ["close_norm"]
        observation_elements.extend(self.technical_indicators)
        observation_elements.extend(["position", "unrealized_profit_norm", "steps_since_last_trade_norm"])
        
        # Combine all features
        obs = np.array([close_norm] + indicators + [position, unrealized_profit_norm, steps_since_last_trade_norm], dtype=np.float32)
        
        # Replace any NaN or infinite values with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure observation is within bounds
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        
        return obs

    def step(self, action: int):
        """
        Apply an action, update the portfolio, and return the next observation.

        Args:
            action (int): Action to execute (0: buy, 1: sell, 2: hold).
                         Buy: Open long if neutral, close short if short, hold if already long
                         Sell: Open short if neutral, close long if long, hold if already short
                         Hold: Stay in current position

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
            
        NOTE: The reward is now based on raw period returns (log returns) rather than Sharpe ratio.
        This provides more direct feedback to the agent about portfolio performance.
        """
        info = {}
        
        # Store the previous net worth for reward calculation
        self.previous_net_worth = self.net_worth

        # Current price
        current_price = money.to_decimal(self.data.loc[self.current_step, "close"])
        
        # Track old position to detect trades
        old_position = self.position
        trade_occurred = False
        
        # Update position based on action
        if action == 0:  # Buy action
            if self.position == 0:  # Currently neutral - go long
                self.position = 1
                self.entry_price = current_price
                trade_occurred = True
            elif self.position == -1:  # Currently short - close position
                self.position = 0
                self.entry_price = Decimal('0.0')
                trade_occurred = True
            else:  # Already long (position == 1) - maintain position
                pass
                
        elif action == 1:  # Sell action
            if self.position == 0:  # Currently neutral - go short
                self.position = -1
                self.entry_price = current_price
                trade_occurred = True
            elif self.position == 1:  # Currently long - close position
                self.position = 0
                self.entry_price = Decimal('0.0')
                trade_occurred = True
            else:  # Already short (position == -1) - maintain position
                pass
                
        elif action == 2:  # Hold current position
            pass  # No change to position
            
        # Update trade tracking
        if trade_occurred:
            self.total_trades += 1
            self.steps_since_last_trade = 0
        else:
            self.steps_since_last_trade += 1
            
        # Advance to next step
        self.current_step += 1
        
        # Calculate portfolio value change based on position and price change
        if self.current_step < self.total_steps:
            # Next price
            next_price = money.to_decimal(self.data.loc[self.current_step, "close"])
            
            # Calculate price change using the same function as in RiskManager
            if self.position != 0:  # Only if we have an active position
                # Calculate price change - matching RiskManager logic
                if self.position == 1:  # Long position
                    price_change = next_price - current_price
                else:  # Short position
                    price_change = current_price - next_price
                
                # For NQ futures, each point is $20
                point_value = money.to_decimal(20.0)
                
                # Calculate dollar change directly
                dollar_change = price_change * point_value * money.to_decimal(self.position_size)
                
                # Update portfolio value
                self.net_worth += dollar_change
        
        # Check if portfolio has hit zero or below - stop trading immediately
        if self.net_worth <= Decimal('0.0'):
            self.net_worth = Decimal('0.0')
            terminated = True
            truncated = False
            info["portfolio_bankrupted"] = True
            info["final_return"] = -100.0  # Record -100% loss
        else:
            terminated = self.current_step >= self.total_steps
            truncated = False

        if terminated and not info.get("portfolio_bankrupted", False):
            self.current_step = self.total_steps  # Clamp to last valid index
        
        # Calculate period return and add to history
        if self.previous_net_worth > Decimal('0.0') and self.net_worth > Decimal('0.0'):
            period_return = float(np.log(float(self.net_worth / self.previous_net_worth)))
            self.returns_history.append(period_return)
        elif self.net_worth <= Decimal('0.0'):
            # Portfolio went bankrupt - use a large negative return instead of undefined
            period_return = -10.0  # Large negative return representing bankruptcy
            self.returns_history.append(period_return)
        else:
            period_return = 0.0
            self.returns_history.append(period_return)
        
        # Update portfolio history for Sharpe calculation
        self.portfolio_history.append(float(self.net_worth))
        
        # Calculate base reward based on selected reward type
        if self.reward_type == "returns":
            base_reward = period_return
        elif self.reward_type == "sharpe":
            # Use Sharpe ratio as reward (scaled appropriately)
            current_sharpe = self._calculate_rolling_sharpe_ratio()
            base_reward = current_sharpe * 0.1  # Scale down Sharpe for reward range
        elif self.reward_type == "hybrid":
            base_reward = self._calculate_hybrid_reward(period_return)
        elif self.reward_type == "risk_adjusted":
            base_reward = self._calculate_risk_adjusted_return()
        else:
            # Default to returns
            base_reward = period_return
            
        # Apply overtrading penalty
        overtrading_penalty = self._calculate_overtrading_penalty()
        reward = base_reward + overtrading_penalty
        
        obs = self._get_obs()
        
        # Add info about current state
        info["position"] = self.position
        info["reward"] = float(reward)
        info["base_reward"] = float(base_reward)
        info["overtrading_penalty"] = float(overtrading_penalty)
        info["period_return"] = float(period_return)
        info["net_worth"] = float(self.net_worth)
        info["position_size"] = self.position_size
        info["unrealized_profit"] = float(self._calculate_unrealized_profit())
        info["sharpe_ratio"] = self._calculate_sharpe_ratio()  # Standard Sharpe for tracking
        info["rolling_sharpe_ratio"] = self._calculate_rolling_sharpe_ratio()  # Rolling Sharpe for tracking
        info["reward_type"] = self.reward_type
        info["total_trades"] = self.total_trades
        info["steps_since_last_trade"] = self.steps_since_last_trade
        info["trade_occurred"] = trade_occurred
        
        return obs, float(reward), terminated, truncated, info
