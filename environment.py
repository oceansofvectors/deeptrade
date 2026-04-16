# Standard library imports
import logging
from decimal import Decimal
from typing import Tuple

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
from action_space import target_allocation_for_action
from config import config

class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for trading with multiple technical indicators.

    Observation Space: [close_norm, technical_indicators..., position, unrealized_pnl_norm, time_in_position_norm]
    Action Space:
        0-2: Long target allocations of 1%, 2%, 5%
        3-5: Short target allocations of 1%, 2%, 5%
        6: Flat

    Agent chooses a target portfolio allocation. The environment rebalances to the
    closest whole-contract MBT position that fits current net worth.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, data: pd.DataFrame, initial_balance: float = 1.0, transaction_cost: float = 0.0, position_size: int = 1, enabled_indicators: list = None, random_start_pct: float = 0.0, min_episode_steps: int = 2, **kwargs):
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

        # Preserve datetime index for time-of-day execution cost model
        if isinstance(data.index, pd.DatetimeIndex):
            self._hours = data.index.hour.values
        elif 'time' in data.columns:
            self._hours = pd.to_datetime(data['time']).dt.hour.values
        else:
            self._hours = None

        self.data = data.reset_index(drop=True)
        self.total_steps = len(self.data) - 1  # Last valid index

        # Dynamic SL/TP configuration
        dynamic_sl_tp_config = config.get("risk_management", {}).get("dynamic_sl_tp", {})
        self.dynamic_sl_tp_enabled = dynamic_sl_tp_config.get("enabled", False)

        if self.dynamic_sl_tp_enabled:
            self.sl_range = dynamic_sl_tp_config.get("sl_multiplier_range", [1.5, 5.0])
            self.tp_range = dynamic_sl_tp_config.get("tp_multiplier_range", [1.5, 5.0])
            self.num_choices = dynamic_sl_tp_config.get("num_choices", 8)

            # Create lookup tables for multipliers
            self.sl_multipliers = np.linspace(self.sl_range[0], self.sl_range[1], self.num_choices)
            self.tp_multipliers = np.linspace(self.tp_range[0], self.tp_range[1], self.num_choices)

            # Action space: [position_action, sl_multiplier_idx, tp_multiplier_idx]
            self.action_space = spaces.MultiDiscrete([7, self.num_choices, self.num_choices])
        else:
            # Action space: 7 target-allocation actions
            self.action_space = spaces.Discrete(7)

        # Fixed ATR-based stop loss / take profit from risk_management config
        risk_config = config.get("risk_management", {})
        sl_config = risk_config.get("stop_loss", {})
        tp_config = risk_config.get("take_profit", {})
        self.fixed_sl_enabled = sl_config.get("enabled", False)
        self.fixed_tp_enabled = tp_config.get("enabled", False)
        self.fixed_sl_atr_mult = Decimal(str(sl_config.get("atr_multiplier", 2.0))) if sl_config.get("mode") == "atr" else None
        self.fixed_tp_atr_mult = Decimal(str(tp_config.get("atr_multiplier", 3.0))) if tp_config.get("mode") == "atr" else None
        self.fixed_sl_pct = Decimal(str(sl_config.get("percentage", 17))) / Decimal('100') if sl_config.get("mode") == "percentage" else None
        self.fixed_tp_pct = Decimal(str(tp_config.get("percentage", 20))) / Decimal('100') if tp_config.get("mode") == "percentage" else None
        self.fixed_sl_price = Decimal('0.0')
        self.fixed_tp_price = Decimal('0.0')

        # SL/TP state tracking (for dynamic/agent-chosen SL/TP)
        self.current_sl_multiplier = None
        self.current_tp_multiplier = None
        self.sl_price = Decimal('0.0')
        self.tp_price = Decimal('0.0')

        # Define the technical indicators to include in the observation space
        self.technical_indicators = []
        
        # If enabled_indicators is provided, use those
        if enabled_indicators is not None:
            self.technical_indicators = enabled_indicators
        else:
            excluded_cols = {
                'time', 'timestamp', 'date', 'symbol', 'rtype', 'publisher_id', 'instrument_id',
                'position', 'close_norm',
                'open', 'high', 'low', 'close', 'volume',
                'Open', 'High', 'Low', 'Close', 'Volume',
                'OPEN', 'HIGH', 'LOW', 'CLOSE',
            }
            for col in self.data.columns:
                if col in excluded_cols:
                    continue
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    self.technical_indicators.append(col)
            
        # Calculate observation space size: close_norm + all technical indicators +
        # signed_exposure + unrealized_pnl + time_in_position + drawdown_pct
        obs_size = 1 + len(self.technical_indicators) + 4

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
        self.current_contracts = 0
        self.target_allocation_pct = Decimal('0.0')
        self.signed_exposure_pct = Decimal('0.0')
        self.entry_price = Decimal('0.0')  # Weighted average entry price
        self.initial_index = 0
        # Random start: on reset(), pick a random start within first random_start_pct of data
        # This gives vectorized envs trajectory diversity across episodes
        self.random_start_pct = random_start_pct
        self.min_episode_steps = max(1, int(min_episode_steps))
        random_window_max = int(self.total_steps * random_start_pct) if random_start_pct > 0 else 0
        safe_offset_max = max(0, self.total_steps - self.min_episode_steps)
        self._max_random_offset = min(random_window_max, safe_offset_max)
        self.time_in_position = 0  # Number of steps since position was opened
        self.time_flat = 0  # Number of steps with no position (for inactivity penalty)
        self.trade_count = 0  # Total number of trades made in episode

        # Portfolio tracking with Decimal for precision
        self.initial_balance = money.to_decimal(initial_balance)
        self.net_worth = self.initial_balance
        self.transaction_cost = money.to_decimal(transaction_cost)
        self.previous_net_worth = self.initial_balance  # Track previous net worth for reward calculation

        # Risk-adjusted reward tracking
        self.returns_history = []  # Store returns for volatility calculation
        self.max_net_worth = self.initial_balance  # Track peak for drawdown calculation

        # Legacy config fallback retained for older code paths/tests.
        self.position_size = position_size

        # Pre-compute Decimal constants (avoid per-step conversions)
        self._point_value_decimal = money.to_decimal(constants.CONTRACT_POINT_VALUE)
        self._position_size_decimal = money.to_decimal(self.position_size)
        self._min_balance_pct_decimal = Decimal(str(constants.MIN_BALANCE_PERCENTAGE))
        self._allocation_choices = [Decimal('0.01'), Decimal('0.02'), Decimal('0.05')]

        # Realistic execution cost model: fee + spread + time-of-day slippage.
        # MBT outright tick = 5.0 quoted BTC price points = $0.50/contract.
        exec_config = config.get("execution_costs", {})
        self._spread_points = Decimal(str(exec_config.get("half_spread_points", 2.5)))
        base_slippage = exec_config.get("base_slippage_points", 0.10)

        # Pre-compute per-bar slippage in points (time-of-day dependent)
        if self._hours is not None and base_slippage > 0:
            slippage = np.full(len(self.data), base_slippage, dtype=np.float64)
            slippage[(self._hours >= 21) | (self._hours < 1)] = base_slippage * 1.5
            slippage[(self._hours >= 15) & (self._hours < 17)] = base_slippage * 1.25
            self._slippage_points = slippage
        else:
            self._slippage_points = np.full(len(self.data), base_slippage, dtype=np.float64)

        # Pre-compute column name lookups (avoid per-step capitalization checks)
        self._close_norm_col = next((c for c in ['CLOSE_NORM', 'Close_norm', 'close_norm'] if c in self.data.columns), 'close_norm')
        self._close_col = next((c for c in ['CLOSE', 'Close', 'close'] if c in self.data.columns), 'close')
        self._high_col = next((c for c in ['HIGH', 'High', 'high'] if c in self.data.columns), None)
        self._low_col = next((c for c in ['LOW', 'Low', 'low'] if c in self.data.columns), None)
        self._atr_col = next((c for c in ['ATR', 'atr'] if c in self.data.columns), None)

        # Pre-compute numpy arrays for fast per-step access (avoid DataFrame .loc[])
        self._close_norm_array = self._sanitize_feature_array(
            self.data[self._close_norm_col].values.astype(np.float32),
            name=self._close_norm_col,
        )
        self._close_array = self._sanitize_price_array(
            self.data[self._close_col].values.astype(np.float64),
            name=self._close_col,
        )
        self._high_array = self._sanitize_price_array(
            self.data[self._high_col].values.astype(np.float64),
            fallback=self._close_array,
            name=self._high_col,
        ) if self._high_col else None
        self._low_array = self._sanitize_price_array(
            self.data[self._low_col].values.astype(np.float64),
            fallback=self._close_array,
            name=self._low_col,
        ) if self._low_col else None
        self._atr_array = self._sanitize_feature_array(
            self.data[self._atr_col].values.astype(np.float64),
            name=self._atr_col,
        ) if self._atr_col else None

        # Resolve indicator column names once and build pre-computed indicator matrix
        resolved_indicator_cols = []
        self._indicator_found = []
        for indicator in self.technical_indicators:
            possible = [indicator, indicator.lower(), indicator.upper(), indicator.capitalize()]
            found = next((n for n in possible if n in self.data.columns), None)
            if found is None and indicator.lower() == 'supertrend':
                found = next((n for n in ['trend_direction', 'TREND_DIRECTION', 'Trend_Direction'] if n in self.data.columns), None)
            resolved_indicator_cols.append(found)
            self._indicator_found.append(found is not None)

        # Build indicator matrix: shape (total_steps+1, num_indicators)
        if any(self._indicator_found):
            cols = [c if c is not None else self._close_norm_col for c in resolved_indicator_cols]
            self._indicator_matrix = self.data[cols].values.astype(np.float32)
            # Zero out columns where indicator was not found
            for i, found in enumerate(self._indicator_found):
                if not found:
                    self._indicator_matrix[:, i] = 0.0
            self._indicator_matrix = np.nan_to_num(self._indicator_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            self._indicator_matrix = np.zeros((len(self.data), len(self.technical_indicators)), dtype=np.float32)

        # Pre-allocate observation buffer
        self._obs_size = obs_size
        self._n_indicators = len(self.technical_indicators)

        # Cache regime feature indices for reward shaping (calm holding bonus)
        self._rolling_dd_idx = None
        self._vol_pct_idx = None
        for i, ind in enumerate(self.technical_indicators):
            if ind == 'ROLLING_DD':
                self._rolling_dd_idx = i
            elif ind == 'VOL_PERCENTILE':
                self._vol_pct_idx = i

        # Cache reward config lookups (avoid repeated config.get() in per-step reward loop)
        reward_config = config.get("reward", {})
        self._reward_base_scale = reward_config.get("base_scale", 500.0)
        self._reward_loss_multiplier = reward_config.get("loss_multiplier", 0.7)
        self._reward_dd_threshold = reward_config.get("drawdown_penalty_threshold", 0.03)
        self._reward_dd_penalty = reward_config.get("drawdown_penalty", 3.0)
        self._reward_turnover_pen = reward_config.get("turnover_penalty", 0.05)
        self._reward_calm_bonus = reward_config.get("calm_holding_bonus", 0.0005)
        self._reward_flat_penalty = reward_config.get("flat_time_penalty", 0.0015)
        self._reward_flat_grace = reward_config.get("flat_time_grace_steps", 6)

    def _sanitize_feature_array(self, values: np.ndarray, *, name: str) -> np.ndarray:
        """Replace NaN/Inf feature values with neutral zeroes."""
        arr = np.asarray(values, dtype=np.float64).copy()
        invalid_mask = ~np.isfinite(arr)
        if invalid_mask.any():
            logging.warning(
                "TradingEnv: replacing %s non-finite values in feature column '%s' with 0.0",
                int(invalid_mask.sum()),
                name,
            )
            arr[invalid_mask] = 0.0
        return arr

    def _sanitize_price_array(self, values: np.ndarray, *, name: str, fallback: np.ndarray | None = None) -> np.ndarray:
        """Replace invalid or implausible price values with nearby valid prices."""
        arr = np.asarray(values, dtype=np.float64).copy()
        invalid_mask = (~np.isfinite(arr)) | (arr <= 0.0)
        finite_positive = arr[np.isfinite(arr) & (arr > 0.0)]
        if finite_positive.size:
            median_price = float(np.median(finite_positive))
            if np.isfinite(median_price) and median_price > 0.0:
                low_cutoff = median_price * 0.1
                high_cutoff = median_price * 10.0
                invalid_mask |= (arr < low_cutoff) | (arr > high_cutoff)
        if invalid_mask.any():
            logging.warning(
                "TradingEnv: replacing %s invalid price values in column '%s'",
                int(invalid_mask.sum()),
                name,
            )
            series = pd.Series(arr)
            if fallback is not None:
                fallback_arr = np.asarray(fallback, dtype=np.float64)
                series.loc[invalid_mask] = fallback_arr[invalid_mask]
            else:
                series.loc[invalid_mask] = np.nan
            arr = np.array(series.ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64), copy=True)
            arr[arr < 0.0] = 0.0
        return arr


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

        # Randomize start position for trajectory diversity in vectorized envs
        if self._max_random_offset > 0:
            offset = self.np_random.integers(0, self._max_random_offset + 1) if hasattr(self, 'np_random') and self.np_random is not None else 0
            self.current_step = self.initial_index + offset
        else:
            self.current_step = self.initial_index
        self.position = 0  # Start flat (no position)
        self.current_contracts = 0
        self.target_allocation_pct = Decimal('0.0')
        self.signed_exposure_pct = Decimal('0.0')
        self.entry_price = Decimal('0.0')
        self.net_worth = self.initial_balance
        self.previous_net_worth = self.initial_balance  # Reset previous net worth for reward calculation
        self.time_in_position = 0  # Reset time in position
        self.time_flat = 0  # Reset time flat
        self.trade_count = 0  # Reset trade count
        self.returns_history = []  # Reset returns history for volatility calculation
        self.max_net_worth = self.initial_balance  # Reset peak for drawdown
        # Reset dynamic SL/TP state
        self.current_sl_multiplier = None
        self.current_tp_multiplier = None
        self.sl_price = Decimal('0.0')
        self.tp_price = Decimal('0.0')
        # Reset fixed SL/TP state
        self.fixed_sl_price = Decimal('0.0')
        self.fixed_tp_price = Decimal('0.0')
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        """
        Constructs the observation at the current time step.
        Uses pre-computed numpy arrays for fast access (no DataFrame lookups).

        Returns:
            np.ndarray: Array with [close_norm, technical_indicators..., position, unrealized_pnl_norm, time_in_position_norm].
        """
        obs = np.empty(self._obs_size, dtype=np.float32)

        # Close norm from pre-computed array
        obs[0] = self._close_norm_array[self.current_step]

        # Indicators from pre-computed matrix (single array slice)
        if self._n_indicators > 0:
            obs[1:1 + self._n_indicators] = self._indicator_matrix[self.current_step]

        # State features
        idx = 1 + self._n_indicators
        obs[idx] = float(self.signed_exposure_pct)
        obs[idx + 1] = self._calculate_unrealized_pnl_normalized()
        obs[idx + 2] = float(np.tanh(self.time_in_position / 50.0))

        # Portfolio drawdown from peak, normalized to [-1, 0]
        if self.max_net_worth > 0:
            drawdown_pct = float((self.net_worth - self.max_net_worth) / self.max_net_worth)
        else:
            drawdown_pct = 0.0
        obs[idx + 3] = max(drawdown_pct, -1.0)

        # Ensure observation is within bounds
        if not np.isfinite(obs).all():
            logging.warning("TradingEnv: encountered non-finite observation values; replacing with zeros")
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        np.clip(obs, self.observation_space.low, self.observation_space.high, out=obs)

        return obs

    def _get_current_price(self) -> Decimal:
        """Get the current price from pre-computed array."""
        return money.to_decimal(self._close_array[self.current_step])

    def _get_current_atr(self) -> Decimal:
        """Get current ATR value from pre-computed array."""
        if self._atr_array is not None:
            return money.to_decimal(self._atr_array[self.current_step])
        return Decimal('0.0')

    def _get_high_low_prices(self) -> Tuple[Decimal, Decimal]:
        """Get high and low prices from pre-computed arrays. Falls back to close price."""
        if self._high_array is not None:
            high = money.to_decimal(self._high_array[self.current_step])
        else:
            high = self._get_current_price()

        if self._low_array is not None:
            low = money.to_decimal(self._low_array[self.current_step])
        else:
            low = self._get_current_price()

        return high, low

    def _set_sl_tp_prices(self, entry_price: Decimal, atr: Decimal) -> None:
        """Calculate and set SL/TP prices based on current multipliers."""
        if self.current_sl_multiplier is None or self.current_tp_multiplier is None:
            return

        # Skip setting SL/TP if ATR is zero or invalid (would cause immediate triggers)
        if atr <= Decimal('0.0'):
            self.sl_price = Decimal('0.0')
            self.tp_price = Decimal('0.0')
            return

        sl_mult = Decimal(str(self.current_sl_multiplier))
        tp_mult = Decimal(str(self.current_tp_multiplier))

        if self.position == 1:  # Long
            self.sl_price = entry_price - (atr * sl_mult)
            self.tp_price = entry_price + (atr * tp_mult)
        elif self.position == -1:  # Short
            self.sl_price = entry_price + (atr * sl_mult)
            self.tp_price = entry_price - (atr * tp_mult)

    def _check_sl_tp_exit(self, high: Decimal, low: Decimal) -> Tuple[bool, str, Decimal]:
        """
        Check if SL or TP has been triggered.

        Returns:
            Tuple of (exit_triggered, exit_reason, exit_price)
        """
        if self.position == 0 or self.current_contracts == 0 or not self.dynamic_sl_tp_enabled:
            return False, "", Decimal('0.0')

        if self.sl_price == Decimal('0.0') and self.tp_price == Decimal('0.0'):
            return False, "", Decimal('0.0')

        if self.position == 1:  # Long
            if low <= self.sl_price:
                return True, "stop_loss", self.sl_price
            if high >= self.tp_price:
                return True, "take_profit", self.tp_price
        elif self.position == -1:  # Short
            if high >= self.sl_price:
                return True, "stop_loss", self.sl_price
            if low <= self.tp_price:
                return True, "take_profit", self.tp_price

        return False, "", Decimal('0.0')

    def _calculate_signed_exposure(self, contracts: int, price: Decimal) -> Decimal:
        """Return signed portfolio exposure for the supplied contract count."""
        if contracts == 0 or price <= Decimal('0.0') or self.net_worth <= Decimal('0.0'):
            return Decimal('0.0')

        notional = price * self._point_value_decimal * Decimal(str(abs(contracts)))
        exposure = notional / self.net_worth
        return exposure if contracts > 0 else -exposure

    def _calculate_target_contracts(self, target_allocation_pct: Decimal, price: Decimal) -> int:
        """Convert a target allocation into a whole-contract MBT quantity."""
        if price <= Decimal('0.0') or self.net_worth <= Decimal('0.0') or target_allocation_pct == Decimal('0.0'):
            return 0

        contract_notional = price * self._point_value_decimal
        if contract_notional <= Decimal('0.0'):
            return 0

        target_notional = self.net_worth * abs(target_allocation_pct)
        contracts = int(target_notional / contract_notional)
        if contracts < 1:
            return 0
        return contracts if target_allocation_pct > 0 else -contracts

    def _sync_position_state(self, price: Decimal) -> None:
        """Synchronize derived position state after a rebalance/exit."""
        if self.current_contracts > 0:
            self.position = 1
        elif self.current_contracts < 0:
            self.position = -1
        else:
            self.position = 0
            self.entry_price = Decimal('0.0')
            self.target_allocation_pct = Decimal('0.0')

        self.signed_exposure_pct = self._calculate_signed_exposure(self.current_contracts, price)

    def _rebalance_position(self, target_contracts: int, current_price: Decimal) -> Tuple[bool, bool, int]:
        """Rebalance to a target contract count.

        Returns:
            (position_changed, is_redundant_action, contracts_traded)
        """
        old_contracts = self.current_contracts
        if target_contracts == old_contracts:
            self.target_allocation_pct = self._calculate_signed_exposure(target_contracts, current_price)
            self.signed_exposure_pct = self.target_allocation_pct
            return False, True, 0

        contracts_traded = 0
        old_direction = 1 if old_contracts > 0 else (-1 if old_contracts < 0 else 0)
        new_direction = 1 if target_contracts > 0 else (-1 if target_contracts < 0 else 0)

        if old_direction == 0:
            self.current_contracts = target_contracts
            self.entry_price = current_price if target_contracts != 0 else Decimal('0.0')
            self.time_in_position = 0 if target_contracts != 0 else 0
            contracts_traded = abs(target_contracts)
        elif new_direction == 0:
            self.current_contracts = 0
            self.entry_price = Decimal('0.0')
            self.time_in_position = 0
            contracts_traded = abs(old_contracts)
        elif old_direction != new_direction:
            contracts_traded = abs(old_contracts) + abs(target_contracts)
            self.current_contracts = target_contracts
            self.entry_price = current_price
            self.time_in_position = 0
        else:
            old_abs = abs(old_contracts)
            new_abs = abs(target_contracts)
            delta = new_abs - old_abs
            contracts_traded = abs(delta)
            if delta > 0:
                weighted_entry = (
                    (self.entry_price * Decimal(str(old_abs))) +
                    (current_price * Decimal(str(delta)))
                ) / Decimal(str(new_abs))
                self.entry_price = weighted_entry
            self.current_contracts = target_contracts

        self.target_allocation_pct = self._calculate_signed_exposure(target_contracts, current_price)
        self._sync_position_state(current_price)
        return True, False, contracts_traded

    def _calculate_unrealized_pnl_normalized(self) -> float:
        """
        Calculate the unrealized P&L normalized to [-1, 1].

        Uses tanh scaling to map P&L to a bounded range.
        A P&L of ~$500 maps to ~0.76, $1000 maps to ~0.96.
        """
        if self.position == 0 or self.current_contracts == 0 or self.entry_price == Decimal('0.0'):
            return 0.0

        current_price = self._get_current_price()

        # Calculate unrealized P&L
        if self.position == 1:  # Long
            price_change = current_price - self.entry_price
        else:  # Short
            price_change = self.entry_price - current_price

        unrealized_pnl = float(price_change * self._point_value_decimal * Decimal(str(abs(self.current_contracts))))

        # Normalize using tanh with scaling factor of $500
        # This maps $500 -> ~0.76, $1000 -> ~0.96, -$500 -> ~-0.76
        normalized = float(np.tanh(unrealized_pnl / 500.0))

        return normalized

    def step(self, action):
        """
        Apply an action, update the portfolio, and return the next observation.

        Args:
            action: Action to execute. Either int (Discrete) or array (MultiDiscrete).
                   Discrete: 7 target-allocation actions
                   MultiDiscrete: [position_action, sl_idx, tp_idx]

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        info = {}
        info["action_masked"] = False

        # Parse action based on action space type
        if self.dynamic_sl_tp_enabled:
            position_action = int(action[0])
            sl_idx = int(action[1])
            tp_idx = int(action[2])
        else:
            position_action = int(action)
            sl_idx, tp_idx = None, None

        # Store the previous net worth for reward calculation
        self.previous_net_worth = self.net_worth

        # Get current price and high/low for SL/TP checking
        current_price = self._get_current_price()

        # Track old position state to detect changes for transaction cost and profit calculation
        old_position = self.position
        old_entry_price = self.entry_price
        old_contracts = self.current_contracts
        position_changed = False
        is_redundant_action = False
        sl_tp_exit_reason = ""
        sl_tp_exit_price = Decimal('0.0')
        contracts_traded = 0

        # Check for SL/TP exit BEFORE processing the action
        if self.position != 0 and self.dynamic_sl_tp_enabled:
            high, low = self._get_high_low_prices()
            exit_triggered, exit_reason, exit_price = self._check_sl_tp_exit(high, low)
            if exit_triggered:
                # Force exit at SL/TP price
                sl_tp_exit_reason = exit_reason
                sl_tp_exit_price = exit_price

                # Calculate P&L at exit price
                if self.position == 1:  # Long
                    price_change = exit_price - self.entry_price
                else:  # Short
                    price_change = self.entry_price - exit_price

                dollar_change = price_change * self._point_value_decimal * Decimal(str(abs(self.current_contracts)))
                self.net_worth += dollar_change

                # Close position
                old_position = self.position
                old_entry_price = self.entry_price
                old_contracts = self.current_contracts
                self.current_contracts = 0
                self.position = 0
                self.target_allocation_pct = Decimal('0.0')
                self.signed_exposure_pct = Decimal('0.0')
                self.entry_price = Decimal('0.0')
                self.time_in_position = 0
                self.trade_count += 1
                position_changed = True
                contracts_traded = abs(old_contracts)

                # Reset SL/TP state
                self.current_sl_multiplier = None
                self.current_tp_multiplier = None
                self.sl_price = Decimal('0.0')
                self.tp_price = Decimal('0.0')

                info["sl_tp_exit"] = exit_reason
                info["sl_tp_exit_price"] = float(exit_price)

        # Only process action if we didn't get stopped out
        if not sl_tp_exit_reason:
            target_allocation = Decimal(str(target_allocation_for_action(position_action)))
            target_contracts = self._calculate_target_contracts(target_allocation, current_price)
            position_changed, is_redundant_action, contracts_traded = self._rebalance_position(
                target_contracts=target_contracts,
                current_price=current_price,
            )
            if position_changed:
                self.trade_count += 1

                # Set dynamic SL/TP on any non-flat rebalance
                if self.dynamic_sl_tp_enabled and self.position != 0 and sl_idx is not None and tp_idx is not None:
                    self.current_sl_multiplier = float(self.sl_multipliers[sl_idx])
                    self.current_tp_multiplier = float(self.tp_multipliers[tp_idx])
                    atr = self._get_current_atr()
                    self._set_sl_tp_prices(self.entry_price, atr)
                    info["sl_multiplier"] = self.current_sl_multiplier
                    info["tp_multiplier"] = self.current_tp_multiplier
                    info["sl_price"] = float(self.sl_price)
                    info["tp_price"] = float(self.tp_price)
                elif self.position == 0:
                    self.fixed_sl_price = Decimal('0.0')
                    self.fixed_tp_price = Decimal('0.0')

        # Set fixed ATR-based SL/TP prices on position entry
        if position_changed and self.position != 0:
            atr = self._get_current_atr()
            if self.position == 1:  # Long
                if self.fixed_sl_enabled:
                    if self.fixed_sl_atr_mult and atr > 0:
                        self.fixed_sl_price = self.entry_price - (atr * self.fixed_sl_atr_mult)
                    elif self.fixed_sl_pct is not None:
                        self.fixed_sl_price = self.entry_price * (Decimal('1.0') - self.fixed_sl_pct)
                if self.fixed_tp_enabled:
                    if self.fixed_tp_atr_mult and atr > 0:
                        self.fixed_tp_price = self.entry_price + (atr * self.fixed_tp_atr_mult)
                    elif self.fixed_tp_pct is not None:
                        self.fixed_tp_price = self.entry_price * (Decimal('1.0') + self.fixed_tp_pct)
            elif self.position == -1:  # Short
                if self.fixed_sl_enabled:
                    if self.fixed_sl_atr_mult and atr > 0:
                        self.fixed_sl_price = self.entry_price + (atr * self.fixed_sl_atr_mult)
                    elif self.fixed_sl_pct is not None:
                        self.fixed_sl_price = self.entry_price * (Decimal('1.0') + self.fixed_sl_pct)
                if self.fixed_tp_enabled:
                    if self.fixed_tp_atr_mult and atr > 0:
                        self.fixed_tp_price = self.entry_price - (atr * self.fixed_tp_atr_mult)
                    elif self.fixed_tp_pct is not None:
                        self.fixed_tp_price = self.entry_price * (Decimal('1.0') - self.fixed_tp_pct)

        # Apply execution costs if position changed: commission + spread + slippage
        transaction_cost_applied = Decimal('0.0')
        if position_changed:
            # Commission (per fill)
            if self.transaction_cost > Decimal('0.0'):
                transaction_cost_applied += self.transaction_cost * Decimal(str(contracts_traded))

            # Spread cost: half-spread per fill (crossing the bid-ask)
            spread_cost = self._spread_points * self._point_value_decimal * Decimal(str(contracts_traded))
            transaction_cost_applied += spread_cost

            # Slippage: time-of-day dependent, per fill
            step_idx = min(self.current_step, len(self._slippage_points) - 1)
            slippage_pts = Decimal(str(self._slippage_points[step_idx]))
            slippage_cost = slippage_pts * self._point_value_decimal * Decimal(str(contracts_traded))
            transaction_cost_applied += slippage_cost

            self.net_worth -= transaction_cost_applied

        # Increment time in position if holding, or time flat if no position
        if self.position != 0 and not position_changed:
            self.time_in_position += 1
            self.time_flat = 0  # Reset flat counter when in position
        elif self.position == 0:
            self.time_flat += 1
            self.time_in_position = 0  # Reset position counter when flat

        # Advance to next step
        self.current_step += 1

        # Calculate portfolio value change from the post-action position over the next bar.
        # Rebalances at the current price should affect exposure for the upcoming price move.
        # If a fixed SL/TP is triggered on the next bar, only mark to the exit price once.
        if self.current_step < self.total_steps and not sl_tp_exit_reason:
            next_price = self._get_current_price()
            fixed_exit = False
            exit_price = Decimal('0.0')

            if self.position != 0:
                # Check fixed SL/TP against the next bar close proxy. Use the exit price
                # directly for MTM so we do not double-count entry-to-exit P&L.
                if self.fixed_sl_enabled and self.fixed_sl_price > 0:
                    if (self.position == 1 and next_price <= self.fixed_sl_price) or \
                       (self.position == -1 and next_price >= self.fixed_sl_price):
                        fixed_exit = True
                        exit_price = self.fixed_sl_price
                        info["fixed_sl_triggered"] = True

                if not fixed_exit and self.fixed_tp_enabled and self.fixed_tp_price > 0:
                    if (self.position == 1 and next_price >= self.fixed_tp_price) or \
                       (self.position == -1 and next_price <= self.fixed_tp_price):
                        fixed_exit = True
                        exit_price = self.fixed_tp_price
                        info["fixed_tp_triggered"] = True

            if self.position != 0 and self.current_contracts != 0:
                mark_price = exit_price if fixed_exit else next_price
                if self.position == 1:
                    price_change = mark_price - current_price
                else:
                    price_change = current_price - mark_price

                dollar_change = price_change * self._point_value_decimal * Decimal(str(abs(self.current_contracts)))
                self.net_worth += dollar_change

            if fixed_exit:
                self.current_contracts = 0
                self.position = 0
                self.entry_price = Decimal('0.0')
                self.time_in_position = 0
                self.trade_count += 1
                self.fixed_sl_price = Decimal('0.0')
                self.fixed_tp_price = Decimal('0.0')
                self.target_allocation_pct = Decimal('0.0')
                self.signed_exposure_pct = Decimal('0.0')

        # Update max net worth for drawdown tracking
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        # Ensure net_worth doesn't go below a minimum threshold (e.g., 1% of initial balance)
        min_balance = self.initial_balance * self._min_balance_pct_decimal
        if not self.net_worth.is_finite():
            logging.warning("TradingEnv: encountered non-finite net worth; clamping to minimum balance")
            self.net_worth = min_balance
            info["non_finite_net_worth"] = True
        if not self.max_net_worth.is_finite():
            self.max_net_worth = self.initial_balance
        if self.net_worth < min_balance:
            self.net_worth = min_balance
            info["hit_minimum_balance"] = True

        terminated = self.current_step >= self.total_steps
        truncated = False

        if terminated:
            self.current_step = self.total_steps  # Clamp to last valid index

        self._sync_position_state(self._get_current_price() if self.current_step <= self.total_steps else current_price)

        # Calculate RISK-ADJUSTED REWARD with scalping incentives
        exit_price_for_reward = sl_tp_exit_price if sl_tp_exit_price > Decimal('0.0') else current_price
        reward = self._calculate_risk_adjusted_reward(
            position_changed=position_changed,
            transaction_cost_applied=float(transaction_cost_applied),
            is_redundant_action=is_redundant_action,
            old_position=old_position,
            old_entry_price=old_entry_price,
            exit_price=exit_price_for_reward
        )

        obs = self._get_obs()

        # Add info about current state
        info["position"] = self.position
        info["old_position"] = old_position
        info["position_changed"] = position_changed
        info["reward"] = float(reward)
        info["net_worth"] = float(self.net_worth)
        info["position_size"] = self.position_size
        info["current_contracts"] = self.current_contracts
        info["old_contracts"] = old_contracts
        info["contracts_traded"] = contracts_traded
        info["signed_exposure"] = float(self.signed_exposure_pct)
        info["avg_entry_price"] = float(self.entry_price) if self.entry_price > Decimal('0.0') else 0.0
        info["target_allocation_pct"] = float(self.target_allocation_pct)
        info["time_in_position"] = self.time_in_position
        info["trade_count"] = self.trade_count
        info["transaction_cost"] = float(transaction_cost_applied)
        info["max_net_worth"] = float(self.max_net_worth)

        return obs, float(reward), terminated, truncated, info

    def _calculate_risk_adjusted_reward(self, position_changed: bool, transaction_cost_applied: float,
                                         is_redundant_action: bool, old_position: int = 0,
                                         old_entry_price: Decimal = Decimal('0.0'),
                                         exit_price: Decimal = Decimal('0.0')) -> float:
        """
        Portfolio-return-based reward aligned with Sortino/Calmar optimization.

        Primary signal: log change in net worth (captures P&L, costs, and compounding).
        Asymmetric penalty: losses penalized at 1.7x gains.
        Calm holding bonus: small reward for staying in position during low-vol regimes.
        Penalties: drawdown increase, excessive turnover.

        Returns:
            float: Reward signal
        """
        base_scale = self._reward_base_scale
        loss_multiplier = self._reward_loss_multiplier
        dd_threshold = self._reward_dd_threshold
        dd_penalty = self._reward_dd_penalty
        turnover_pen = self._reward_turnover_pen
        calm_bonus = self._reward_calm_bonus

        # Primary signal: log change in equity
        if self.previous_net_worth > 0 and self.net_worth > 0:
            log_return = float(np.log(float(self.net_worth) / float(self.previous_net_worth)))
        else:
            log_return = 0.0

        # Scale log return to useful reward magnitude
        # For NQ with $100k account: a 10-point move = $200 = 0.2% = log_return ~0.002
        # Scale by base_scale so 0.2% move → reward ~1.0
        reward = log_return * base_scale

        # Asymmetric downside penalty: additive extra penalty on losses
        # loss_multiplier=0.7 → losses at 1.7x base, gains at 1x
        if log_return < 0:
            reward += log_return * base_scale * loss_multiplier

        # Drawdown penalty: penalize increases in drawdown from peak
        if self.max_net_worth > self.initial_balance:
            current_dd = float((self.max_net_worth - self.net_worth) / self.max_net_worth)
            if current_dd > dd_threshold:
                reward -= current_dd * dd_penalty

        # Calm holding bonus: reward holding LONG in calm, low-drawdown regimes
        # This directly counters the "trade too little" problem in bull windows
        if self.position == 1 and calm_bonus > 0:
            rolling_dd_val = 0.0
            vol_pct_val = 50.0
            if self._rolling_dd_idx is not None:
                rolling_dd_val = float(self._indicator_matrix[self.current_step, self._rolling_dd_idx])
            if self._vol_pct_idx is not None:
                vol_pct_val = float(self._indicator_matrix[self.current_step, self._vol_pct_idx])

            # Stronger bonus when near highs and volatility is low
            if rolling_dd_val > -2.5 and vol_pct_val < 68.0:
                reward += calm_bonus * 2.0  # Double the bonus for holding long in good conditions

        # Turnover penalty: discourage excessive trading (cost already in net_worth, this is extra signal)
        if position_changed:
            reward -= turnover_pen

        # Inactivity penalty: gently discourage staying flat for long stretches.
        if self.position == 0 and self.time_flat > self._reward_flat_grace and self._reward_flat_penalty > 0:
            reward -= self._reward_flat_penalty * float(self.time_flat - self._reward_flat_grace)

        if not np.isfinite(reward):
            logging.warning("TradingEnv: encountered non-finite reward; replacing with 0.0")
            reward = 0.0

        # Clamp reward to prevent gradient explosion from extreme values
        return max(min(reward, 10.0), -10.0)
