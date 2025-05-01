from ib_insync import IB, Future, MarketOrder, util, Order, LimitOrder, StopOrder, BracketOrder
import datetime
from datetime import UTC  # Add this import for the UTC timezone
import sys
import pandas as pd
import numpy as np
import pytz
from decimal import Decimal
import time
import logging
from pathlib import Path
import os
import json
import threading
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
import argparse

# Import stable_baselines3 for loading the trained model
from stable_baselines3 import PPO

# Import money module and config for risk management
import money
from config import config

# Import process_technical_indicators and ensure_numeric from get_data
from get_data import process_technical_indicators, ensure_numeric

# Import normalization module
from normalization import load_scaler, normalize_data, get_standardized_column_names

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Global variables for bar buffer and model trader
bar_buffer = []
model_trader = None
last_execution_time = None
last_data_timestamp = datetime.now()
is_data_flowing = False
reconnection_attempts = 0
MAX_RECONNECTION_ATTEMPTS = 3
DATA_FLOW_THRESHOLD = 60  # seconds
MIN_EXECUTION_INTERVAL = 10  # seconds

# Constants for bar synchronization
FIVE_SEC_PER_BAR = 5
BARS_PER_FIVE_MIN = (5 * 60) // FIVE_SEC_PER_BAR  # 60
UTC = pytz.UTC           # convenience alias
ROUND_TO = timedelta(minutes=5)

# Heartbeat monitoring globals
heartbeat_interval = 60  # Seconds between heartbeat checks
state_file = "trader_state.pkl"

# Globals for real-time bar handling
bar_buckets = defaultdict(list)  # Organize bars by 5-minute intervals

# Function to aggregate a list of 5-second bars into one 5-minute bar.
def aggregate_bars(bars):
    """
    Given a list of real-time bars (5-sec each), compute a 5-minute bar.
    """
    if not bars:
        return None
    
    # Handle IB bars which may have different attribute names
    try:    
        # Check if we have IB bar objects
        if hasattr(bars[0], 'open_'):
            open_price = bars[0].open_
            high_price = max(bar.high for bar in bars)
            low_price = min(bar.low for bar in bars)
            close_price = bars[-1].close
            volume = sum(bar.volume for bar in bars)
            
            # Handle different time formats
            if isinstance(bars[0].time, datetime):
                start_time = bars[0].time
                end_time = bars[-1].time
            else:
                try:
                    start_time = datetime.fromtimestamp(bars[0].time)
                    end_time = datetime.fromtimestamp(bars[-1].time)
                except (TypeError, ValueError):
                    start_time = datetime.now() - timedelta(minutes=5)
                    end_time = datetime.now()
        else:
            # Handle dictionary format
            open_price = bars[0]['open']
            high_price = max(bar['high'] for bar in bars)
            low_price = min(bar['low'] for bar in bars)
            close_price = bars[-1]['close']
            volume = sum(bar['volume'] for bar in bars)
            
            # Get time values
            start_time = bars[0].get('time', datetime.now() - timedelta(minutes=5))
            end_time = bars[-1].get('time', datetime.now())
        
        # Ensure times are timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=UTC)
        else:
            start_time = start_time.astimezone(UTC)
            
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=UTC)
        else:
            end_time = end_time.astimezone(UTC)
            
        # Create the interval end time as key for the bar
        interval_end = end_of_interval(end_time)
        
        # Create and return the aggregated bar
        return {
            'time': interval_end,  # Use interval end for indexing
            'start': start_time,
            'end': end_time,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        }
    except Exception as e:
        logger.error(f"Error aggregating bars: {e}")
        return None

def get_interval_key(timestamp):
    """
    Convert a timestamp to its 5-minute interval key.
    Example: 09:32:45 -> '09:30'
    
    Handles timezone-aware datetime objects by removing the timezone info.
    """
    # Handle timezone-aware datetimes by removing timezone info if present
    if hasattr(timestamp, 'tzinfo') and timestamp.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=None)
        
    # Floor to nearest 5-minute interval
    minute = (timestamp.minute // 5) * 5
    interval_time = timestamp.replace(minute=minute, second=0, microsecond=0)
    return interval_time.strftime('%Y-%m-%d %H:%M')

def end_of_interval(ts: datetime, width=ROUND_TO) -> datetime:
    """
    Return the *end* of the current `width`-minute interval.
    `ts` must be timezone-aware (UTC recommended).
    """
    ts = ts.astimezone(UTC)
    floored = ts - timedelta(
        minutes=ts.minute % (width.seconds // 60),
        seconds=ts.second,
        microseconds=ts.microsecond,
    )
    return floored + width  # ⬅️  end, not start

def synchronize_bars():
    """
    Check if we have complete 5-minute bars to process.
    Returns the latest complete bar if available, None otherwise.
    """
    now_utc = datetime.now(UTC) - timedelta(seconds=1)
    last_end = end_of_interval(now_utc)
    key = last_end.strftime("%Y-%m-%d %H:%M")

    if key in bar_buckets and len(bar_buckets[key]) == BARS_PER_FIVE_MIN:
        logger.info(f"Processing synchronized 5-minute bar for interval {key} with exactly {BARS_PER_FIVE_MIN} 5-second bars")
        bars = bar_buckets.pop(key)
        
        # Clean up old buckets to prevent memory growth
        current_time = datetime.now(UTC)
        for old_key in list(bar_buckets.keys()):
            try:
                old_time = datetime.strptime(old_key, '%Y-%m-%d %H:%M').replace(tzinfo=UTC)
                if (current_time - old_time).total_seconds() > 3600:  # Older than 1 hour
                    del bar_buckets[old_key]
                    logger.debug(f"Cleaned up old bucket: {old_key}")
            except Exception as e:
                logger.warning(f"Error cleaning up old bar bucket {old_key}: {e}")
        
        return aggregate_bars(bars)

    # Calculate and log the percentage completion of the current interval
    current_interval = end_of_interval(datetime.now(UTC))
    current_key = current_interval.strftime('%Y-%m-%d %H:%M')
    
    # Count bars in the current interval
    current_bars_count = len(bar_buckets.get(current_key, []))
    
    # Calculate elapsed seconds in this interval
    now = datetime.now(UTC)
    interval_start = current_interval - ROUND_TO
    elapsed_seconds = (now - interval_start).total_seconds()
    total_seconds = 5 * 60  # 5 minutes = 300 seconds
    time_percent = min(100, round((elapsed_seconds / total_seconds) * 100, 1))
    
    # Expected bars based on elapsed time
    expected_bars = int(elapsed_seconds / FIVE_SEC_PER_BAR)
    bars_percent = min(100, round((current_bars_count / BARS_PER_FIVE_MIN) * 100, 1))
    
    logger.info(f"Current interval: {current_key} | Time: {time_percent}% complete | "
                f"Bars: {current_bars_count}/{BARS_PER_FIVE_MIN} ({bars_percent}%) | "
                f"Next prediction at interval completion")
    
    return None

class ModelTrader:
    def __init__(self, ib_instance, model_path="best_model", use_risk_management=True):
        """
        Initialize the ModelTrader with an Interactive Brokers instance and model.
        
        Args:
            ib_instance: An instance of the Interactive Brokers client
            model_path: Path to the trained model folder
            use_risk_management: Whether to use risk management
        """
        logger.info(f"Initializing ModelTrader with model at {model_path}")
        
        self.ib = ib_instance
        self.model_path = model_path
        self.model = None
        self.current_position = 0  # Current position (quantity)
        self.contract = None  # Current trading contract
        self.order_ids = []  # Track order IDs
        self.bar_history = []  # Store historical bars
        self.min_bars_needed = 50  # Minimum bars needed for indicator calculation
        self.execution_counter = 0  # Count executions for managing trade execution ID collisions
        
        # Load risk management parameters
        self.use_risk_management = use_risk_management
        self._initialize_risk_parameters()
        
        # Initialize trading data
        self.last_candle_execution_time = None
        self.last_bar_end_time = None
        self.last_action = None
        self.pending_orders = {}  # Map of order IDs to order details
        self.locked_contract = None  # Lock on a specific contract for continuous trading
        
        # Used for tracking and synchronizing ticks to form OHLC bars
        self.tick_data = {}  # Store ticks for aggregating bars
        self.current_bar = {}  # Current incomplete bar
        self.start_time = None  # Track when we started collecting ticks
        
        # Enable/disable settings
        self.use_ib_bars = True  # Use IB's bar mechanism instead of tick aggregation
        self.wait_for_bar_completion = True  # Wait for bar to complete before executing trades
        
        # Diagnostics
        self.diagnostic_mode = True  # Enable diagnostic mode for troubleshooting
        
        # Get list of enabled indicators from saved file (ensures consistency with training)
        # This is critical to get the exact same indicators in the exact same order
        self.enabled_indicators = self._load_enabled_indicators_from_file()
        
        # Load the model 
        self.load_model()
        
        # Load the scaler for normalization
        self._load_normalization_scalers()
        
        # Store feature range for normalization
        self.feature_range = config.get("normalization", {}).get("feature_range", (-1, 1))
        
        # Track the required features for the model
        self.required_features = ["close_norm"]
        self.required_features.extend(self.enabled_indicators)
        
        # Initialize missing fields
        self.expected_position = 0  # Expected absolute position size
        self.active_orders = {}  # Active orders by order ID
        self.order_status = {}  # Track order statuses
        self.last_position_check = datetime.now()  # Last time positions were verified
        self.position_check_interval = 60  # Check positions every 60 seconds
        
        logger.info(f"ModelTrader initialized with {len(self.enabled_indicators)} indicators")
    
    def _initialize_risk_parameters(self):
        """
        Initialize risk management parameters from config
        """
        # Get risk management config
        risk_config = config.get("risk_management", {})
        risk_enabled = risk_config.get("enabled", False)
        
        # Default values
        self.stop_loss_pct = None
        self.take_profit_pct = None
        self.trailing_stop_pct = None
        self.position_size = 1
        
        if not self.use_risk_management or not risk_enabled:
            logger.info("Risk management is disabled")
            return

        # Initialize stop loss if enabled
        stop_loss_config = risk_config.get("stop_loss", {})
        if stop_loss_config.get("enabled", False):
            self.stop_loss_pct = stop_loss_config.get("percentage", 0.5)
            logger.info(f"Stop loss enabled at {self.stop_loss_pct}%")
        else:
            logger.info("Stop loss is disabled")

        # Initialize take profit if enabled
        take_profit_config = risk_config.get("take_profit", {})
        if take_profit_config.get("enabled", False):
            self.take_profit_pct = take_profit_config.get("percentage", 1.0)
            logger.info(f"Take profit enabled at {self.take_profit_pct}%")
        else:
            logger.info("Take profit is disabled")

        # Initialize trailing stop if enabled
        trailing_stop_config = risk_config.get("trailing_stop", {})
        if trailing_stop_config.get("enabled", False):
            self.trailing_stop_pct = trailing_stop_config.get("percentage", 0.3)
            logger.info(f"Trailing stop enabled at {self.trailing_stop_pct}%")
        else:
            logger.info("Trailing stop is disabled")

        # Initialize position sizing if enabled
        position_sizing_config = risk_config.get("position_sizing", {})
        if position_sizing_config.get("enabled", False):
            self.position_size = position_sizing_config.get("size", 1)
            logger.info(f"Position sizing enabled, using {self.position_size} contracts per trade")
        else:
            self.position_size = config["environment"].get("position_size", 1)
            logger.info(f"Using fixed position size of {self.position_size} contracts per trade")
    
    def _load_enabled_indicators_from_file(self):
        """
        Load enabled indicators from the enabled_indicators.json file if it exists,
        otherwise fall back to _get_enabled_indicators method.
        """
        # Try to load from enabled_indicators.json if it exists
        model_dir = os.path.dirname(self.model_path)
        indicator_files = [
            os.path.join(model_dir, "enabled_indicators.json"),
            "enabled_indicators.json",
        ]
        
        for file_path in indicator_files:
            if os.path.exists(file_path):
                try:
                    logger.info(f"Loading indicators from {file_path}")
                    with open(file_path, 'r') as f:
                        indicators = json.load(f)
                    
                    # Check if time-based features are enabled in config but missing from file
                    if config.get("indicators", {}).get("day_of_week", {}).get("enabled", True):
                        if "DOW_SIN" not in indicators:
                            logger.warning("DOW_SIN needed but not in indicators file, adding it")
                            indicators.append("DOW_SIN")
                        if "DOW_COS" not in indicators:
                            logger.warning("DOW_COS needed but not in indicators file, adding it")
                            indicators.append("DOW_COS")
                    
                    if config.get("indicators", {}).get("minutes_since_open", {}).get("enabled", False):
                        if "MSO_SIN" not in indicators:
                            logger.warning("MSO_SIN needed but not in indicators file, adding it")
                            indicators.append("MSO_SIN")
                        if "MSO_COS" not in indicators:
                            logger.warning("MSO_COS needed but not in indicators file, adding it")
                            indicators.append("MSO_COS")
                    
                    # Log the loaded indicators for debugging
                    logger.info(f"Loaded {len(indicators)} indicators from file: {indicators}")
                    return indicators
                except Exception as e:
                    logger.warning(f"Error loading indicators from {file_path}: {e}")
        
        # Fall back to _get_enabled_indicators if no file is found
        logger.info("No indicators file found, using config-based indicators")
        return self._get_enabled_indicators()
    
    def _get_enabled_indicators(self):
        """Get list of enabled indicators from config"""
        enabled_indicators = []
        
        # Check configuration
        try:
            indicators_config = config.get("indicators", {})
            
            # Debug: log indicators config
            logger.info(f"DEBUG: Indicators config from config.yaml: {indicators_config}")
            
            # Check each indicator
            if indicators_config.get("rsi", {}).get("enabled", False):
                enabled_indicators.append("RSI")
            if indicators_config.get("cci", {}).get("enabled", False):
                enabled_indicators.append("CCI")
            if indicators_config.get("macd", {}).get("enabled", False):
                enabled_indicators.extend(["MACD", "MACD_SIGNAL", "MACD_HIST"])
            if indicators_config.get("atr", {}).get("enabled", False):
                enabled_indicators.append("ATR")
            if indicators_config.get("supertrend", {}).get("enabled", False):
                enabled_indicators.append("SUPERTREND")
            if indicators_config.get("adx", {}).get("enabled", False):
                enabled_indicators.append("ADX")
            if indicators_config.get("adx_pos", {}).get("enabled", False):
                enabled_indicators.append("ADX_POS")
            if indicators_config.get("adx_neg", {}).get("enabled", False):
                enabled_indicators.append("ADX_NEG")
            if indicators_config.get("stoch_k", {}).get("enabled", False):
                enabled_indicators.append("STOCH_K")
            if indicators_config.get("stoch_d", {}).get("enabled", False):
                enabled_indicators.append("STOCH_D")
            if indicators_config.get("roc", {}).get("enabled", False):
                enabled_indicators.append("ROC")
            if indicators_config.get("williams_r", {}).get("enabled", False):
                enabled_indicators.append("WILLIAMS_R")
            if indicators_config.get("sma", {}).get("enabled", False):
                enabled_indicators.append("SMA_NORM")
            if indicators_config.get("ema", {}).get("enabled", False):
                enabled_indicators.append("EMA_NORM")
            if indicators_config.get("disparity", {}).get("enabled", False):
                enabled_indicators.append("DISPARITY")
            if indicators_config.get("obv", {}).get("enabled", False):
                enabled_indicators.append("OBV_NORM")
            if indicators_config.get("cmf", {}).get("enabled", False):
                enabled_indicators.append("CMF")
            if indicators_config.get("psar", {}).get("enabled", False):
                enabled_indicators.extend(["PSAR_NORM", "PSAR_DIR"])
            if indicators_config.get("volume", {}).get("enabled", False):
                enabled_indicators.append("VOLUME_NORM")
            if indicators_config.get("vwap", {}).get("enabled", False):
                enabled_indicators.append("VWAP_NORM")
            if indicators_config.get("minutes_since_open", {}).get("enabled", False):
                enabled_indicators.extend(["MSO_SIN", "MSO_COS"])
            if indicators_config.get("day_of_week", {}).get("enabled", False):
                enabled_indicators.extend(["DOW_SIN", "DOW_COS"])
            
            
            # Debug: log indicators from config
            logger.info(f"DEBUG: Indicators from config: {enabled_indicators}")
            
            # Try to load from enabled_indicators.json if it exists
            indicators_file = os.path.join(os.path.dirname(self.model_path), "enabled_indicators.json")
            if os.path.exists(indicators_file):
                logger.info(f"Loading indicators from {indicators_file}")
                try:
                    with open(indicators_file, "r") as f:
                        file_indicators = json.load(f)
                    
                    # Debug: log indicators from file
                    logger.info(f"DEBUG: Original indicators from file: {file_indicators}")
                    logger.info(f"DEBUG: Count from file: {len(file_indicators)}")
                        
                    # Use the indicators from file instead
                    return file_indicators
                except Exception as e:
                    logger.error(f"Error loading indicators from file: {e}")
            
            return enabled_indicators
        except Exception as e:
            logger.error(f"Error getting enabled indicators: {e}")
            return ["RSI", "CCI"]  # Default minimum set
    
    def load_model(self):
        """Load the trained model from the specified path."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.model = PPO.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            
            # Diagnose expected model features
            self.diagnose_model_features()
            
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def diagnose_model_features(self):
        """
        Diagnose any issues with model features and try to fix them.
        """
        logger.info(f"Model observation space shape: {self.model.observation_space.shape}")
        
        # Check if model has a features extractor
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'features_extractor'):
            logger.info(f"Model has a features extractor: {self.model.policy.features_extractor}")
        
        # Check if the model has env_code saved in it
        if hasattr(self.model, 'env_code_text'):
            logger.info("Model has stored environment information")
            
        # Get the expected number of features from the observation space
        expected_features = self.model.observation_space.shape[0]
        logger.info(f"Model expects {expected_features} features")
        
        try:
            # Get the enabled indicators (either from instance attribute or method call)
            indicators = getattr(self, 'enabled_indicators', None)
            if indicators is None:
                indicators = self._get_enabled_indicators()
                
            # Debug: log all indicators before filtering
            logger.info(f"DEBUG: All indicators before filtering: {indicators}")
            logger.info(f"DEBUG: Total count before filtering: {len(indicators)}")
            
            # Expected number of features should be: close_norm + indicators + position = n+2
            expected_indicators_count = expected_features - 2  # Subtract close_norm and position
            
            # Debug: log the breakdown of expected feature composition
            logger.info(f"DEBUG: Expected feature composition: {expected_features} total = {expected_indicators_count} indicators + close_norm + position")
            
            # Check if the number of indicators matches
            if len(indicators) != expected_indicators_count:
                logger.error(f"Feature count mismatch: Model expects {expected_indicators_count} indicators, but we have {len(indicators)}")
                # Debug: analyze the exact mismatch
                logger.error(f"DEBUG: Mismatch amount: {expected_indicators_count - len(indicators)}")
                self._diagnose_feature_mismatch(expected_indicators_count - len(indicators), indicators)
                return False
            else:
                logger.info(f"Feature count matches: {len(indicators)} indicators")
                return True
                
        except Exception as e:
            logger.error(f"Error diagnosing model features: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _diagnose_feature_mismatch(self, feature_delta, current_indicators):
        """
        Attempt to diagnose feature mismatches between what the model expects and what we're providing.
        
        Args:
            feature_delta: Number of missing features (positive) or extra features (negative)
            current_indicators: List of indicators we're currently using
        """
        try:
            # Look for metadata files to find actual training indicators
            model_dir = os.path.dirname(self.model_path)
            indicator_files = [
                os.path.join(model_dir, "enabled_indicators.json"),
                "enabled_indicators.json",
            ]
            
            # Debug: log all potential indicator files we'll check
            logger.info(f"DEBUG: Checking indicator files: {indicator_files}")
            
            for file_path in indicator_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            trained_indicators = json.load(f)
                        
                        # Debug: log the contents of the indicator file    
                        logger.info(f"DEBUG: Found indicator file {file_path} with {len(trained_indicators)} indicators: {trained_indicators}")
                            
                        # Find indicators in the training file that aren't in our current list
                        missing_from_current = [ind for ind in trained_indicators 
                                                if ind not in current_indicators]
                        
                        # Find indicators in our current list that aren't in the training file
                        extra_in_current = [ind for ind in current_indicators 
                                            if ind not in trained_indicators]
                        
                        # Debug: show exact comparison between trained and current
                        logger.info(f"DEBUG: Trained indicators not in current set: {missing_from_current}")
                        logger.info(f"DEBUG: Current indicators not in trained set: {extra_in_current}")
                        
                        if missing_from_current:
                            logger.error(f"FOUND MISSING INDICATORS from training file {file_path}:")
                            logger.error(f"Missing: {missing_from_current}")
                            logger.error(f"Please ensure these are in your enabled indicators list")
                            
                            # Store for reference
                            self.missing_indicators = missing_from_current
                        
                        if extra_in_current:
                            logger.error(f"FOUND EXTRA INDICATORS not in training file {file_path}:")
                            logger.error(f"Extra: {extra_in_current}")
                            logger.error(f"These should not be in your enabled indicators list")
                            
                            # Store for reference
                            self.extra_indicators = extra_in_current
                            
                        return
                    except Exception as e:
                        logger.warning(f"Error reading indicator file {file_path}: {e}")
            
            # If we get here, we couldn't find the indicator files, so fall back to a basic diagnostic
            # with common indicators
            common_indicators = [
                "RSI", "CCI", "ATR", "MACD", "MACD_SIGNAL", "MACD_HIST",
                "ADX", "ADX_POS", "ADX_NEG", "STOCH_K", "STOCH_D", 
                "ROC", "WILLIAMS_R", "SMA_NORM", "EMA_NORM", "DISPARITY", 
                "OBV_NORM", "CMF", "PSAR_NORM", "PSAR_DIR", "VOLUME_MA",
                "DOW_SIN", "DOW_COS", "MSO_SIN", "MSO_COS", "VWAP_NORM",
                "SUPERTREND"
            ]
            
            # Debug: Log common indicators not in current indicators
            missing_common = [ind for ind in common_indicators if ind not in current_indicators]
            logger.info(f"DEBUG: Common indicators not in current set: {missing_common}")
            
            if feature_delta > 0:
                # We're missing indicators
                # Find indicators that are commonly used but not in our current list
                possible_missing = [ind for ind in common_indicators if ind not in current_indicators]
                
                logger.warning(f"Possible missing indicators: {possible_missing[:feature_delta]}")
                logger.warning("Consider adding these indicators to your config.yaml file")
            else:
                # We have extra indicators
                possible_extra = [ind for ind in current_indicators if ind not in common_indicators]
                
                logger.warning(f"Possible extra indicators: {possible_extra}")
                logger.warning("Consider removing these from your config.yaml file")
        
        except Exception as e:
            logger.error(f"Error in feature mismatch diagnosis: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def set_active_contract(self, contract):
        """Set the active trading contract."""
        self.contract = contract
        logger.info(f"Active contract set to {contract}")
    
    def save_state(self):
        """
        Save the current state of the trader including bar history.
        """
        try:
            # Create a dictionary with the current state
            state = {
                "bar_history": self.bar_history,
                "position": self.current_position,
                "contract": {
                    "symbol": self.contract.symbol if self.contract else None,
                    "secType": self.contract.secType if self.contract else None,
                    "exchange": self.contract.exchange if self.contract else None,
                    "lastTradeDateOrContractMonth": self.contract.lastTradeDateOrContractMonth if self.contract else None,
                }
            }
            
            # Save to a file
            with open("trader_state.pkl", "wb") as f:
                pickle.dump(state, f)
                
            logger.info(f"State saved successfully with {len(self.bar_history)} bars")
            return True
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False
    
    def load_state(self):
        """Load the trading state from file if it exists."""
        if not os.path.exists(state_file):
            logger.info("No saved state found")
            return False
        
        try:
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
                
            self.current_position = state.get('current_position', 0)
            self.expected_position = state.get('expected_position', 0)
            self.active_orders = state.get('active_orders', {})
            self.bar_history = state.get('bar_history', [])
            
            logger.info(f"Trading state loaded successfully: position={self.current_position}")
            return True
        except Exception as e:
            logger.error(f"Error loading trading state: {e}")
            return False
    
    def verify_position(self, force=False):
        """Verify that the internal position matches the actual position at IB."""
        now = datetime.now()
        
        # Only check positions periodically unless forced
        if not force and (now - self.last_position_check).total_seconds() < self.position_check_interval:
            return
            
        self.last_position_check = now
        
        try:
            # Get current positions from IB
            positions = self.ib.positions()
            
            # Get detailed debug info about active contract and positions
            logger.info(f"Active contract: {self.contract}")
            logger.info(f"Active contract details: Symbol={self.contract.symbol}, Exchange={self.contract.exchange}, SecType={self.contract.secType}")
            
            # More detailed logging of all positions
            position_details = []
            for pos in positions:
                contract_info = {
                    'symbol': pos.contract.symbol,
                    'exchange': pos.contract.exchange,
                    'secType': pos.contract.secType,
                    'position': pos.position,
                    'avgCost': pos.avgCost
                }
                position_details.append(contract_info)
                
            logger.info(f"All IB positions: {position_details}")
            
            # Find our contract in the positions with more flexible matching
            actual_position = 0
            for position in positions:
                # More detailed logging for each position being checked
                logger.info(f"Checking position: Symbol={position.contract.symbol}, Exchange={position.contract.exchange}, SecType={position.contract.secType}")
                
                # Try more flexible matching
                if (position.contract.symbol == self.contract.symbol and 
                    position.contract.secType == self.contract.secType):
                    logger.info(f"Found matching position: {position.position} contracts")
                    actual_position = position.position
                    break
            
            # If we have an actual position but our internal state is wrong, update it
            logger.info(f"Position verification: Internal={self.current_position}, Actual Quantity={actual_position}")
            
            # Update the directional flag based on the actual position
            new_position_flag = 1 if actual_position > 0 else (-1 if actual_position < 0 else 0)
            
            # Check if our internal state matches the actual position
            if self.current_position != new_position_flag or self.expected_position != actual_position:
                logger.warning(f"Position mismatch: Internal={self.current_position} (expected quantity: {self.expected_position}), Actual={actual_position}")
                
                # Reconcile by updating our internal state
                self.current_position = new_position_flag
                self.expected_position = actual_position
                
                logger.info(f"Reconciling position: updated to direction={self.current_position}, quantity={actual_position}")
                
                # Save the reconciled state
                self.save_state()
            else:
                logger.info(f"Position verification successful: Internal={self.current_position}, Actual={actual_position}")
                
        except Exception as e:
            logger.error(f"Error verifying position: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def verify_order_execution(self, order_id):
        """Verify that an order has been executed."""
        if order_id not in self.order_status:
            logger.warning(f"Order ID {order_id} not found in order status tracking")
            return False
            
        status = self.order_status[order_id]
        logger.info(f"Order {order_id} status: {status}")
        
        # Check if the order is filled
        if status == "Filled":
            logger.info(f"Order {order_id} has been successfully executed")
            return True
        elif status in ["Cancelled", "Error"]:
            logger.warning(f"Order {order_id} failed with status: {status}")
            return False
        else:
            # Order is still pending
            logger.info(f"Order {order_id} is still pending with status: {status}")
            return None
    
    def update_order_status(self, trade):
        """Update the status of an order based on trade updates."""
        order_id = trade.order.orderId
        status = trade.orderStatus.status
        
        # Track the order status change
        if order_id not in self.order_status:
            self.order_status[order_id] = status
            logger.info(f"New order status: ID={order_id}, Status={status}")
        elif self.order_status[order_id] != status:
            prev_status = self.order_status[order_id]
            self.order_status[order_id] = status
            logger.info(f"Order status updated: ID={order_id}, Status changed from {prev_status} to {status}")
            
            # If the order is filled, update our position
            if status == "Filled":
                # Get the order details
                order_info = self.active_orders.get(order_id)
                logger.info(f"Order filled: ID={order_id}, Info={order_info}")
                
                if order_info:
                    order_type = order_info.get('type')
                    logger.info(f"Filled order type: {order_type}")
                    
                    # Handle take profit or stop loss fills
                    if order_type in ['take_profit', 'stop_loss']:
                        parent_id = order_info.get('parent_id')
                        logger.info(f"Child order filled - parent ID: {parent_id}")
                        
                        if parent_id and parent_id in self.active_orders:
                            # Get all child orders
                            children = self.active_orders[parent_id].get('children', [])
                            
                            # Cancel any other child orders
                            for child_id in children:
                                if child_id != order_id and child_id in self.order_status and self.order_status[child_id] not in ["Filled", "Cancelled", "Inactive"]:
                                    logger.info(f"Cancelling other child order: {child_id}")
                                    try:
                                        self.ib.cancelOrder(self.ib.order(child_id))
                                    except Exception as e:
                                        logger.error(f"Error cancelling child order {child_id}: {e}")
                
                # Force immediate position verification to get actual state from IB
                logger.info("Order filled - verifying actual position with IB")
                self.verify_position(force=True)
                
                # Save state after position change
                self.save_state()
                
                # Now that position is verified, log the current state
                logger.info(f"After order fill: Position now {self.current_position} (quantity: {self.expected_position})")
            
            # Log when order is cancelled or failed
            elif status in ["Cancelled", "Inactive", "Error"]:
                logger.warning(f"Order {order_id} failed with status: {status}")
    
    def preprocess_bar(self, bar: dict) -> pd.DataFrame:
        """
        Update internal 5‑minute bar history, (re)calculate all enabled indicators
        for the full window, normalise, and return the **latest** normalised row.

        Steps replicate the training pipeline:
            1. Append bar to history
            2. Re‑calculate technical indicators (process_technical_indicators)
            3. Add day‑of‑week + minutes‑since‑open features
            4. Normalise using saved scaler
            5. Build ordered row: [close_norm] + indicators + position
        """

        # 1️⃣ Maintain sliding window
        self.bar_history.append(bar)
        max_history = max(500, self.min_bars_needed)
        if len(self.bar_history) > max_history:
            self.bar_history = self.bar_history[-max_history:]

        hist_df = pd.DataFrame(self.bar_history).set_index('time')
        hist_df[['open', 'high', 'low', 'close', 'volume']] = (
            hist_df[['open', 'high', 'low', 'close', 'volume']]
            .apply(pd.to_numeric, errors='coerce')
        )

        # Debug: Log the shape of bar history dataframe
        logger.info(f"DEBUG: Bar history dataframe shape: {hist_df.shape}")

        # 2️⃣ Indicators
        processed_df = process_technical_indicators(hist_df)
        if processed_df is None or processed_df.empty:
            logger.error("Indicator calculation failed – skipping bar")
            return None
            
        # Debug: Log columns after processing indicators
        logger.info(f"DEBUG: Columns after indicator calculation: {processed_df.columns.tolist()}")

        # 3️⃣ Time‑based features
        processed_df = self._add_time_features(processed_df)
        
        # Debug: Log columns after adding time features
        logger.info(f"DEBUG: Columns after adding time features: {processed_df.columns.tolist()}")

        # 4️⃣ Normalise
        try:
            norm_df = self._normalize_bar(processed_df)
            if norm_df is None:
                logger.error("Normalization returned None – skipping bar")
                return None
            
            if hasattr(norm_df, 'empty') and norm_df.empty:
                logger.error("Normalization returned empty DataFrame – skipping bar")
                return None
        except Exception as e:
            logger.error(f"Exception during normalization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
        # Debug: Log columns after normalization
        logger.info(f"DEBUG: Columns after normalization: {norm_df.columns.tolist()}")

        # 5️⃣ Position flag
        pos_flag = 0 if self.current_position == 0 else (1 if self.current_position > 0 else -1)
        norm_df['position'] = pos_flag

        # Ensure completeness & order
        latest = norm_df.iloc[[-1]].copy()
        
        # Debug: Log required features before ensuring completeness
        logger.info(f"DEBUG: Required features: {self.required_features}")
        
        # Check if time-based features are in enabled_indicators but not calculated
        has_dow_indicators = 'DOW_SIN' in self.enabled_indicators or 'DOW_COS' in self.enabled_indicators
        has_mso_indicators = 'MSO_SIN' in self.enabled_indicators or 'MSO_COS' in self.enabled_indicators
        
        # If DOW indicators are in enabled_indicators, ensure they're in the latest dataframe
        if has_dow_indicators:
            if 'DOW_SIN' not in latest.columns:
                logger.warning("DOW_SIN required but not calculated, adding placeholder")
                latest['DOW_SIN'] = 0.0
            if 'DOW_COS' not in latest.columns:
                logger.warning("DOW_COS required but not calculated, adding placeholder")
                latest['DOW_COS'] = 0.0
                
        # If MSO indicators are in enabled_indicators, ensure they're in the latest dataframe
        if has_mso_indicators:
            if 'MSO_SIN' not in latest.columns:
                logger.warning("MSO_SIN required but not calculated, adding placeholder")
                latest['MSO_SIN'] = 0.0
            if 'MSO_COS' not in latest.columns:
                logger.warning("MSO_COS required but not calculated, adding placeholder")
                latest['MSO_COS'] = 0.0
        
        # Ensure all required features are present
        for col in self.required_features:
            if col not in latest.columns:
                latest[col] = 0.0
                logger.info(f"DEBUG: Added missing column: {col}")

        # Map VOLUME_NORM to VOLUME_MA if needed (for backward compatibility)
        if 'VOLUME_MA' in self.enabled_indicators and 'VOLUME_MA' not in latest.columns and 'VOLUME_NORM' in latest.columns:
            logger.info("Mapping VOLUME_NORM to VOLUME_MA for compatibility")
            latest['VOLUME_MA'] = latest['VOLUME_NORM']
        
        ordered_cols = ['close_norm'] + self.enabled_indicators + ['position']
        
        # Debug: Log the ordered columns
        logger.info(f"DEBUG: Ordered columns for prediction: {ordered_cols}")
        logger.info(f"DEBUG: Number of columns in ordered_cols: {len(ordered_cols)}")
        
        # Verify all ordered columns exist in the dataframe
        missing_cols = [col for col in ordered_cols if col not in latest.columns]
        if missing_cols:
            logger.error(f"Missing columns in normalized data: {missing_cols}")
            for col in missing_cols:
                logger.info(f"Adding missing column {col} with zero values")
                latest[col] = 0.0
        
        latest = latest[ordered_cols]
        
        # Debug: Log the final dataframe shape
        logger.info(f"DEBUG: Final prediction dataframe shape: {latest.shape}")

        return latest

    def _load_normalization_scalers(self):
        """
        Load the normalization scalers from files to use for standardizing data.
        This ensures we use the same normalization as during training.
        """
        try:
            # Try to load from the model directory first
            model_dir = os.path.dirname(self.model_path)
            self.scaler = load_scaler(os.path.join(model_dir, "scaler.pkl"))
            
            # If not found in model dir, try in current directory
            if self.scaler is None:
                self.scaler = load_scaler("scaler.pkl")
                
            if self.scaler is None:
                logger.warning("No scaler found. A new scaler will be created during the first normalization.")
                # Not setting self.scaler here, let _normalize_bar handle it
            else:
                logger.info(f"Normalization scaler loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading normalization scalers: {e}")
            # Set to None so _normalize_bar will create a new scaler
            self.scaler = None
    
    def execute_trade(self, prediction, bar):
        """
        Execute a trade based on the prediction from the model.
        
        Args:
            prediction: The action predicted by the model (0=long/buy, 1=short/sell, 2=hold)
            bar: The current price bar
            
        Returns:
            bool: Whether a trade was executed
        """
        try:
            if prediction is None:
                logger.error("No prediction available for trade execution")
                return False
                
            # Convert prediction to action
            # In environment.py: 0=long/buy, 1=short/sell, 2=hold
            action = prediction  
            logger.info(f"Trade decision based on prediction: {action}")
            
            # Get current position from IB to verify internal state is correct
            self.verify_position()
            
            # Get price information from the bar
            current_price = bar['close']
            logger.info(f"Current price: {current_price}")
            
            # Default position size from config or risk management
            position_size = self.position_size
            
            # Generate a unique order ID to avoid collisions
            execution_id = int(time.time() * 1000) + self.execution_counter
            self.execution_counter += 1
            
            # Process the action (0=long/buy, 1=short/sell, 2=hold)
            if action == 0:  # Buy/Long signal
                if self.current_position <= 0:  # If not already long
                    # First cancel ALL existing orders (including take profit/stop loss)
                    self._cancel_all_existing_orders()
                    
                    # Wait a brief moment for order cancellations to be processed
                    time.sleep(0.5)
                    
                    # Close any existing short position first
                    if self.current_position < 0:
                        logger.info(f"Closing existing short position before going long")
                        # Create a market order to close the short position
                        close_order = MarketOrder('BUY', abs(self.expected_position))
                        close_order.transmit = True  # Set transmit flag to true
                        trade = self.ib.placeOrder(self.contract, close_order)
                        self.order_ids.append(trade.order.orderId)
                        logger.info(f"Placed order to close short position: {trade.order}")
                        
                        # Wait for the position to be closed
                        self._wait_for_position_change()
                    
                    # Now open a new long position
                    logger.info(f"Opening long position of {position_size} contracts")
                    
                    # If risk management is enabled, use bracket orders
                    if self.use_risk_management and (self.stop_loss_pct or self.take_profit_pct):
                        # Calculate stop loss and take profit prices based on portfolio percentage
                        # For futures, we need to convert portfolio percentage to price points
                        
                        # Point value for NQ futures ($20 per point)
                        point_value = 20.0
                        
                        # Calculate stop loss and take profit prices
                        stop_loss_price = None
                        take_profit_price = None
                        
                        # Use the initial_balance from config.yaml instead of getting from IB
                        portfolio_value = config["environment"]["initial_balance"]
                        logger.info(f"Using configured portfolio value for risk: ${portfolio_value}")
                        
                        if self.stop_loss_pct:
                            # Calculate dollar risk based on portfolio percentage
                            risk_dollars = portfolio_value * (self.stop_loss_pct / 100)
                            
                            # Convert to points (how many points can we risk)
                            risk_points = risk_dollars / (point_value * position_size)
                            
                            # Calculate stop loss price for long position (entry - risk points)
                            stop_loss_price = round(current_price - risk_points, 2)
                            logger.info(f"Stop loss: ${risk_dollars:.2f} ({self.stop_loss_pct}% of portfolio), {risk_points:.2f} points, price: {stop_loss_price}")
                            
                        if self.take_profit_pct:
                            # Calculate dollar profit target based on portfolio percentage
                            profit_dollars = portfolio_value * (self.take_profit_pct / 100)
                            
                            # Convert to points (how many points for target)
                            profit_points = profit_dollars / (point_value * position_size)
                            
                            # Calculate take profit price for long position (entry + profit points)
                            take_profit_price = round(current_price + profit_points, 2)
                            logger.info(f"Take profit: ${profit_dollars:.2f} ({self.take_profit_pct}% of portfolio), {profit_points:.2f} points, price: {take_profit_price}")
                        
                        # Create parent market order
                        parent = Order()
                        parent.orderId = self.ib.client.getReqId()
                        parent.action = 'BUY'
                        parent.orderType = 'MKT'
                        parent.totalQuantity = position_size
                        parent.transmit = False  # Parent will not transmit until children are attached
                        
                        # Store parent order ID for reference
                        parent_id = parent.orderId
                        self.active_orders[parent_id] = {
                            'type': 'parent',
                            'direction': 'long',
                            'children': []
                        }
                        
                        # Create child orders list
                        child_orders = []
                        
                        # Take profit order (limit order)
                        if take_profit_price:
                            take_profit = Order()
                            take_profit.orderId = self.ib.client.getReqId()
                            take_profit.action = 'SELL'
                            take_profit.orderType = 'LMT'
                            take_profit.totalQuantity = position_size
                            take_profit.lmtPrice = take_profit_price
                            take_profit.parentId = parent_id
                            take_profit.transmit = False  # Don't transmit yet
                            
                            # Add to active orders
                            tp_id = take_profit.orderId
                            self.active_orders[tp_id] = {
                                'type': 'take_profit',
                                'parent_id': parent_id,
                                'direction': 'long'
                            }
                            self.active_orders[parent_id]['children'].append(tp_id)
                            
                            child_orders.append(take_profit)
                        
                        # Stop loss order (stop order)
                        if stop_loss_price:
                            stop_loss = Order()
                            stop_loss.orderId = self.ib.client.getReqId()
                            stop_loss.action = 'SELL'
                            stop_loss.orderType = 'STP'
                            stop_loss.totalQuantity = position_size
                            stop_loss.auxPrice = stop_loss_price
                            stop_loss.parentId = parent_id
                            # This is the last order, so it will transmit the entire bracket
                            stop_loss.transmit = True
                            
                            # Add to active orders
                            sl_id = stop_loss.orderId
                            self.active_orders[sl_id] = {
                                'type': 'stop_loss',
                                'parent_id': parent_id,
                                'direction': 'long'
                            }
                            self.active_orders[parent_id]['children'].append(sl_id)
                            
                            child_orders.append(stop_loss)
                        else:
                            # If no stop loss, the last (or only) take profit order must transmit
                            if child_orders:
                                child_orders[-1].transmit = True
                            else:
                                # If no child orders, parent must transmit
                                parent.transmit = True
                        
                        # Place parent order first
                        parent_trade = self.ib.placeOrder(self.contract, parent)
                        self.order_ids.append(parent_id)
                        logger.info(f"Placed parent order for long position: {parent_trade.order}")
                        
                        # Place child orders
                        for child in child_orders:
                            child_trade = self.ib.placeOrder(self.contract, child)
                            self.order_ids.append(child.orderId)
                            logger.info(f"Placed child order: {child_trade.order}")
                    else:
                        # Simple market order if no risk management
                        order = MarketOrder('BUY', position_size)
                        order.transmit = True  # Set transmit flag to true
                        trade = self.ib.placeOrder(self.contract, order)
                        self.order_ids.append(trade.order.orderId)
                        logger.info(f"Placed market order to go long: {order}")
                    
                    # Update internal state
                    self.current_position = 1
                    self.expected_position = position_size
                    return True
                else:
                    logger.info("Already in long position, no trade executed")
                    return False
                    
            elif action == 1:  # Sell/Short signal
                if self.current_position >= 0:  # If not already short
                    # First cancel ALL existing orders (including take profit/stop loss)
                    self._cancel_all_existing_orders()
                    
                    # Wait a brief moment for order cancellations to be processed
                    time.sleep(0.5)
                    
                    # Close any existing long position first
                    if self.current_position > 0:
                        logger.info(f"Closing existing long position before going short")
                        # Create a market order to close the long position
                        close_order = MarketOrder('SELL', abs(self.expected_position))
                        close_order.transmit = True  # Set transmit flag to true
                        trade = self.ib.placeOrder(self.contract, close_order)
                        self.order_ids.append(trade.order.orderId)
                        logger.info(f"Placed order to close long position: {trade.order}")
                        
                        # Wait for the position to be closed
                        self._wait_for_position_change()
                    
                    # Now open a new short position
                    logger.info(f"Opening short position of {position_size} contracts")
                    
                    # If risk management is enabled, use bracket orders
                    if self.use_risk_management and (self.stop_loss_pct or self.take_profit_pct):
                        # Calculate stop loss and take profit prices based on portfolio percentage
                        # For futures, we need to convert portfolio percentage to price points
                        
                        # Point value for NQ futures ($20 per point)
                        point_value = 20.0
                        
                        # Calculate stop loss and take profit prices
                        stop_loss_price = None
                        take_profit_price = None
                        
                        # Use the initial_balance from config.yaml instead of getting from IB
                        portfolio_value = config["environment"]["initial_balance"]
                        logger.info(f"Using configured portfolio value for risk: ${portfolio_value}")
                        
                        if self.stop_loss_pct:
                            # Calculate dollar risk based on portfolio percentage
                            risk_dollars = portfolio_value * (self.stop_loss_pct / 100)
                            
                            # Convert to points (how many points can we risk)
                            risk_points = risk_dollars / (point_value * position_size)
                            
                            # Calculate stop loss price for short position (entry + risk points)
                            stop_loss_price = round(current_price + risk_points, 2)
                            logger.info(f"Stop loss: ${risk_dollars:.2f} ({self.stop_loss_pct}% of portfolio), {risk_points:.2f} points, price: {stop_loss_price}")
                            
                        if self.take_profit_pct:
                            # Calculate dollar profit target based on portfolio percentage
                            profit_dollars = portfolio_value * (self.take_profit_pct / 100)
                            
                            # Convert to points (how many points for target)
                            profit_points = profit_dollars / (point_value * position_size)
                            
                            # Calculate take profit price for short position (entry - profit points)
                            take_profit_price = round(current_price - profit_points, 2)
                            logger.info(f"Take profit: ${profit_dollars:.2f} ({self.take_profit_pct}% of portfolio), {profit_points:.2f} points, price: {take_profit_price}")
                        
                        # Create parent market order
                        parent = Order()
                        parent.orderId = self.ib.client.getReqId()
                        parent.action = 'SELL'
                        parent.orderType = 'MKT'
                        parent.totalQuantity = position_size
                        parent.transmit = False  # Parent will not transmit until children are attached
                        
                        # Store parent order ID for reference
                        parent_id = parent.orderId
                        self.active_orders[parent_id] = {
                            'type': 'parent',
                            'direction': 'short',
                            'children': []
                        }
                        
                        # Create child orders list
                        child_orders = []
                        
                        # Take profit order (limit order)
                        if take_profit_price:
                            take_profit = Order()
                            take_profit.orderId = self.ib.client.getReqId()
                            take_profit.action = 'BUY'
                            take_profit.orderType = 'LMT'
                            take_profit.totalQuantity = position_size
                            take_profit.lmtPrice = take_profit_price
                            take_profit.parentId = parent_id
                            take_profit.transmit = False  # Don't transmit yet
                            
                            # Add to active orders
                            tp_id = take_profit.orderId
                            self.active_orders[tp_id] = {
                                'type': 'take_profit',
                                'parent_id': parent_id,
                                'direction': 'short'
                            }
                            self.active_orders[parent_id]['children'].append(tp_id)
                            
                            child_orders.append(take_profit)
                        
                        # Stop loss order (stop order)
                        if stop_loss_price:
                            stop_loss = Order()
                            stop_loss.orderId = self.ib.client.getReqId()
                            stop_loss.action = 'BUY'
                            stop_loss.orderType = 'STP'
                            stop_loss.totalQuantity = position_size
                            stop_loss.auxPrice = stop_loss_price
                            stop_loss.parentId = parent_id
                            # This is the last order, so it will transmit the entire bracket
                            stop_loss.transmit = True
                            
                            # Add to active orders
                            sl_id = stop_loss.orderId
                            self.active_orders[sl_id] = {
                                'type': 'stop_loss',
                                'parent_id': parent_id,
                                'direction': 'short'
                            }
                            self.active_orders[parent_id]['children'].append(sl_id)
                            
                            child_orders.append(stop_loss)
                        else:
                            # If no stop loss, the last (or only) take profit order must transmit
                            if child_orders:
                                child_orders[-1].transmit = True
                            else:
                                # If no child orders, parent must transmit
                                parent.transmit = True
                        
                        # Place parent order first
                        parent_trade = self.ib.placeOrder(self.contract, parent)
                        self.order_ids.append(parent_id)
                        logger.info(f"Placed parent order for short position: {parent_trade.order}")
                        
                        # Place child orders
                        for child in child_orders:
                            child_trade = self.ib.placeOrder(self.contract, child)
                            self.order_ids.append(child.orderId)
                            logger.info(f"Placed child order: {child_trade.order}")
                    else:
                        # Simple market order if no risk management
                        order = MarketOrder('SELL', position_size)
                        order.transmit = True  # Set transmit flag to true
                        trade = self.ib.placeOrder(self.contract, order)
                        self.order_ids.append(trade.order.orderId)
                        logger.info(f"Placed market order to go short: {order}")
                    
                    # Update internal state
                    self.current_position = -1
                    self.expected_position = -position_size
                    return True
                else:
                    logger.info("Already in short position, no trade executed")
                    return False
                    
            elif action == 2:  # Hold signal
                logger.info(f"Hold signal received: {action}, no trade executed")
                return False
                
            else:  # Invalid action
                logger.info(f"Invalid action: {action}, no trade executed")
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def _cancel_all_existing_orders(self):
        """
        Cancel ALL existing orders for the current contract.
        This is more robust than trying to track individual orders.
        """
        logger.info("Cancelling ALL existing orders to ensure clean slate")
        
        # Get all open trades from IB (trades contain both order and contract)
        open_trades = self.ib.trades()
        if not open_trades:
            logger.info("No open orders found to cancel")
            return
            
        # Count orders for our contract
        contract_orders = 0
        
        # Cancel all orders for our contract
        for trade in open_trades:
            # Check if this order is for our contract
            if trade.contract.symbol == self.contract.symbol and trade.contract.secType == self.contract.secType:
                contract_orders += 1
                try:
                    logger.info(f"Cancelling order: ID={trade.order.orderId}, Action={trade.order.action}, Type={trade.order.orderType}")
                    self.ib.cancelOrder(trade.order)
                except Exception as e:
                    logger.error(f"Error cancelling order {trade.order.orderId}: {e}")
        
        logger.info(f"Cancelled {contract_orders} orders for {self.contract.symbol}")
        
        # Clear our internal order tracking
        self.order_ids = []
        self.active_orders = {}
    
    def _wait_for_position_change(self, max_wait_seconds=3):
        """
        Wait for position to change after closing a position.
        This helps ensure we don't move on to placing new orders before the close is processed.
        
        Args:
            max_wait_seconds: Maximum time to wait in seconds
        """
        logger.info(f"Waiting for position change confirmation (max {max_wait_seconds} seconds)")
        start_time = time.time()
        
        # Store initial position state
        initial_position = self.current_position
        
        while time.time() - start_time < max_wait_seconds:
            # Force position verification
            self.verify_position(force=True)
            
            # If position has changed to neutral (0), we can proceed
            if initial_position != 0 and self.current_position == 0:
                logger.info(f"Position successfully changed from {initial_position} to {self.current_position}")
                return
                
            # Wait a bit before checking again
            time.sleep(0.2)
            
        logger.warning(f"Timed out waiting for position to change. Current position: {self.current_position}")
        
        # One final position verification to make sure we have latest state
        self.verify_position(force=True)
    
    def fetch_historical_bars(self, days_back=30):
        """
        Fetch historical bars for the active contract to initialize bar history.
        
        Args:
            days_back: Number of days to look back for historical data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.contract:
                logger.error("No active contract set for fetching historical data")
                return False
                
            # Calculate end time (now) and start time (days_back days ago)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            logger.info(f"Fetching historical data from {start_time} to {end_time}")
            
            # Request historical data from IB
            bars = self.ib.reqHistoricalData(
                self.contract,
                endDateTime=end_time.strftime('%Y%m%d %H:%M:%S'),
                durationStr=f'{days_back} D',
                barSizeSetting='5 mins',
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1
            )
            
            if not bars or len(bars) == 0:
                logger.error("No historical bars received")
                return False
                
            logger.info(f"Received {len(bars)} historical bars")
            
            # Convert bars to dictionary format and add to bar history
            for bar in bars:
                bar_dict = {
                    'time': bar.date,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                }
                self.bar_history.append(bar_dict)
                
            logger.info(f"Added {len(bars)} historical bars to bar history")
            return True
            
        except Exception as e:
            logger.error(f"Error fetching historical bars: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _normalize_bar(self, df):
        """
        Normalize the dataframe using the loaded scalers.
        """
        try:
            if not df.empty:
                # Log sample of pre-normalized values for key features
                last_row = df.iloc[-1]
                logger.info(f"DEBUG: PRE-NORMALIZATION values for latest bar:")
                for col in ['close', 'high', 'low', 'open', 'volume']:
                    if col in last_row:
                        logger.info(f"    {col}: {last_row[col]}")
                
                # Get key technical indicators if they exist
                for col in ['RSI', 'MACD', 'ATR', 'SMA_20', 'EMA_20']:
                    if col in last_row:
                        logger.info(f"    {col}: {last_row[col]}")
                
                # Get the columns to normalize (exclude certain columns like time, position, etc.)
                cols_to_scale = get_standardized_column_names(df)
                logger.info(f"Normalizing {len(cols_to_scale)} columns: {cols_to_scale}")

                # Check if self.scaler is a scikit-learn MinMaxScaler
                from sklearn.preprocessing import MinMaxScaler
                if not isinstance(self.scaler, MinMaxScaler):
                    logger.warning("Scaler is not a MinMaxScaler. Creating a new scaler.")
                    scaler_to_use = None  # Will create a new scaler in normalize_data
                else:
                    scaler_to_use = self.scaler

                # Use the normalize_data function from the normalization module with proper parameters
                norm_result = normalize_data(
                    data=df,
                    cols_to_scale=cols_to_scale,
                    feature_range=self.feature_range,
                    scaler=scaler_to_use,
                    use_sigmoid=True,  # Enable sigmoid normalization
                    sigmoid_k=2.0  # Default steepness parameter
                )
                
                # Handle case where normalize_data returns a tuple instead of a DataFrame
                if isinstance(norm_result, tuple):
                    logger.info(f"normalize_data returned a tuple of length {len(norm_result)}")
                    # Assuming the first element of the tuple is the normalized DataFrame
                    norm_df = norm_result[0] if len(norm_result) > 0 else None
                    
                    # If we created a new scaler, save it for future use
                    if scaler_to_use is None and len(norm_result) > 1:
                        self.scaler = norm_result[1]
                        logger.info("Created and saved new scaler for future normalization")
                else:
                    norm_df = norm_result
                
                # Ensure categorical indicators like supertrend and PSAR_DIR maintain their values
                if norm_df is not None and not norm_df.empty:
                    # Fix supertrend values (should be -1 or 1, not normalized)
                    if 'supertrend' in df.columns:
                        norm_df['supertrend'] = df['supertrend']
                        logger.info(f"Restored supertrend original values (should be -1 or 1)")
                    if 'SUPERTREND' in df.columns:
                        norm_df['SUPERTREND'] = df['SUPERTREND']
                        logger.info(f"Restored SUPERTREND original values (should be -1 or 1)")
                    
                    # Fix PSAR_DIR values (also a directional indicator that should be -1 or 1)
                    if 'PSAR_DIR' in df.columns:
                        norm_df['PSAR_DIR'] = df['PSAR_DIR']
                        logger.info(f"Restored PSAR_DIR original values (should be -1 or 1)")
                    
                    # Handle SMA_NORM based on available columns (matching normalization.py logic)
                    sigmoid_k = 2.0
                    
                    # SMA_NORM from either SMA_20 or SMA
                    if 'SMA_20' in df.columns and 'SMA_NORM' not in norm_df.columns:
                        # Same logic as in normalization.py
                        sma_mean = df['SMA_20'].mean()
                        sma_std = max(df['SMA_20'].std(), 1e-6)
                        norm_df['SMA_NORM'] = 2 / (1 + np.exp(-sigmoid_k * ((df['SMA_20'] - sma_mean) / sma_std))) - 1
                        logger.info(f"Created SMA_NORM from SMA_20 using sigmoid normalization")
                    elif 'SMA' in df.columns and 'SMA_NORM' not in norm_df.columns:
                        # Same logic as before
                        close_mean = df['close'].mean()
                        close_std = max(df['close'].std(), 1e-6)
                        norm_df['SMA_NORM'] = 2 / (1 + np.exp(-sigmoid_k * ((df['SMA'] - close_mean) / close_std))) - 1
                        logger.info(f"Created SMA_NORM from SMA using sigmoid normalization")
                    
                    # EMA_NORM from either EMA_20 or EMA
                    if 'EMA_20' in df.columns and 'EMA_NORM' not in norm_df.columns:
                        # Same logic as in normalization.py
                        ema_mean = df['EMA_20'].mean()
                        ema_std = max(df['EMA_20'].std(), 1e-6)
                        norm_df['EMA_NORM'] = 2 / (1 + np.exp(-sigmoid_k * ((df['EMA_20'] - ema_mean) / ema_std))) - 1
                        logger.info(f"Created EMA_NORM from EMA_20 using sigmoid normalization")
                    elif 'EMA' in df.columns and 'EMA_NORM' not in norm_df.columns:
                        # Same logic as before
                        close_mean = df['close'].mean()
                        close_std = max(df['close'].std(), 1e-6)
                        norm_df['EMA_NORM'] = 2 / (1 + np.exp(-sigmoid_k * ((df['EMA'] - close_mean) / close_std))) - 1
                        logger.info(f"Created EMA_NORM from EMA using sigmoid normalization")
                    
                    # VOLUME_MA from either VOLUME or VOLUME_NORM
                    if 'VOLUME' in df.columns and 'VOLUME_MA' not in norm_df.columns:
                        # Same logic as in normalization.py
                        vol_mean = df['VOLUME'].mean()
                        vol_std = max(df['VOLUME'].std(), 1e-6)
                        norm_df['VOLUME_MA'] = 2 / (1 + np.exp(-sigmoid_k * ((df['VOLUME'] - vol_mean) / vol_std))) - 1
                        logger.info(f"Created VOLUME_MA from VOLUME using sigmoid normalization")
                    elif 'VOLUME_NORM' in df.columns and 'VOLUME_MA' not in norm_df.columns:
                        # Previous logic - directly copy VOLUME_NORM to VOLUME_MA
                        norm_df['VOLUME_MA'] = norm_df['VOLUME_NORM']
                        logger.info(f"Set VOLUME_MA to match VOLUME_NORM value")
                
                # Log sample of normalized values for key features
                if norm_df is not None and not norm_df.empty:
                    last_norm_row = norm_df.iloc[-1]
                    logger.info(f"DEBUG: POST-NORMALIZATION values for latest bar:")
                    
                    # Check for normalized price features
                    for col in ['close_norm', 'high_norm', 'low_norm', 'open_norm']:
                        if col in last_norm_row:
                            logger.info(f"    {col}: {last_norm_row[col]}")
                    
                    # Check for normalized technical indicators
                    for col in ['RSI', 'MACD', 'ATR', 'SMA_NORM', 'EMA_NORM']:
                        if col in last_norm_row:
                            logger.info(f"    {col}: {last_norm_row[col]}")
                    
                    # Specifically log the supertrend and PSAR_DIR values to confirm they're correct
                    for col in ['supertrend', 'SUPERTREND', 'PSAR_DIR']:
                        if col in last_norm_row:
                            logger.info(f"    {col}: {last_norm_row[col]}")
                
                return norm_df
            return None
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _add_time_features(self, df):
        """
        Add time-based features (day of week, minutes since open) to the dataframe.
        """
        try:
            if df.empty:
                return df
                
            # Copy to avoid modifying the original
            result_df = df.copy()
            
            # Add day of week (DOW) cyclical features if enabled in config
            if config.get("indicators", {}).get("day_of_week", {}).get("enabled", True):
                result_df['DOW_SIN'] = df.index.dayofweek.map(lambda x: np.sin(2 * np.pi * x / 7))
                result_df['DOW_COS'] = df.index.dayofweek.map(lambda x: np.cos(2 * np.pi * x / 7))
                logger.info("Added DOW_SIN and DOW_COS features based on config")
            
            # Add minutes since market open (MSO) if enabled
            if config.get("indicators", {}).get("minutes_since_open", {}).get("enabled", False):
                # Calculate minutes since market open (9:30am Eastern)
                def minutes_since_open(dt):
                    # Convert to Eastern time
                    eastern = pytz.timezone('US/Eastern')
                    dt_eastern = dt.tz_localize(pytz.UTC).tz_convert(eastern) if dt.tzinfo else eastern.localize(dt)
                    
                    # Market opens at 9:30 AM ET
                    market_open = dt_eastern.replace(hour=9, minute=30, second=0, microsecond=0)
                    
                    # If before market open, return 0
                    if dt_eastern < market_open:
                        return 0
                        
                    # Calculate minutes since open
                    delta = dt_eastern - market_open
                    minutes = delta.total_seconds() / 60
                    
                    # Normalize to a daily cycle (0-390, market open for 6.5 hours = 390 minutes)
                    return min(minutes, 390)
                
                # Apply to each timestamp
                mso_values = [minutes_since_open(dt) for dt in df.index]
                
                # Create cyclical features (sin/cos) for better representation
                result_df['MSO_SIN'] = [np.sin(2 * np.pi * m / 390) for m in mso_values]
                result_df['MSO_COS'] = [np.cos(2 * np.pi * m / 390) for m in mso_values]
                logger.info("Added MSO_SIN and MSO_COS features based on config")
            
            return result_df
        except Exception as e:
            logger.error(f"Error adding time features: {e}")
            return df

    def get_prediction(self, normalized_df):
        """
        Get a prediction from the model based on the normalized dataframe.
        """
        try:
            if normalized_df is None or normalized_df.empty:
                logger.error("Invalid input for prediction")
                return None
            
            # Ensure the input is in the correct format
            if 'close_norm' not in normalized_df.columns:
                logger.error("Missing 'close_norm' column in input dataframe")
                return None
            
            # Debug: Log input data shape and columns
            logger.info(f"DEBUG: Prediction input shape: {normalized_df.shape}")
            logger.info(f"DEBUG: Prediction input columns: {normalized_df.columns.tolist()}")
            
            # Debug: Compare with model's expected features
            expected_features = self.model.observation_space.shape[0]
            logger.info(f"DEBUG: Model expects {expected_features} features, input has {normalized_df.shape[1]} features")
            
            # Get the prediction from the model
            prediction_input = normalized_df.values.reshape(1, -1)
            
            # Debug: Log the input array shape
            logger.info(f"DEBUG: Reshaped prediction input shape: {prediction_input.shape}")
            
            # Debug: Log the full NORMALIZED observation vector
            logger.info(f"DEBUG: Full NORMALIZED observation vector (values should be mostly in [-1,1] range): {prediction_input[0].tolist()}")
            
            # Create a more human-readable version with column names
            readable_vector = {}
            columns = normalized_df.columns.tolist()
            values = prediction_input[0].tolist()
            for i, column in enumerate(columns):
                if i < len(values):
                    readable_vector[column] = values[i]
            
            logger.info(f"DEBUG: NORMALIZED observation vector with feature names: {readable_vector}")
            
            # Additional check to verify normalization worked correctly
            # Most values should be between -1 and 1 after normalization
            in_range_count = sum(1 for v in values if -1.1 <= v <= 1.1)  # Allow slight buffer outside [-1,1]
            pct_in_range = (in_range_count / len(values)) * 100 if values else 0
            logger.info(f"DEBUG: Normalization check: {in_range_count}/{len(values)} values ({pct_in_range:.1f}%) are within range [-1.1,1.1]")
            
            # Assert correctness of dimensions before calling the policy
            assert prediction_input.shape[1] == self.model.observation_space.shape[0], \
                f"vector:{prediction_input.shape[1]} obs:{self.model.observation_space.shape[0]}"
            
            logger.info(f"Normalized observation vector: {prediction_input}")
            prediction = self.model.predict(prediction_input)
            
            # Convert the prediction to a readable format
            readable_prediction = prediction[0]
            
            # Debug: Log the prediction
            logger.info(f"DEBUG: Raw prediction output: {prediction}")
            logger.info(f"DEBUG: Readable prediction: {readable_prediction}")
            
            return readable_prediction
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

def onBar(bars, hasNewBar):
    """
    Callback function for new bars from Interactive Brokers.
    This is called by IBApi when new bars arrive.
    
    Args:
        bars: RealTimeBarList object from IB API
        hasNewBar: Whether the update contains a new completed bar
    """
    if not hasNewBar:
        return
        
    global model_trader, last_execution_time, last_data_timestamp, is_data_flowing, bar_buckets
    
    # Update heartbeat monitoring
    last_data_timestamp = datetime.now()
    is_data_flowing = True
    
    try:
        # With RealTimeBarList, get the latest bar directly
        if len(bars) == 0:
            logger.warning("Received empty bar update")
            return
            
        # Get the latest bar
        latest_bar = bars[-1]
        
        # Ensure the bar time is timezone-aware (UTC)
        bar_time = latest_bar.time
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=pytz.UTC)
        else:
            bar_time = bar_time.astimezone(pytz.UTC)
        
        # Convert bar to dictionary format - using the actual field names from IB API
        bar_dict = {
            'timestamp': bar_time,
            'time': bar_time,  # Add time key for consistent processing
            'open': latest_bar.open_,  # Note the trailing underscore
            'high': latest_bar.high,
            'low': latest_bar.low,
            'close': latest_bar.close,
            'volume': latest_bar.volume
        }
        
        # Log the new bar
        logger.info(f"New 5-second bar: {bar_dict}")
        
        # Add the bar to the appropriate bucket based on the END of the 5-minute interval it belongs to
        bucket_key_dt = end_of_interval(bar_time)
        bucket_key = bucket_key_dt.strftime("%Y-%m-%d %H:%M")
        
        # Add this bar to the appropriate bucket
        bar_buckets[bucket_key].append(bar_dict)
        logger.info(f"Added bar to bucket {bucket_key}, now has {len(bar_buckets[bucket_key])}/{BARS_PER_FIVE_MIN} bars")
        
        # Debug logging to help diagnose timing issues
        logger.debug(
            f"bucket {bucket_key} size={len(bar_buckets[bucket_key])} last_bar={bar_time} now={datetime.now(UTC)}"
        )
        
        # Check if we have a complete 5-minute bar to process
        five_min_bar = synchronize_bars()
        
        if five_min_bar:
            # Process the 5-minute bar
            logger.info(f"Processing complete 5-minute bar: {five_min_bar}")
            
            # Check if enough time has passed since last execution
            now = datetime.now()
            if last_execution_time and now - last_execution_time < timedelta(seconds=MIN_EXECUTION_INTERVAL):
                time_since_last = (now - last_execution_time).total_seconds()
                logger.info(f"Skipping execution - only {time_since_last:.1f} seconds since last execution (min {MIN_EXECUTION_INTERVAL}s)")
                return
            
            # Process the bar through indicators and normalization
            normalized_df = model_trader.preprocess_bar(five_min_bar)
            
            if normalized_df is None:
                logger.error("Failed to preprocess 5-minute bar, skipping execution")
                return
            
            # Get prediction from model
            prediction = model_trader.get_prediction(normalized_df)
            
            if prediction is None:
                logger.error("Failed to get prediction, skipping execution")
                return
            
            # Execute trade based on prediction
            trade_executed = model_trader.execute_trade(prediction, five_min_bar)
            
            if trade_executed:
                # Update last execution time
                last_execution_time = now
                logger.info(f"Trade executed at {now} based on 5-minute bar prediction")
    except Exception as e:
        logger.error(f"Error in onBar: {e}")
        import traceback
        logger.error(traceback.format_exc())

def heartbeat_monitor():
    """
    Monitor data flow and reconnect if necessary.
    """
    global ib, model_trader, reconnection_attempts, contract
    
    logger.info("Starting heartbeat monitor thread")
    
    while True:
        try:
            current_time = datetime.now()
            
            # Check if data is still flowing
            if is_data_flowing:
                time_since_last = (current_time - last_data_timestamp).total_seconds()
                
                # If no data for over DATA_FLOW_THRESHOLD seconds, try to reconnect
                if time_since_last > DATA_FLOW_THRESHOLD:
                    logger.error(f"No data received for {time_since_last:.1f} seconds. Attempting reconnection.")
                    reconnect_to_ib()
                    
                    # If we still have the contract, re-subscribe to data
                    if 'contract' in globals() and contract:
                        bars = ib.reqRealTimeBars(contract, barSize=5, whatToShow='TRADES', useRTH=False)
                        logger.info(f"Re-subscribed to real-time bars for {contract.localSymbol}")
            
            # Sleep before next check
            time.sleep(15)
            
        except Exception as e:
            logger.error(f"Error in heartbeat monitor: {e}")
            time.sleep(15)

def reconnect_to_ib():
    """Attempt to reconnect to IB and restore the trading state."""
    global ib, model_trader, reconnection_attempts, active_contract
    
    reconnection_attempts += 1
    
    try:
        # Disconnect if currently connected
        if ib.isConnected():
            ib.disconnect()
        
        # Wait before reconnecting
        time.sleep(5)
        
        # Reconnect
        ib.connect('127.0.0.1', 7496, clientId=1)
        logger.info("Reconnected to IB")
        
        # Save the current state before reinitializing
        if model_trader:
            model_trader.save_state()
        
        # Restart our market data subscription
        if 'active_contract' in globals() and active_contract:
            bars = ib.reqRealTimeBars(active_contract, barSize=5, whatToShow='TRADES', useRTH=False)
            bars.updateEvent += onBar
            logger.info("Resubscribed to market data")
        
        # Restore the trading state
        if model_trader:
            model_trader.load_state()
            model_trader.verify_position()
        
        return True
    except Exception as e:
        logger.error(f"Reconnection attempt failed: {e}")
        return False

def get_most_recent_contract(contracts):
    """
    From a list of contract details for NQ futures, select the contract with the earliest expiration date
    that is still in the future (the active or front-month contract).
    """
    from datetime import datetime

    # Helper function to parse the expiration string.
    def parse_date(date_str):
        try:
            # Handle format 'YYYYMMDD' or sometimes 'YYYYMM'
            if len(date_str) == 8:
                return datetime.strptime(date_str, '%Y%m%d')
            elif len(date_str) == 6:
                return datetime.strptime(date_str, '%Y%m')
        except Exception as e:
            return datetime.max

    today = datetime.today()
    # Filter contracts that have an expiration date in the future.
    valid_contracts = [cd for cd in contracts if parse_date(cd.contract.lastTradeDateOrContractMonth) > today]
    if valid_contracts:
        # Sort filtered contracts by expiration date (earliest first)
        sorted_contracts = sorted(valid_contracts, key=lambda cd: parse_date(cd.contract.lastTradeDateOrContractMonth))
        return sorted_contracts[0].contract
    else:
        # Fallback to the first contract if for some reason none are in the future.
        return contracts[0].contract

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live trading script for executing model predictions via IB")
    parser.add_argument("--model", type=str, default="best_model", help="Path to the model directory")
    parser.add_argument("--contract", type=str, default="NQ", help="Contract symbol to trade")
    parser.add_argument("--paper", action="store_true", help="Connect to paper trading")
    parser.add_argument("--no_risk", action="store_true", help="Disable risk management")
    args = parser.parse_args()
    
    # Set up logging to file and console
    log_file = f"live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Add handlers to root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    
    # Log platform information
    logger.info(f"Starting live trading with model {args.model}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Operating system: {os.name}")
    
    # Log detailed configuration for transparency
    logger.info("====== Configuration ======")
    logger.info(f"- Model path: {args.model}")
    logger.info(f"- Contract: {args.contract}")
    logger.info(f"- Paper trading: {args.paper}")
    logger.info(f"- Risk management: {not args.no_risk}")
    logger.info(f"- Transaction costs in config: {config['environment'].get('transaction_cost', 0.0)}")
    logger.info(f"- Position size in config: {config['environment'].get('position_size', 1)}")
    
    # Log indicators configuration from config.yaml
    logger.info("====== Enabled Indicators in config.yaml ======")
    for indicator, settings in config["indicators"].items():
        if isinstance(settings, dict) and settings.get("enabled", False):
            logger.info(f"- {indicator}: {settings}")
    
    # Display startup banner
    logger.info("=" * 50)
    logger.info("STARTING LIVE TRADING SYSTEM")
    logger.info("=" * 50)
    
    # Setup and start the IB connection
    ib = IB()
    
    # Try connecting to IB
    try:
        # Use 7496 for TWS live, 7497 for TWS paper, 4002 for Gateway (paper)
        if args.paper:
            # Connect to paper trading
            logger.info("Connecting to IB Paper Trading")
            ib.connect('127.0.0.1', 7497, clientId=1)
        else:
            # Connect to live trading
            logger.info("Connecting to IB Live Trading")
            ib.connect('127.0.0.1', 7496, clientId=1)
            
        # Log connection status    
        logger.info(f"IB connection established: {ib.isConnected()}")
        
        # Create model trader instance (use the global variable)
        model_trader = ModelTrader(ib, model_path=args.model, 
                                 use_risk_management=not args.no_risk)
        
        # Get the most liquid contract for the chosen symbol
        current_date = datetime.now()
        contract = None
        
        if args.contract == "NQ":
            # NQ futures - get the most recent contract
            logger.info("Setting up NQ Futures contract")
            contracts = ib.reqContractDetails(Future(symbol='NQ', exchange='CME'))
            contract = get_most_recent_contract(contracts)
            
            if contract:
                logger.info(f"Selected contract: {contract.localSymbol}")
                model_trader.set_active_contract(contract)
            else:
                logger.error("No valid contract found for NQ!")
                sys.exit(1)
                
        elif args.contract == "ES":
            # ES futures - get the most recent contract
            logger.info("Setting up ES Futures contract")
            contracts = ib.reqContractDetails(Future(symbol='ES', exchange='CME'))
            contract = get_most_recent_contract(contracts)
            
            if contract:
                logger.info(f"Selected contract: {contract.localSymbol}")
                model_trader.set_active_contract(contract)
            else:
                logger.error("No valid contract found for ES!")
                sys.exit(1)
                
        else:
            logger.error(f"Unsupported contract symbol: {args.contract}")
            sys.exit(1)
        
        # Fetch historical bars to initialize the model
        historical_data_success = model_trader.fetch_historical_bars(days_back=30)
        if not historical_data_success:
            logger.error("Failed to fetch historical data")
            sys.exit(1)
        
        # Test loading the model to ensure it's working
        logger.info("Verifying model is loaded correctly")
        if model_trader.model is None:
            logger.error("Model failed to load!")
            sys.exit(1)
        
        # Set up real-time bar subscription (5 second bars)
        logger.info("Setting up real-time bar subscription")
        bars = ib.reqRealTimeBars(contract, 5, 'TRADES', False)
        bars.updateEvent += onBar
        
        # Log the bar aggregation setup for clarity
        logger.info("====== Bar Aggregation Setup ======")
        logger.info("- Raw bar interval: 5 seconds (from IB API)")
        logger.info("- Aggregated bar interval: 5 minutes")
        logger.info("- Aggregation function: aggregate_bars")
        logger.info("- Buffer: bar_buckets (organized by 5-minute intervals)")
        logger.info("- Processing: synchronize_bars (checks for completed intervals)")
        
        # Start heartbeat monitoring thread
        heartbeat_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
        heartbeat_thread.start()
        logger.info("Heartbeat monitoring started")
        
        # Load saved state if available
        model_trader.load_state()
        
        # Add force reconnect timer
        ib.setTimeout(120)  # 2-minute timeout for IB API calls
        
        # Main loop - just keep the connection alive
        logger.info("Entering main loop, waiting for bar events")
        ib.run()
        
    except Exception as e:
        logger.error(f"Error in main thread: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Attempt to save state before exit
        if model_trader:
            model_trader.save_state()
            
        # Ensure IB connection is closed
        if ib and ib.isConnected():
            ib.disconnect()
            logger.info("IB connection closed")

