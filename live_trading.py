from ib_insync import IB, Future, MarketOrder, util, Order, LimitOrder, StopOrder, BracketOrder
import datetime
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

# Import stable_baselines3 for loading the trained model
from stable_baselines3 import PPO

# Import money module and config for risk management
import money
from config import config

# Import process_technical_indicators and ensure_numeric from get_data
from get_data import process_technical_indicators, ensure_numeric

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for bar buffer and model trader
bar_buffer = []
model_trader = None

# Heartbeat monitoring globals
last_data_timestamp = datetime.now()
heartbeat_interval = 60  # Seconds between heartbeat checks
is_data_flowing = False
reconnection_attempts = 0
max_reconnection_attempts = 5
state_file = "trader_state.pkl"

# Globals for real-time bar handling
bar_buckets = defaultdict(list)  # Organize bars by 5-minute intervals
processed_intervals = set()  # Track which intervals we've already processed

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
        
        # Create and return the aggregated bar
        return {
            'time': start_time,  # Use start time for indexing
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

def synchronize_bars():
    """
    Check if we have complete 5-minute bars to process.
    Returns the latest complete bar if available, None otherwise.
    """
    global bar_buckets, processed_intervals, model_trader
    
    # Get current time and calculate the most recently completed 5-minute interval
    now = datetime.now()
    
    # Go back 1 minute to ensure we're not trying to process a partial interval
    one_minute_ago = now - timedelta(minutes=1)
    last_complete_interval = one_minute_ago.replace(minute=(one_minute_ago.minute // 5) * 5, second=0, microsecond=0)
    
    # Format as a string key
    interval_key = last_complete_interval.strftime('%Y-%m-%d %H:%M')
    
    # Check if we have bars for this interval and haven't processed it yet
    if interval_key in bar_buckets and interval_key not in processed_intervals:
        interval_bars = bar_buckets[interval_key]
        
        # Only process if we have a reasonable number of bars (at least 10)
        if len(interval_bars) >= 10:
            logger.info(f"Processing synchronized 5-minute bar for interval {interval_key} with {len(interval_bars)} 5-second bars")
            
            # Handle potential errors in bar aggregation
            try:
                five_min_bar = aggregate_bars(interval_bars)
                processed_intervals.add(interval_key)
            except Exception as e:
                logger.error(f"Error aggregating bars for interval {interval_key}: {e}")
                return None
            
            # Clean up old buckets to prevent memory growth
            current_time = datetime.now()
            for old_key in list(bar_buckets.keys()):
                try:
                    old_time = datetime.strptime(old_key, '%Y-%m-%d %H:%M')
                    if (current_time - old_time).total_seconds() > 3600:  # Older than 1 hour
                        del bar_buckets[old_key]
                except Exception as e:
                    logger.warning(f"Error cleaning up old bar bucket {old_key}: {e}")
            
            return five_min_bar
    
    return None

class ModelTrader:
    def __init__(self, ib_instance, model_path="best_model", use_risk_management=True):
        """
        Initialize the ModelTrader.
        
        Args:
            ib_instance: Interactive Brokers API instance
            model_path: Path to the saved model
            use_risk_management: Whether to use risk management
        """
        self.ib = ib_instance
        self.model_path = model_path
        self.model = None
        self.active_contract = None
        self.current_position = 0  # -1 for short, 0 for flat, 1 for long
        self.expected_position = 0
        self.use_risk_management = use_risk_management
        self.active_order = None
        self.active_orders = {}
        self.order_status = {}
        self.bar_history = []
        self.min_bars_needed = 100  # Minimum number of bars needed for indicator calculation
        
        # Risk management settings from config.yaml
        self.risk_config = config.get("risk_management", {})
        self.stop_loss_pct = None
        self.take_profit_pct = None
        self.trailing_stop_pct = None
        self.position_size = 1
        
        # Position tracking for verification
        self.last_position_check = datetime.now()
        self.position_check_interval = 60  # seconds
        
        # Initialize risk parameters if risk management is enabled
        if self.use_risk_management:
            self._initialize_risk_parameters()
        
        # Get enabled indicators using the improved method that checks multiple sources
        self.enabled_indicators = self._get_enabled_indicators()
        
        logger.info(f"ModelTrader initialized with model path: {model_path}")
        logger.info(f"Risk management: {use_risk_management}")
        logger.info(f"Enabled indicators: {self.enabled_indicators}")
        if use_risk_management:
            logger.info(f"Stop Loss: {self.stop_loss_pct}%, Take Profit: {self.take_profit_pct}%, Trailing Stop: {self.trailing_stop_pct}%")
    
    def _initialize_risk_parameters(self):
        """Initialize risk management parameters from config.yaml."""
        # Check if risk management is enabled globally
        risk_enabled = self.risk_config.get("enabled", False)
        if not risk_enabled:
            logger.info("Risk management is disabled in config.yaml")
            return
        
        # Get portfolio value
        portfolio_value = config.get("environment", {}).get("initial_balance", 10000.0)
        point_value = 20.0  # $20 per point for NQ futures
        
        # Stop loss configuration
        stop_loss_config = self.risk_config.get("stop_loss", {})
        if stop_loss_config.get("enabled", False):
            self.stop_loss_pct = stop_loss_config.get("percentage", 1.0)
            max_loss_dollars = portfolio_value * (self.stop_loss_pct / 100)
            max_loss_points = max_loss_dollars / point_value
            
            logger.info(f"Stop loss enabled from config: {self.stop_loss_pct}%")
            logger.info(f"This equals ${max_loss_dollars:.2f} of the ${portfolio_value:.2f} portfolio")
            logger.info(f"For NQ futures, this represents {max_loss_points:.1f} points")
        else:
            logger.info("Stop loss is disabled in config")
            self.stop_loss_pct = None
        
        # Take profit configuration
        take_profit_config = self.risk_config.get("take_profit", {})
        if take_profit_config.get("enabled", False):
            self.take_profit_pct = take_profit_config.get("percentage", 2.0)
            target_profit_dollars = portfolio_value * (self.take_profit_pct / 100)
            target_points = target_profit_dollars / point_value
            
            logger.info(f"Take profit enabled from config: {self.take_profit_pct}%")
            logger.info(f"This equals ${target_profit_dollars:.2f} of the ${portfolio_value:.2f} portfolio")
            logger.info(f"For NQ futures, this represents {target_points:.1f} points")
            logger.info(f"Example: If entry at 18000, take profit would be at {18000 + target_points:.1f} for a long position")
        else:
            logger.info("Take profit is disabled in config")
            self.take_profit_pct = None
        
        # Trailing stop configuration
        trailing_stop_config = self.risk_config.get("trailing_stop", {})
        if trailing_stop_config.get("enabled", False):
            self.trailing_stop_pct = trailing_stop_config.get("percentage", 0.5)
            trailing_dollars = portfolio_value * (self.trailing_stop_pct / 100)
            trailing_points = trailing_dollars / point_value
            
            logger.info(f"Trailing stop enabled from config: {self.trailing_stop_pct}%")
            logger.info(f"This equals ${trailing_dollars:.2f} of the ${portfolio_value:.2f} portfolio")
            logger.info(f"For NQ futures, this represents {trailing_points:.1f} points")
        else:
            logger.info("Trailing stop is disabled in config")
            self.trailing_stop_pct = None
        
        # Position sizing configuration
        position_sizing_config = self.risk_config.get("position_sizing", {})
        if position_sizing_config.get("enabled", False):
            self.position_size = position_sizing_config.get("size_multiplier", 1.0)
            logger.info(f"Position sizing enabled from config: {self.position_size}")
        else:
            logger.info(f"Position sizing is disabled in config, using default: {self.position_size}")
    
    def _get_enabled_indicators(self):
        """Get list of enabled indicators from config"""
        enabled_indicators = []
        
        # Check configuration
        try:
            indicators_config = config.get("indicators", {})
            
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
            
            # Always include day of week cyclical indicators
            enabled_indicators.extend(["DOW_SIN", "DOW_COS"])
            
            # Include minutes since market open indicators if enabled
            if indicators_config.get("minutes_since_open", {}).get("enabled", False):
                enabled_indicators.extend(["MSO_SIN", "MSO_COS"])
            
            # Try to load from enabled_indicators.json if it exists
            indicators_file = os.path.join(os.path.dirname(self.model_path), "enabled_indicators.json")
            if os.path.exists(indicators_file):
                logger.info(f"Loading indicators from {indicators_file}")
                try:
                    with open(indicators_file, "r") as f:
                        file_indicators = json.load(f)
                        
                    # If the file indicators don't contain DOW features, add them
                    if "DOW_SIN" not in file_indicators and "DOW_COS" not in file_indicators:
                        file_indicators.extend(["DOW_SIN", "DOW_COS"])
                        logger.info("Added day of week indicators to loaded indicators")
                    
                    # Check if the minutes since open indicators should be included
                    if indicators_config.get("minutes_since_open", {}).get("enabled", False):
                        if "MSO_SIN" not in file_indicators and "MSO_COS" not in file_indicators:
                            file_indicators.extend(["MSO_SIN", "MSO_COS"])
                            logger.info("Added minutes since market open indicators to loaded indicators")
                        
                    # Use the indicators from file instead
                    return file_indicators
                except Exception as e:
                    logger.error(f"Error loading indicators from file: {e}")
            
            return enabled_indicators
        except Exception as e:
            logger.error(f"Error getting enabled indicators: {e}")
            return ["RSI", "CCI", "DOW_SIN", "DOW_COS"]  # Default minimum set
    
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
        Diagnose the model's expected feature input to help reconcile with our actual features.
        """
        if self.model is None:
            logger.error("No model loaded to diagnose")
            return
        
        try:
            # Check observation space
            if hasattr(self.model, 'observation_space'):
                obs_shape = self.model.observation_space.shape
                logger.info(f"Model observation space shape: {obs_shape}")
                
                # Try to extract feature information from the model's policy
                if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'features_extractor'):
                    logger.info(f"Model has a features extractor: {self.model.policy.features_extractor}")
                    
                # Check if environment is stored
                if hasattr(self.model, 'env'):
                    logger.info(f"Model has stored environment information")
                
                # Log expected number of features
                expected_features = obs_shape[0]
                logger.info(f"Model expects {expected_features} features")
                
                # Filter out supertrend from indicators if it exists
                filtered_indicators = [ind for ind in self.enabled_indicators if ind.upper() != 'SUPERTREND']
                
                # Log our enabled indicators to compare
                logger.info(f"We have {len(filtered_indicators) + 2} features (close_norm + {len(filtered_indicators)} indicators + position)")
                logger.info(f"Raw enabled indicators from config: {self.enabled_indicators}")
                logger.info(f"Filtered indicators for observation: {filtered_indicators}")
                
                # Calculate and log the feature delta
                feature_delta = expected_features - (len(filtered_indicators) + 2)
                if feature_delta > 0:
                    logger.error(f"MISSING {feature_delta} FEATURES compared to what model expects")
                    # Try to determine what might be missing by comparing with common indicator sets
                    possible_missing = []
                    logger.info("Attempting to identify possible missing indicators:")
                    
                    # Check common indicator combinations from environment.py
                    common_indicators = ["SUPERTREND", "RSI", "CCI", "ADX", "ADX_POS", "ADX_NEG", 
                                       "STOCH_K", "STOCH_D", "MACD", "MACD_SIGNAL", "MACD_HIST", 
                                       "ROC", "WILLIAMS_R", "SMA_NORM", "EMA_NORM", "DISPARITY", 
                                       "ATR", "OBV_NORM", "CMF", "PSAR_NORM", "PSAR_DIR", "VOLUME_MA"]
                    
                    for indicator in common_indicators:
                        if indicator not in filtered_indicators and indicator.upper() != 'SUPERTREND':
                            possible_missing.append(indicator)
                    
                    logger.info(f"Possible missing indicators: {possible_missing[:feature_delta]}")
                elif feature_delta < 0:
                    logger.error(f"EXTRA {-feature_delta} FEATURES compared to what model expects")
                else:
                    logger.info("Feature count matches model's expectations")
                
                # Try to find and load any training metadata that might exist
                metadata_paths = ["training_metadata.json", "model_metadata.json", "training_config.json"]
                for path in metadata_paths:
                    if os.path.exists(path):
                        try:
                            with open(path, 'r') as f:
                                metadata = json.load(f)
                                if 'indicators' in metadata:
                                    logger.info(f"Found training indicators in {path}: {metadata['indicators']}")
                                    break
                        except Exception as e:
                            logger.warning(f"Failed to load {path}: {e}")
                
        except Exception as e:
            logger.error(f"Error diagnosing model features: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def set_active_contract(self, contract):
        """Set the active trading contract."""
        self.active_contract = contract
        logger.info(f"Active contract set to {contract}")
    
    def save_state(self):
        """Save the current trading state to a file."""
        state = {
            'current_position': self.current_position,
            'expected_position': self.expected_position,
            'active_orders': self.active_orders,
            'bar_history': self.bar_history
        }
        
        try:
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
            logger.info("Trading state saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving trading state: {e}")
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
            logger.info(f"Active contract: {self.active_contract}")
            logger.info(f"Active contract details: Symbol={self.active_contract.symbol}, Exchange={self.active_contract.exchange}, SecType={self.active_contract.secType}")
            
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
                if (position.contract.symbol == self.active_contract.symbol and 
                    position.contract.secType == self.active_contract.secType):
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
    
    def preprocess_bar(self, bar):
        """
        Preprocess a single bar into features expected by the model using the same
        process_technical_indicators function used during training.
        
        Args:
            bar: A dictionary containing OHLCV data
            
        Returns:
            np.ndarray: Observation array for the model
        """
        # Add the new bar to history
        self.bar_history.append(bar)
        
        # Keep only the most recent bars
        if len(self.bar_history) > 300:  # Keep more history for better indicator calculation
            self.bar_history = self.bar_history[-300:]
        
        # If we don't have enough bars yet, return None
        if len(self.bar_history) < self.min_bars_needed:
            logger.info(f"Not enough bars yet. Have {len(self.bar_history)}/{self.min_bars_needed}")
            return None
        
        try:
            # Convert bar history to DataFrame with proper column names
            df = pd.DataFrame(self.bar_history)
            
            # Convert time column to datetime if it's not already
            if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
                try:
                    # Handle timezone-aware datetimes by explicitly using UTC=True
                    df['time'] = pd.to_datetime(df['time'], utc=True)
                except Exception as e:
                    logger.warning(f"Error converting times with utc=True: {e}")
                    try:
                        # Try to remove timezone information as a fallback
                        df['time'] = df['time'].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') else x)
                        df['time'] = pd.to_datetime(df['time'])
                    except Exception as e2:
                        logger.error(f"Failed to process timestamps: {e2}")
                        # Create a simple datetime index as last resort
                        logger.warning("Creating a synthetic time index as fallback")
                        now = datetime.now()
                        start_time = now - timedelta(minutes=len(df))
                        synthetic_times = [start_time + timedelta(minutes=i) for i in range(len(df))]
                        df['time'] = synthetic_times
            
            # Set time as index
            df = df.set_index('time')
            
            # Rename OHLCV columns to match the format expected by process_technical_indicators
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Calculate close_norm if not already present
            if 'close_norm' not in df.columns and 'Close_norm' not in df.columns and 'CLOSE_NORM' not in df.columns:
                df['close_norm'] = df['Close'].pct_change().fillna(0)
            
            # Apply technical indicators processing
            processed_df = process_technical_indicators(df)
            
            # Add day of week features if missing
            if 'DOW_SIN' not in processed_df.columns or 'DOW_COS' not in processed_df.columns:
                logger.info("Adding missing day of week features")
                # Calculate day of week (0=Monday, 6=Sunday)
                processed_df['DOW'] = processed_df.index.dayofweek
                
                # Convert to sine and cosine representation
                processed_df['DOW_SIN'] = np.sin(2 * np.pi * processed_df['DOW'] / 7)
                processed_df['DOW_COS'] = np.cos(2 * np.pi * processed_df['DOW'] / 7)
                logger.info(f"Added missing day of week features: DOW={processed_df['DOW'].iloc[-1]}, " +
                           f"DOW_SIN={processed_df['DOW_SIN'].iloc[-1]:.4f}, DOW_COS={processed_df['DOW_COS'].iloc[-1]:.4f}")
            
            # Add minutes since market open features if missing and enabled
            if ('MSO_SIN' not in processed_df.columns or 'MSO_COS' not in processed_df.columns) and \
               config["indicators"].get("minutes_since_open", {}).get("enabled", False):
                logger.info("Adding minutes since market open features")
                
                # Convert to Eastern Time (market time)
                eastern = pytz.timezone('US/Eastern')
                
                # Create a column with the Eastern time
                processed_df['time_et'] = processed_df.index.tz_localize(pytz.UTC).tz_convert(eastern)
                
                # Cash market opens at 9:30 AM ET
                processed_df['minutes_since_open'] = (processed_df['time_et'].dt.hour - 9) * 60 + processed_df['time_et'].dt.minute - 30
                
                # Handle times before market open (negative values)
                processed_df.loc[processed_df['minutes_since_open'] < 0, 'minutes_since_open'] = 0
                
                # Handle times after market close (> 390 minutes)
                processed_df.loc[processed_df['minutes_since_open'] > 390, 'minutes_since_open'] = 390
                
                # Normalize to [0, 1] range (390 minutes = 6.5 hours of trading day)
                processed_df['minutes_since_open_norm'] = processed_df['minutes_since_open'] / 390.0
                
                # Convert to sine and cosine representation (circular encoding)
                processed_df['MSO_SIN'] = np.sin(2 * np.pi * processed_df['minutes_since_open_norm'])
                processed_df['MSO_COS'] = np.cos(2 * np.pi * processed_df['minutes_since_open_norm'])
                
                logger.info(f"Added minutes since open features: Minutes={processed_df['minutes_since_open'].iloc[-1]}, " +
                           f"MSO_SIN={processed_df['MSO_SIN'].iloc[-1]:.4f}, MSO_COS={processed_df['MSO_COS'].iloc[-1]:.4f}")
                
                # Clean up temporary columns
                processed_df = processed_df.drop(['time_et', 'minutes_since_open', 'minutes_since_open_norm'], axis=1)
            
            # Use the last row of processed dataframe to build observation vector
            last_row = processed_df.iloc[-1]
            
            # Build observation vector EXACTLY as done in environment.py in _get_obs()
            # Start with close_norm
            if 'close_norm' in processed_df.columns:
                close_norm = last_row['close_norm']
            elif 'Close_norm' in processed_df.columns:  
                close_norm = last_row['Close_norm']
            elif 'CLOSE_NORM' in processed_df.columns:
                close_norm = last_row['CLOSE_NORM']
            else:
                logger.error("No normalized close price column found")
                return None
            
            # Start observation vector with close_norm
            obs = [float(close_norm)]

            # IMPORTANT: Filter out the supertrend from the enabled indicators
            # because it is handled separately in the preprocessing
            filtered_indicators = [ind for ind in self.enabled_indicators if ind.upper() != 'SUPERTREND']
            
            # Log enabled indicators to help debug
            logger.info(f"Using {len(filtered_indicators)} enabled indicators from config: {filtered_indicators}")
            
            # Add indicators in EXACT SAME ORDER as specified in self.enabled_indicators
            # This matches how environment.py builds observations
            missing_indicators = []
            for indicator in filtered_indicators:
                # Try different case variations
                found = False
                for col_variant in [indicator, indicator.lower(), indicator.upper()]:
                    if col_variant in processed_df.columns:
                        obs.append(float(last_row[col_variant]))
                        found = True
                        break
                
                if not found:
                    missing_indicators.append(indicator)
                    logger.warning(f"Indicator {indicator} not found in processed data")
                    # Add a placeholder value of 0 to maintain vector size and order
                    obs.append(0.0)
            
            if missing_indicators:
                logger.error(f"Missing indicators: {missing_indicators}. This may cause model prediction errors.")
            
            # Add current position to observation (always the last feature)
            # This uses the same values as in environment.py: 1 for long, -1 for short, 0 for flat
            obs.append(float(self.current_position))
            
            # Convert to numpy array with proper dtype
            obs = np.array(obs, dtype=np.float32)
            
            # Check if our observation matches the expected shape from the model
            if self.model and hasattr(self.model, 'observation_space'):
                expected_shape = self.model.observation_space.shape[0]
                actual_shape = len(obs)
                
                # Log the shapes for debugging
                logger.info(f"Model expects observation shape: {expected_shape}, observation has shape: {actual_shape}")
                
                if actual_shape != expected_shape:
                    logger.error(f"CRITICAL ERROR: Observation shape mismatch: {actual_shape} vs expected {expected_shape}")
                    logger.error(f"This means the enabled indicators in config.yaml DON'T match what was used in training")
                    logger.error(f"Current observation: {obs}")
                    logger.error(f"Enabled indicators: {self.enabled_indicators}")
                    logger.error(f"Filtered indicators: {filtered_indicators}")
                    return None
            
            # Ensure observation values are within appropriate ranges
            obs = np.clip(obs, -1.0, 1.0)  # Most indicators should be in [-1, 1]
            
            # But close_norm should be in [0, 1]
            if len(obs) > 0:
                obs[0] = np.clip(obs[0], 0.0, 1.0)
            
            logger.debug(f"Final observation: {obs}")
            return obs
            
        except Exception as e:
            logger.error(f"Error processing bar: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_prediction(self, observation):
        """
        Get a trading action prediction from the model.
        
        Args:
            observation: Preprocessed observation array
            
        Returns:
            int: Action (0=long, 1=short, 2=hold)
        """
        if self.model is None:
            logger.error("Model not loaded. Cannot make prediction.")
            return 2  # Default to hold if model not loaded
        
        # Print feature vector for debugging
        logger.info(f"Feature vector for prediction: {observation}")
        
        # Print feature vector with names for better interpretation
        feature_names = ["close_norm"]
        # Filter out supertrend from indicators list
        filtered_indicators = [ind for ind in self.enabled_indicators if ind.upper() != 'SUPERTREND']
        feature_names.extend(filtered_indicators)
        feature_names.append("position")
        
        # Create a dictionary mapping feature names to values
        feature_dict = {}
        for i, name in enumerate(feature_names):
            if i < len(observation):
                feature_dict[name] = observation[i]
        
        logger.info(f"Named features: {feature_dict}")
        
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)
    
    def execute_trade(self, action, bar):
        """
        Execute a trade based on the model's action.
        
        Args:
            action: Model action (0=long, 1=short, 2=hold)
            bar: Current price bar
        """
        if self.active_contract is None:
            logger.error("No active contract set. Cannot execute trade.")
            return
        
        # Get current actual position from IB before making decisions
        self.verify_position(force=True)
        actual_position_quantity = self.expected_position  # This should be updated by verify_position
        
        current_price = bar['close']
        logger.info(f"Current price: {current_price}, Action: {action}, Current position: {self.current_position}, Quantity: {actual_position_quantity}")
        
        # Check if we're already in the position the model wants
        same_direction = (action == 0 and self.current_position == 1) or (action == 1 and self.current_position == -1)
        
        # If the model wants the same position we already have, keep existing orders and position
        if same_direction:
            if (action == 0 and actual_position_quantity == self.position_size) or (action == 1 and actual_position_quantity == -self.position_size):
                logger.info(f"Already have the target {'long' if action == 0 else 'short'} position with correct size. Keeping position and existing orders.")
                return
        
        # If we're changing direction or don't have an active position, cancel existing orders
        if not same_direction:
            # Cancel ALL existing orders for this contract when we have a new signal
            open_trades = self.ib.openTrades()
            canceled_orders = 0
            
            for trade in open_trades:
                if (trade.contract and trade.contract.symbol == self.active_contract.symbol and
                    trade.isActive()):
                    logger.info(f"Cancelling order: {trade.order.orderId} ({trade.order.action} {trade.order.totalQuantity} @ {trade.order.orderType})")
                    self.ib.cancelOrder(trade.order)
                    canceled_orders += 1
                    
            if canceled_orders > 0:
                logger.info(f"Canceled {canceled_orders} existing orders for {self.active_contract.symbol}")
                # Wait briefly for cancellations to be processed
                self.ib.sleep(0.5)
            
            # Cancel any existing tracked order
            if self.active_order:
                logger.info(f"Cancelling tracked order: {self.active_order}")
                self.ib.cancelOrder(self.active_order)
                self.active_order = None
        
        # Long signal
        if action == 0:
            # If we're currently short, we need to exit that position first
            if self.current_position == -1:
                logger.info("Exiting short position before entering long")
                self._exit_position()
                # Wait for position to be verified as closed before proceeding
                self.verify_position(force=True)
                if self.current_position != 0:
                    logger.warning("Position still not closed, waiting for exit order to fill")
                    return
            
            # Now we can enter the long position
            if actual_position_quantity == self.position_size:
                logger.info(f"Already have target long position of {self.position_size} contracts. Maintaining position.")
                return
            elif actual_position_quantity == 0:
                # Enter full long position
                logger.info(f"Entering full long position of {self.position_size} contracts")
                self._enter_position(1, current_price)
            elif 0 < actual_position_quantity < self.position_size:
                # Increase position
                additional_quantity = self.position_size - actual_position_quantity
                logger.info(f"Increasing long position by {additional_quantity} contracts")
                self._enter_position(1, current_price, additional_quantity)
            elif actual_position_quantity > self.position_size:
                # Reduce position
                excess = actual_position_quantity - self.position_size
                logger.info(f"Reducing long position by {excess} contracts")
                self._reduce_position(excess, is_long=True)
        
        # Short signal
        elif action == 1:
            # If we're currently long, we need to exit that position first
            if self.current_position == 1:
                logger.info("Exiting long position before entering short")
                self._exit_position()
                # Wait for position to be verified as closed before proceeding
                self.verify_position(force=True)
                if self.current_position != 0:
                    logger.warning("Position still not closed, waiting for exit order to fill")
                    return
            
            # Now we can enter the short position
            if actual_position_quantity == -self.position_size:
                logger.info(f"Already have target short position of {self.position_size} contracts. Maintaining position.")
                return
            elif actual_position_quantity == 0:
                # Enter full short position
                logger.info(f"Entering full short position of {self.position_size} contracts")
                self._enter_position(-1, current_price)
            elif -self.position_size < actual_position_quantity < 0:
                # Increase position
                additional_quantity = self.position_size + actual_position_quantity  # actual_pos is negative
                logger.info(f"Increasing short position by {additional_quantity} contracts")
                self._enter_position(-1, current_price, additional_quantity)
            elif actual_position_quantity < -self.position_size:
                # Reduce position
                excess = -actual_position_quantity - self.position_size  # Convert to positive
                logger.info(f"Reducing short position by {excess} contracts")
                self._reduce_position(excess, is_long=False)
        
        # Hold signal - do nothing
        elif action == 2:
            logger.info("Hold signal - maintaining current position")
    
    def _enter_position(self, direction, price, quantity=None):
        """
        Enter a new position with optional stop loss and take profit.
        
        Args:
            direction: 1 for long, -1 for short
            price: Current price for order execution
            quantity: Number of contracts to trade, defaults to self.position_size
        """
        if quantity is None:
            quantity = self.position_size
            
        action_text = "BUY" if direction == 1 else "SELL"
        logger.info(f"Entering {action_text} position with {quantity} contracts at price {price}")
        
        # If risk management is enabled, create a bracket order with take profit and stop loss
        if self.use_risk_management and (self.take_profit_pct is not None or self.stop_loss_pct is not None):
            self._create_bracket_order(direction, price, quantity)
        else:
            # Simple market order without risk management
            order = MarketOrder(action_text, quantity)
            trade = self.ib.placeOrder(self.active_contract, order)
            self.active_order = order
            
            # Track the order
            order_id = order.orderId if hasattr(order, 'orderId') else None
            if order_id:
                self.active_orders[order_id] = {
                    'action': action_text,
                    'quantity': quantity,
                    'price': price,
                    'time': datetime.now()
                }
                self.order_status[order_id] = "Submitted"
                
            # Register for order status updates
            trade.statusEvent += self.update_order_status
        
        # Update expected position (actual position will be verified later)
        self.expected_position = direction * quantity
        
        # Save state after placing order
        self.save_state()
        
        # Update current position
        self.current_position = direction
        
    def _reduce_position(self, quantity, is_long=True):
        """
        Reduce an existing position by a specific quantity.
        
        Args:
            quantity: Number of contracts to reduce by
            is_long: Whether the current position is long (True) or short (False)
        """
        # To reduce a position, we do the opposite action
        action_text = "SELL" if is_long else "BUY"
        
        logger.info(f"Reducing position with {action_text} {quantity} contracts")
        
        # Create market order to reduce position
        order = MarketOrder(action_text, quantity)
        trade = self.ib.placeOrder(self.active_contract, order)
        
        # Track the order
        order_id = order.orderId if hasattr(order, 'orderId') else None
        if order_id:
            self.active_orders[order_id] = {
                'action': action_text,
                'quantity': quantity,
                'price': 0,  # We don't know the price for market orders
                'time': datetime.now(),
                'type': 'reduce'
            }
            self.order_status[order_id] = "Submitted"
            
        # Register for order status updates
        trade.statusEvent += self.update_order_status
    
    def _exit_position(self):
        """Exit the current position."""
        if self.current_position == 0:
            logger.info("No position to exit")
            return
        
        action_text = "SELL" if self.current_position == 1 else "BUY"
        quantity = self.position_size
        
        logger.info(f"Exiting position with {action_text} {quantity} contracts")
        
        # Create market order to exit position
        order = MarketOrder(action_text, quantity)
        trade = self.ib.placeOrder(self.active_contract, order)
        
        # Track the order
        order_id = order.orderId if hasattr(order, 'orderId') else None
        if order_id:
            self.active_orders[order_id] = {
                'action': action_text,
                'quantity': quantity,
                'price': 0,  # We don't know the price for market orders
                'time': datetime.now(),
                'type': 'exit'
            }
            self.order_status[order_id] = "Submitted"
            
            # Register for order status updates
            trade.statusEvent += self.update_order_status
        
        # Note: Don't update current_position here as it will be updated when the order is filled
        # Just track that we're in the process of exiting
        logger.info("Position exit order submitted - waiting for confirmation")
    
    def _create_bracket_order(self, direction, price, quantity):
        """
        Create a bracket order with take profit and stop loss.
        
        Args:
            direction: 1 for long, -1 for short
            price: Current price
            quantity: Number of contracts
        """
        action_text = "BUY" if direction == 1 else "SELL"
        opposite_action = "SELL" if direction == 1 else "BUY"
        
        # Calculate take profit and stop loss prices
        # NQ futures contract value = $20 per point
        point_value = 20.0  # $20 per point for NQ futures
        
        # Portfolio value from config (default to $10,000 if not specified)
        portfolio_value = config.get("environment", {}).get("initial_balance", 10000.0)
        
        # Set minimum tick size for the contract - NQ futures use 0.25 point increments
        min_tick_size = 0.25
        
        # For bracket orders, we need price levels based on dollar profit targets
        if self.take_profit_pct is not None:
            # Calculate how much we want to make in dollars
            target_profit_dollars = portfolio_value * (self.take_profit_pct / 100)
            
            # Convert dollar profit to points
            target_points = target_profit_dollars / point_value
            
            # Calculate take profit price
            if direction == 1:  # Long position
                take_profit_price = price + target_points
            else:  # Short position
                take_profit_price = price - target_points
            
            # Round to the correct tick size
            take_profit_price = round(take_profit_price / min_tick_size) * min_tick_size
            
            logger.info(f"Take profit set at {take_profit_price}")
            logger.info(f"Target profit: ${target_profit_dollars:.2f} ({self.take_profit_pct}% of ${portfolio_value:.2f})")
            logger.info(f"This equals {target_points} points for NQ futures")
        else:
            take_profit_price = None
            logger.info("Take profit is disabled")
        
        if self.stop_loss_pct is not None:
            # Calculate how much we're willing to lose in dollars
            max_loss_dollars = portfolio_value * (self.stop_loss_pct / 100)
            
            # Convert dollar loss to points
            loss_points = max_loss_dollars / point_value
            
            # Calculate stop loss price
            if direction == 1:  # Long position
                stop_loss_price = price - loss_points
            else:  # Short position
                stop_loss_price = price + loss_points
            
            # Round to the correct tick size
            stop_loss_price = round(stop_loss_price / min_tick_size) * min_tick_size
            
            logger.info(f"Stop loss set at {stop_loss_price}")
            logger.info(f"Max loss: ${max_loss_dollars:.2f} ({self.stop_loss_pct}% of ${portfolio_value:.2f})")
            logger.info(f"This equals {loss_points} points for NQ futures")
        else:
            stop_loss_price = None
            logger.info("Stop loss is disabled")
            
        # Check if trailing stop is enabled
        if self.trailing_stop_pct is not None:
            # Calculate trailing stop in dollars
            trailing_stop_dollars = portfolio_value * (self.trailing_stop_pct / 100)
            trailing_points = trailing_stop_dollars / point_value
            logger.info(f"Trailing stop is enabled: ${trailing_stop_dollars:.2f} ({self.trailing_stop_pct}% of portfolio)")
            logger.info(f"This equals {trailing_points} points for NQ futures")
        else:
            logger.info("Trailing stop is disabled")
        
        # Create parent order
        parent_order = Order()
        parent_order.action = action_text
        parent_order.orderType = "MKT"
        parent_order.totalQuantity = quantity
        parent_order.transmit = False  # Set to False until we've created child orders
        
        # Place the parent order to get an orderId
        parent_trade = self.ib.placeOrder(self.active_contract, parent_order)
        parent_id = parent_order.orderId
        
        # Creating a bracket order requires placing multiple orders
        bracket_orders = [parent_order]
        
        # Create take profit order
        if take_profit_price is not None:
            take_profit_order = Order()
            take_profit_order.action = opposite_action
            take_profit_order.orderType = "LMT"
            take_profit_order.totalQuantity = quantity
            take_profit_order.lmtPrice = take_profit_price
            take_profit_order.parentId = parent_id
            take_profit_order.transmit = stop_loss_price is None  # Transmit if this is the last order
            
            bracket_orders.append(take_profit_order)
            
            # Track this order in our active_orders
            self.active_orders[take_profit_order.orderId] = {
                'action': opposite_action,
                'quantity': quantity,
                'price': take_profit_price,
                'time': datetime.now(),
                'type': 'take_profit',
                'parent_id': parent_id
            }
        
        # Create stop loss order
        if stop_loss_price is not None:
            stop_loss_order = Order()
            stop_loss_order.action = opposite_action
            stop_loss_order.orderType = "STP"
            stop_loss_order.totalQuantity = quantity
            stop_loss_order.auxPrice = stop_loss_price
            stop_loss_order.parentId = parent_id
            stop_loss_order.transmit = True  # Always transmit the last order
            
            bracket_orders.append(stop_loss_order)
            
            # Track this order in our active_orders
            self.active_orders[stop_loss_order.orderId] = {
                'action': opposite_action,
                'quantity': quantity,
                'price': stop_loss_price,
                'time': datetime.now(),
                'type': 'stop_loss',
                'parent_id': parent_id
            }
        
        # If no child orders, we need to transmit the parent
        if take_profit_price is None and stop_loss_price is None:
            parent_order.transmit = True
            self.ib.placeOrder(self.active_contract, parent_order)
        
        # Place all child orders
        for order in bracket_orders[1:]:  # Skip parent which was already placed
            trade = self.ib.placeOrder(self.active_contract, order)
            trade.statusEvent += self.update_order_status
        
        # Track the parent order
        self.active_order = parent_order
        self.active_orders[parent_id] = {
            'action': action_text,
            'quantity': quantity,
            'price': 0,  # Market order
            'time': datetime.now(),
            'type': 'parent',
            'children': [order.orderId for order in bracket_orders[1:]]
        }
        self.order_status[parent_id] = "Submitted"

    def fetch_historical_bars(self, days_back=5):
        """
        Fetch historical bars for the current contract.
        
        Args:
            days_back: Number of days to look back for historical data
            
        Returns:
            bool: Whether fetching was successful
        """
        if self.active_contract is None:
            logger.error("No active contract. Cannot fetch historical data.")
            return False
        
        logger.info(f"Fetching historical data for {self.active_contract.symbol} going back {days_back} days")
        
        # Calculate the lookback period - increased for more reliable indicator calculation
        end_datetime = datetime.now()
        start_datetime = end_datetime - timedelta(days=days_back)
        
        # Request data with 5-minute bars (consistent with training)
        try:
            bars = self.ib.reqHistoricalData(
                self.active_contract,
                endDateTime='',  # '' for latest data
                durationStr=f'{days_back} D',
                barSizeSetting='5 mins',  # Match the interval used in training
                whatToShow='TRADES',
                useRTH=True,  # Only regular trading hours
                formatDate=1   # Use integer format for dates
            )
            
            if not bars:
                logger.error("No historical bars returned")
                return False
            
            logger.info(f"Received {len(bars)} historical bars")
            
            # Clear existing bar history
            self.bar_history = []
            
            # Convert IB bars to standard format
            for bar in bars:
                standard_bar = {
                    'time': bar.date,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': float(bar.volume)
                }
                
                # Add to bar history
                self.bar_history.append(standard_bar)
            
            logger.info(f"Processed {len(self.bar_history)} historical bars")
            
            # Ensure we have enough bars for indicator calculation
            if len(self.bar_history) < self.min_bars_needed:
                logger.warning(f"Only got {len(self.bar_history)} bars, which is less than the minimum needed ({self.min_bars_needed})")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return False

def onBar(bars, hasNewBar):
    """
    Callback function triggered on each new real-time bar.
    'bars' is a list of bars, and 'hasNewBar' is a boolean indicating if there's a new bar.
    """
    global bar_buffer, bar_buckets, model_trader, last_data_timestamp, is_data_flowing
    
    # Update heartbeat timestamp
    last_data_timestamp = datetime.now()
    is_data_flowing = True
    
    if not bars or len(bars) == 0:
        return
    
    # Get the latest bar from the list
    latest_bar = bars[-1]
    
    # Print immediate feedback for each new bar
    try:
        logger.info(f"5-sec Bar received: Time={latest_bar.time} Close={latest_bar.close} Volume={latest_bar.volume}")
    except Exception as e:
        logger.warning(f"Error displaying bar info: {e}")
    
    # Only append if it's a new bar
    if hasNewBar:
        try:
            # Keep legacy buffer for backward compatibility
            bar_buffer.append(latest_bar)
            if len(bar_buffer) > 300:  # Prevent excessive memory usage
                bar_buffer = bar_buffer[-300:]
            
            # Add the bar to its appropriate 5-minute bucket
            # Handle the case where latest_bar.time is already a datetime object
            try:
                if isinstance(latest_bar.time, datetime):
                    bar_time = latest_bar.time
                else:
                    # If it's a timestamp or string, convert accordingly
                    try:
                        bar_time = datetime.fromtimestamp(latest_bar.time)
                    except (TypeError, ValueError):
                        # If conversion fails, use current time as fallback
                        logger.warning(f"Failed to parse bar time: {latest_bar.time}, using current time")
                        bar_time = datetime.now()
                
                interval_key = get_interval_key(bar_time)
                bar_buckets[interval_key].append(latest_bar)
                logger.info(f"Bar added to interval {interval_key} (now contains {len(bar_buckets[interval_key])} bars)")
            except Exception as e:
                logger.error(f"Error processing bar time: {e}")
                # Use current time as a fallback for interval
                interval_key = get_interval_key(datetime.now())
                bar_buckets[interval_key].append(latest_bar)
                logger.warning(f"Using current time for interval {interval_key} due to error")
            
            # Attempt to synchronize and process complete bars
            five_min_bar = None
            try:
                five_min_bar = synchronize_bars()
            except Exception as e:
                logger.error(f"Error in synchronize_bars: {e}")
            
            # Process the synchronized bar if available
            if five_min_bar is not None:
                logger.info("Processing synchronized 5-minute bar:")
                logger.info(f"Time: {five_min_bar.get('time', 'Unknown')} | Open: {five_min_bar['open']} | High: {five_min_bar['high']} | Low: {five_min_bar['low']} | Close: {five_min_bar['close']} | Volume: {five_min_bar['volume']}")
                
                # Use the model trader to make a prediction and execute a trade
                if model_trader is not None:
                    try:
                        # Force position verification before making trading decisions
                        model_trader.verify_position(force=True)
                        
                        # Preprocess the bar for model input
                        observation = model_trader.preprocess_bar(five_min_bar)
                        
                        # If we have enough data to make a prediction
                        if observation is not None:
                            # Get model prediction
                            action = model_trader.get_prediction(observation)
                            logger.info(f"Model prediction: {action} (0=long, 1=short, 2=hold)")
                            
                            # Execute trade based on prediction
                            model_trader.execute_trade(action, five_min_bar)
                            
                            # Verify position after trade execution
                            logger.info("Verifying position after trade execution...")
                            model_trader.verify_position(force=True)
                    except Exception as e:
                        logger.error(f"Error in model prediction or trading: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
            
            # For immediate feedback, also process every N bars regardless of time boundaries
            # This ensures we see activity right away while still maintaining synchronized trading
            legacy_processing_frequency = 12  # Process every 12 bars (approximately 1 minute)
            if len(bar_buffer) % legacy_processing_frequency == 0:
                logger.info(f"Interim bar processing: {len(bar_buffer)} bars collected")
                
                # Only create interim bars if we haven't already processed a synchronized bar in this cycle
                if five_min_bar is None and len(bar_buffer) >= legacy_processing_frequency:
                    try:
                        # Use the last N bars for interim processing
                        interim_bars = bar_buffer[-legacy_processing_frequency:]
                        five_min_bar_interim = aggregate_bars(interim_bars)
                        
                        if five_min_bar_interim and model_trader is not None:
                            logger.info(f"Created interim bar for immediate feedback - Close: {five_min_bar_interim['close']}")
                    except Exception as e:
                        logger.error(f"Error creating interim bar: {e}")
            
            # Legacy approach (keep for fallback in case synchronization fails)
            # When 60 bars are accumulated, aggregate them into one 5-minute bar
            if len(bar_buffer) == 60:
                try:
                    logger.info("Using legacy approach to create 5-minute bar (60 bars accumulated)")
                    five_min_bar_legacy = aggregate_bars(bar_buffer)
                    
                    # Only use this if we haven't already processed a synchronized bar
                    if five_min_bar is None and model_trader is not None:
                        # Force position verification before making trading decisions
                        model_trader.verify_position(force=True)
                        
                        # Preprocess the bar for model input
                        observation = model_trader.preprocess_bar(five_min_bar_legacy)
                        
                        # If we have enough data to make a prediction
                        if observation is not None:
                            # Get model prediction
                            action = model_trader.get_prediction(observation)
                            logger.info(f"Model prediction (legacy): {action} (0=long, 1=short, 2=hold)")
                            
                            # Execute trade based on prediction
                            model_trader.execute_trade(action, five_min_bar_legacy)
                            
                            # Verify position after trade execution
                            logger.info("Verifying position after trade execution...")
                            model_trader.verify_position(force=True)
                    
                    # Reset the buffer for the next aggregation
                    bar_buffer = []
                except Exception as e:
                    logger.error(f"Error in legacy bar processing: {e}")
                    
        except Exception as e:
            logger.error(f"Unexpected error in onBar: {e}")
            import traceback
            logger.error(traceback.format_exc())

def heartbeat_monitor():
    """Monitor for data flow and connection health."""
    global last_data_timestamp, is_data_flowing, reconnection_attempts, model_trader
    
    while True:
        time.sleep(5)  # Check every 5 seconds
        
        # Check time since last data
        now = datetime.now()
        time_since_last_data = (now - last_data_timestamp).total_seconds()
        
        # Try to synchronize bars every 5 seconds
        # This ensures we catch any completed 5-minute intervals even if data flow is sparse
        if is_data_flowing and model_trader:
            five_min_bar = synchronize_bars()
            if five_min_bar is not None:
                # We have a valid synchronized bar to process
                logger.info("Heartbeat monitor found a synchronized bar to process")
                
                # Force position verification before making trading decisions
                model_trader.verify_position(force=True)
                
                # Preprocess the bar for model input
                observation = model_trader.preprocess_bar(five_min_bar)
                
                # If we have enough data to make a prediction
                if observation is not None:
                    # Get model prediction
                    action = model_trader.get_prediction(observation)
                    logger.info(f"Model prediction (heartbeat): {action} (0=long, 1=short, 2=hold)")
                    
                    # Execute trade based on prediction
                    model_trader.execute_trade(action, five_min_bar)
        
        # Log heartbeat status periodically
        if time_since_last_data > heartbeat_interval:
            if is_data_flowing:
                logger.warning(f"No data received for {time_since_last_data:.1f} seconds!")
                is_data_flowing = False
            
            # If no data for too long, attempt reconnection
            if time_since_last_data > heartbeat_interval * 2 and reconnection_attempts < max_reconnection_attempts:
                logger.error(f"Connection may be lost. Attempting reconnection ({reconnection_attempts + 1}/{max_reconnection_attempts})...")
                reconnect_to_ib()
        else:
            # If we're getting data and previously flagged as not flowing, log recovery
            if not is_data_flowing:
                logger.info("Data flow resumed!")
                is_data_flowing = True
                reconnection_attempts = 0  # Reset counter when data flows again
        
        # Verify positions regularly
        if model_trader:
            model_trader.verify_position()

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

if __name__ == '__main__':
    # Check if the model exists
    model_path = "best_model"
    model_zip_path = "best_model.zip"
    
    if not os.path.exists(model_path) and os.path.exists(model_zip_path):
        model_path = model_zip_path
        print(f"Using zipped model at {model_path}")
    elif not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path} or {model_zip_path}")
        print("Please train a model first using train.py")
        sys.exit(1)
    
    ib = IB()
    
    # Connect to IB Gateway/TWS; update host/port/clientId if needed.
    try:
        ib.connect('127.0.0.1', 7496, clientId=1)
        print("Connected")
    except Exception as e:
        print(f"Error connecting to IB: {e}")
        sys.exit(1)
    
    # Define the generic NQ futures contract.
    nq_contract_generic = Future('NQ', exchange='CME')
    contracts = ib.reqContractDetails(nq_contract_generic)
    if not contracts:
        print("No contract details found for NQ")
        ib.disconnect()
        exit(1)
    
    # Select the most recent (front-month) contract based on expiration.
    active_contract = get_most_recent_contract(contracts)
    print(f"Using contract: {active_contract}")
    
    # Initialize the model trader
    model_trader = ModelTrader(ib, model_path=model_path, use_risk_management=True)
    
    # Load the model
    if not model_trader.load_model():
        print("Failed to load model. Exiting.")
        ib.disconnect()
        sys.exit(1)
    
    # Set the active contract
    model_trader.set_active_contract(active_contract)
    
    # Load previous state if exists
    state_loaded = model_trader.load_state()
    
    # Check if we have enough bar history. If not, fetch historical data
    if len(model_trader.bar_history) < model_trader.min_bars_needed:
        print(f"Insufficient bar history ({len(model_trader.bar_history)}/{model_trader.min_bars_needed}). Fetching historical data...")
        historical_data_loaded = model_trader.fetch_historical_bars(days_back=5)
        if historical_data_loaded:
            print(f"Successfully loaded historical data. Bar history now contains {len(model_trader.bar_history)} bars.")
        else:
            print("Failed to load sufficient historical data. The system will wait for enough real-time bars before trading.")
    else:
        print(f"Loaded {len(model_trader.bar_history)} bars from saved state")
    
    # Start the heartbeat monitoring thread
    heartbeat_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
    heartbeat_thread.start()
    print("Heartbeat monitoring started")
    
    # Request real-time bars using the active contract.
    try:
        bars = ib.reqRealTimeBars(active_contract, barSize=5, whatToShow='TRADES', useRTH=False)
        
        # Register to receive order status updates for all orders
        ib.connectedEvent += lambda: ib.reqAllOpenOrders()
        
        # Attach the onBar event handler to the updateEvent of the bars list.
        bars.updateEvent += onBar
    
            
    except Exception as e:
        print(f"Error requesting real-time data: {e}")
        print("Your account may not have market data permissions for CME futures.")
        print("Attempting to use delayed market data instead...")
        
        try:
            # Request delayed market data.
            ib.reqMarketDataType(3)  # 3 = Delayed data.
            ticker = ib.reqMktData(active_contract)
            util.sleep(5)  # Give it time to receive some data.
            
            # Verify we have a connection with data flowing
            if not is_data_flowing:
                raise Exception("Not receiving market data after 5 seconds")
                
            ib.sleep(60*60)  # Run for an hour or until interrupted.
            
        except Exception as e2:
            print(f"Error with delayed data as well: {e2}")
            print("Please check your IB account market data permissions.")
            ib.disconnect()
            sys.exit(1)
    
    # Start the IB event loop to begin receiving real-time data.
    print("Starting IB event loop. Data should begin flowing shortly.")
    print("If you don't see any data within 10-15 seconds, make sure your TWS/IB Gateway is running and properly connected.")
    
    try:
        ib.run()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected, saving state and shutting down...")
        if model_trader:
            model_trader.save_state()
            print(f"State saved with {len(model_trader.bar_history)} historical bars")
        ib.disconnect()
    except Exception as e:
        print(f"Error in main loop: {e}")
        if model_trader:
            model_trader.save_state()
            print(f"State saved with {len(model_trader.bar_history)} historical bars")
        ib.disconnect()
