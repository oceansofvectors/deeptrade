import os
import json
import pickle
import time
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz
from stable_baselines3 import PPO
from ib_insync import MarketOrder, Order, LimitOrder, StopOrder, BracketOrder # Future, Contract may not be needed directly if ib_instance handles it

from config import config
from get_data import process_technical_indicators # ensure_numeric is a dependency of this
from normalization import load_scaler, normalize_data, get_standardized_column_names

logger = logging.getLogger(__name__)

class ModelTrader:
    def __init__(self, ib_instance, model_path, state_file_path, use_risk_management=True): # Added state_file_path
        """
        Initialize the ModelTrader with an Interactive Brokers instance and model.
        
        Args:
            ib_instance: An instance of the Interactive Brokers client
            model_path: Path to the trained model folder
            state_file_path: Path to the state file
            use_risk_management: Whether to use risk management
        """
        logger.info(f"Initializing ModelTrader with model at {model_path}")
        
        self.ib = ib_instance
        self.model_path = model_path
        self.state_file_path = state_file_path # Store state_file_path
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
                            logger.error("Please ensure these are in your enabled indicators list")
                            
                            # Store for reference
                            self.missing_indicators = missing_from_current
                        
                        if extra_in_current:
                            logger.error(f"FOUND EXTRA INDICATORS not in training file {file_path}:")
                            logger.error(f"Extra: {extra_in_current}")
                            logger.error("These should not be in your enabled indicators list")
                            
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
            with open(self.state_file_path, "wb") as f: # Use self.state_file_path
                pickle.dump(state, f)
                
            logger.info(f"State saved successfully with {len(self.bar_history)} bars to {self.state_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return False
    
    def load_state(self):
        """Load the trading state from file if it exists."""
        if not os.path.exists(self.state_file_path): # Use self.state_file_path
            logger.info(f"No saved state found at {self.state_file_path}")
            return False
        
        try:
            with open(self.state_file_path, 'rb') as f: # Use self.state_file_path
                state = pickle.load(f)
                
            self.current_position = state.get('current_position', 0)
            self.expected_position = state.get('expected_position', 0)
            self.active_orders = state.get('active_orders', {})
            self.bar_history = state.get('bar_history', [])
            
            logger.info(f"Trading state loaded successfully from {self.state_file_path}: position={self.current_position}")
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
            if self.contract: # Add check for self.contract
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
            if self.contract: # Add check for self.contract
                for position in positions:
                    # More detailed logging for each position being checked
                    logger.info(f"Checking position: Symbol={position.contract.symbol}, Exchange={position.contract.exchange}, SecType={position.contract.secType}")
                    
                    # Try more flexible matching
                    if (position.contract.symbol == self.contract.symbol and 
                        position.contract.secType == self.contract.secType):
                        logger.info(f"Found matching position: {position.position} contracts")
                        actual_position = position.position
                        break
            else: # If no self.contract, actual_position remains 0
                logger.warning("Cannot verify position as no active contract is set in ModelTrader.")

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
        Update internal 5-minute bar history, (re)calculate all enabled indicators
        for the full window, normalise, and return the **latest** normalised row.

        Steps replicate the training pipeline:
            1. Append bar to history
            2. Re-calculate technical indicators (process_technical_indicators)
            3. Add day-of-week + minutes-since-open features
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

        # 3️⃣ Time-based features
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
                logger.info("Normalization scaler loaded successfully")
                
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
                        logger.info("Closing existing short position before going long")
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
                        
                        # Point value for MNQ futures ($2 per point)
                        point_value = 2.0
                        
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
                        logger.info("Closing existing long position before going short")
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
                        
                        # Point value for MNQ futures ($2 per point)
                        point_value = 2.0
                        
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
        if self.contract: # Check if contract is set
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
        else:
            logger.warning("Cannot cancel orders as no active contract is set in ModelTrader.")

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
            # Need to import Future from ib_insync if it's constructed here,
            # but self.contract is usually set by set_active_contract from an existing Future object
            bars = self.ib.reqHistoricalData(
                self.contract, # Assumes self.contract is a valid IB Contract object
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
                    'time': bar.date, # bar.date is usually datetime object from IB
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
                logger.info("DEBUG: PRE-NORMALIZATION values for latest bar:")
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
                from sklearn.preprocessing import MinMaxScaler # Local import is fine here
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
                        logger.info("Restored supertrend original values (should be -1 or 1)")
                    if 'SUPERTREND' in df.columns:
                        norm_df['SUPERTREND'] = df['SUPERTREND']
                        logger.info("Restored SUPERTREND original values (should be -1 or 1)")
                    
                    # Fix PSAR_DIR values (also a directional indicator that should be -1 or 1)
                    if 'PSAR_DIR' in df.columns:
                        norm_df['PSAR_DIR'] = df['PSAR_DIR']
                        logger.info("Restored PSAR_DIR original values (should be -1 or 1)")
                    
                    # Handle SMA_NORM based on available columns (matching normalization.py logic)
                    sigmoid_k = 2.0
                    
                    # SMA_NORM from either SMA_20 or SMA
                    if 'SMA_20' in df.columns and 'SMA_NORM' not in norm_df.columns:
                        # Same logic as in normalization.py
                        sma_mean = df['SMA_20'].mean()
                        sma_std = max(df['SMA_20'].std(), 1e-6)
                        norm_df['SMA_NORM'] = 2 / (1 + np.exp(-sigmoid_k * ((df['SMA_20'] - sma_mean) / sma_std))) - 1
                        logger.info("Created SMA_NORM from SMA_20 using sigmoid normalization")
                    elif 'SMA' in df.columns and 'SMA_NORM' not in norm_df.columns:
                        # Same logic as before
                        close_mean = df['close'].mean()
                        close_std = max(df['close'].std(), 1e-6)
                        norm_df['SMA_NORM'] = 2 / (1 + np.exp(-sigmoid_k * ((df['SMA'] - close_mean) / close_std))) - 1
                        logger.info("Created SMA_NORM from SMA using sigmoid normalization")
                    
                    # EMA_NORM from either EMA_20 or EMA
                    if 'EMA_20' in df.columns and 'EMA_NORM' not in norm_df.columns:
                        # Same logic as in normalization.py
                        ema_mean = df['EMA_20'].mean()
                        ema_std = max(df['EMA_20'].std(), 1e-6)
                        norm_df['EMA_NORM'] = 2 / (1 + np.exp(-sigmoid_k * ((df['EMA_20'] - ema_mean) / ema_std))) - 1
                        logger.info("Created EMA_NORM from EMA_20 using sigmoid normalization")
                    elif 'EMA' in df.columns and 'EMA_NORM' not in norm_df.columns:
                        # Same logic as before
                        close_mean = df['close'].mean()
                        close_std = max(df['close'].std(), 1e-6)
                        norm_df['EMA_NORM'] = 2 / (1 + np.exp(-sigmoid_k * ((df['EMA'] - close_mean) / close_std))) - 1
                        logger.info("Created EMA_NORM from EMA using sigmoid normalization")
                    
                    # VOLUME_MA from either VOLUME or VOLUME_NORM
                    if 'VOLUME' in df.columns and 'VOLUME_MA' not in norm_df.columns:
                        # Same logic as in normalization.py
                        vol_mean = df['VOLUME'].mean()
                        vol_std = max(df['VOLUME'].std(), 1e-6)
                        norm_df['VOLUME_MA'] = 2 / (1 + np.exp(-sigmoid_k * ((df['VOLUME'] - vol_mean) / vol_std))) - 1
                        logger.info("Created VOLUME_MA from VOLUME using sigmoid normalization")
                    elif 'VOLUME_NORM' in df.columns and 'VOLUME_MA' not in norm_df.columns:
                        # Previous logic - directly copy VOLUME_NORM to VOLUME_MA
                        norm_df['VOLUME_MA'] = norm_df['VOLUME_NORM']
                        logger.info("Set VOLUME_MA to match VOLUME_NORM value")
                
                # Log sample of normalized values for key features
                if norm_df is not None and not norm_df.empty:
                    last_norm_row = norm_df.iloc[-1]
                    logger.info("DEBUG: POST-NORMALIZATION values for latest bar:")
                    
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
                    # Ensure dt is timezone-aware before converting
                    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None: # If naive
                         dt_eastern = eastern.localize(dt) # Localize if naive (assuming it's local time to be made aware)
                    else: # If aware
                         dt_eastern = dt.astimezone(eastern) # Convert if already aware
                    
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
                # df.index might be timezone-aware (UTC from bar processing) or naive.
                # The original code had: dt.tz_localize(pytz.UTC).tz_convert(eastern) if dt.tzinfo else eastern.localize(dt)
                # This was problematic if dt was already localized.
                # The updated minutes_since_open handles this.
                mso_values = [minutes_since_open(idx.to_pydatetime()) for idx in df.index] # Convert Pandas Timestamp to python datetime

                # Create cyclical features (sin/cos) for better representation
                result_df['MSO_SIN'] = [np.sin(2 * np.pi * m / 390) for m in mso_values]
                result_df['MSO_COS'] = [np.cos(2 * np.pi * m / 390) for m in mso_values]
                logger.info("Added MSO_SIN and MSO_COS features based on config")
            
            return result_df
        except Exception as e:
            logger.error(f"Error adding time features: {e}")
            import traceback # Add import for traceback
            logger.error(traceback.format_exc())
            return df # Return original df on error

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
            import traceback # Add import for traceback
            logger.error(traceback.format_exc())
            return None 