from ib_insync import IB, Future, MarketOrder, util, Order, LimitOrder, StopOrder, BracketOrder
import datetime
import sys
import pandas as pd
import numpy as np
from decimal import Decimal
import time
import logging
from pathlib import Path
import os
import json
import threading
import pickle
from datetime import datetime, timedelta

# Import stable_baselines3 for loading the trained model
from stable_baselines3 import PPO

# Import money module and config for risk management
import money
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for bar buffer and model trader
bar_buffer = []
model_trader = None

# Heartbeat monitoring globals
last_data_timestamp = datetime.now()
heartbeat_interval = 30  # seconds
is_data_flowing = True
reconnection_attempts = 0
max_reconnection_attempts = 5
state_file = "trading_state.pkl"

# Function to aggregate a list of 60 5-second bars into one 5-minute bar.
def aggregate_bars(bars):
    """
    Given a list of 60 real-time bars (5-sec each), compute a 5-minute bar.
    """
    open_price = bars[0].open_
    high_price = max(bar.high for bar in bars)
    low_price = min(bar.low for bar in bars)
    close_price = bars[-1].close
    volume = sum(bar.volume for bar in bars)
    start_time = bars[0].time
    end_time = bars[-1].time
    return {
        'start': start_time,
        'end': end_time,
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price,
        'volume': volume
    }

class ModelTrader:
    def __init__(self, ib_instance, model_path="best_model", use_risk_management=True):
        """
        Initialize the model trader.
        
        Args:
            ib_instance: The IB class instance
            model_path: Path to the trained model
            use_risk_management: Whether to use risk management (take profit, stop loss)
        """
        self.ib = ib_instance
        self.model_path = model_path
        self.model = None
        self.active_contract = None
        self.current_position = 0  # 0=no position, 1=long, -1=short
        self.active_order = None
        self.active_orders = {}  # Dictionary to track all active orders
        self.order_status = {}   # Track order status
        self.use_risk_management = use_risk_management
        
        # Store bars for feature calculation
        self.bar_history = []
        self.min_bars_needed = 50  # Minimum number of bars needed for feature calculation
        
        # Risk management settings
        self.risk_config = config.get("risk_management", {})
        self.stop_loss_pct = None
        self.take_profit_pct = None
        self.trailing_stop_pct = None
        self.position_size = 1
        
        # Position tracking for verification
        self.expected_position = 0
        self.last_position_check = datetime.now()
        self.position_check_interval = 60  # seconds
        
        # Initialize risk parameters if risk management is enabled
        if self.use_risk_management:
            self._initialize_risk_parameters()
        
        # Trading indicators needed for model
        self.technical_indicators = []
        
        logger.info(f"ModelTrader initialized with model path: {model_path}")
        logger.info(f"Risk management: {use_risk_management}")
        if use_risk_management:
            logger.info(f"Stop Loss: {self.stop_loss_pct}%, Take Profit: {self.take_profit_pct}%, Trailing Stop: {self.trailing_stop_pct}%")
    
    def _initialize_risk_parameters(self):
        """Initialize risk management parameters from risk_params.json or config."""
        # First try to load from risk_params.json (created by train_live_model.py)
        try:
            if os.path.exists("risk_params.json"):
                logger.info("Loading risk parameters from risk_params.json")
                with open("risk_params.json", "r") as f:
                    risk_params = json.load(f)
                
                # Check if risk management is enabled globally
                risk_enabled = risk_params.get("enabled", False)
                if not risk_enabled:
                    logger.info("Risk management is disabled in risk_params.json")
                    return
                
                # Apply the risk parameters from the file
                logger.info(f"Risk parameters from file: {risk_params}")
                
                # Set parameters only if they are enabled (not None) in the file
                if risk_params.get("stop_loss") is not None:
                    self.stop_loss_pct = risk_params["stop_loss"]
                    logger.info(f"Using stop loss from file: {self.stop_loss_pct}%")
                
                if risk_params.get("take_profit") is not None:
                    self.take_profit_pct = risk_params["take_profit"]
                    logger.info(f"Using take profit from file: {self.take_profit_pct}%")
                
                if risk_params.get("trailing_stop") is not None:
                    self.trailing_stop_pct = risk_params["trailing_stop"]
                    logger.info(f"Using trailing stop from file: {self.trailing_stop_pct}%")
                
                if risk_params.get("position_size") is not None:
                    self.position_size = risk_params["position_size"]
                    logger.info(f"Using position size from file: {self.position_size}")
                
                return
        except Exception as e:
            logger.warning(f"Error loading risk_params.json: {e}. Falling back to config.yaml.")
        
        # Fall back to config.yaml if risk_params.json is not available or has an error
        logger.info("Using risk parameters from config.yaml")
        risk_enabled = self.risk_config.get("enabled", False)
        
        if not risk_enabled:
            logger.info("Risk management is disabled in config.yaml")
            return
        
        # Stop loss configuration
        stop_loss_config = self.risk_config.get("stop_loss", {})
        if stop_loss_config.get("enabled", False):
            self.stop_loss_pct = stop_loss_config.get("percentage", 1.0)
            logger.info(f"Stop loss enabled from config: {self.stop_loss_pct}%")
        else:
            logger.info("Stop loss is disabled in config")
        
        # Take profit configuration
        take_profit_config = self.risk_config.get("take_profit", {})
        if take_profit_config.get("enabled", False):
            self.take_profit_pct = take_profit_config.get("percentage", 2.0)
            logger.info(f"Take profit enabled from config: {self.take_profit_pct}%")
        else:
            logger.info("Take profit is disabled in config")
        
        # Trailing stop configuration
        trailing_stop_config = self.risk_config.get("trailing_stop", {})
        if trailing_stop_config.get("enabled", False):
            self.trailing_stop_pct = trailing_stop_config.get("percentage", 0.5)
            logger.info(f"Trailing stop enabled from config: {self.trailing_stop_pct}%")
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
    
    def load_model(self):
        """Load the trained model from the specified path."""
        logger.info(f"Loading model from {self.model_path}")
        try:
            self.model = PPO.load(self.model_path)
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
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
    
    def verify_position(self):
        """Verify that the internal position matches the actual position at IB."""
        now = datetime.now()
        
        # Only check positions periodically
        if (now - self.last_position_check).total_seconds() < self.position_check_interval:
            return
            
        self.last_position_check = now
        
        try:
            # Get current positions from IB
            positions = self.ib.positions()
            
            # Find our contract in the positions
            actual_position = 0
            for position in positions:
                if position.contract.symbol == self.active_contract.symbol and position.contract.exchange == self.active_contract.exchange:
                    actual_position = position.position
                    break
            
            # Check if our internal state matches the actual position
            if self.current_position != actual_position:
                logger.warning(f"Position mismatch: Internal={self.current_position}, Actual={actual_position}")
                
                # Reconcile by updating our internal state
                logger.info(f"Reconciling position: updating internal position to {actual_position}")
                self.current_position = actual_position
                self.expected_position = actual_position
                
                # Save the reconciled state
                self.save_state()
            else:
                logger.info(f"Position verification successful: {self.current_position}")
                
        except Exception as e:
            logger.error(f"Error verifying position: {e}")
    
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
        if trade.order.orderId not in self.order_status:
            self.order_status[trade.order.orderId] = trade.orderStatus.status
            logger.info(f"New order status: ID={trade.order.orderId}, Status={trade.orderStatus.status}")
        elif self.order_status[trade.order.orderId] != trade.orderStatus.status:
            self.order_status[trade.order.orderId] = trade.orderStatus.status
            logger.info(f"Order status updated: ID={trade.order.orderId}, Status={trade.orderStatus.status}")
            
            # If the order is filled, update our position
            if trade.orderStatus.status == "Filled":
                # Determine direction from the order
                if trade.order.action == "BUY":
                    position_change = trade.order.totalQuantity
                else:  # SELL
                    position_change = -trade.order.totalQuantity
                    
                # Update expected position
                self.expected_position += position_change
                logger.info(f"Order filled: Updating expected position to {self.expected_position}")
                
                # Save state after position change
                self.save_state()
    
    def preprocess_bar(self, bar):
        """
        Preprocess a single bar into features expected by the model.
        
        Args:
            bar: A dictionary containing OHLCV data
            
        Returns:
            np.ndarray: Observation array for the model
        """
        # Add bar to history
        self.bar_history.append(bar)
        
        # If we don't have enough bars yet, return None
        if len(self.bar_history) < self.min_bars_needed:
            logger.info(f"Not enough bars yet. Have {len(self.bar_history)}/{self.min_bars_needed}")
            return None
        
        # Keep only the most recent bars
        if len(self.bar_history) > 100:
            self.bar_history = self.bar_history[-100:]
        
        # Create a DataFrame from bar history
        df = pd.DataFrame(self.bar_history)
        
        # Calculate normalized close price (between 0 and 1)
        # Get min and max close price from the history
        min_close = df['close'].min()
        max_close = df['close'].max()
        
        # Handle case where min and max are the same
        if max_close == min_close:
            close_norm = 0.5
        else:
            close_norm = (bar['close'] - min_close) / (max_close - min_close)
        
        # Calculate technical indicators based on config

        # 1. RSI (Relative Strength Index)
        rsi = self._calculate_rsi(df, length=14)
        rsi = rsi / 100.0  # Normalize to [0, 1]
        
        # 2. CCI (Commodity Channel Index)
        cci = self._calculate_cci(df, length=20)
        # Normalize CCI
        max_abs_cci = max(abs(cci.max() if not np.isnan(cci.max()) else 1), 
                          abs(cci.min() if not np.isnan(cci.min()) else 1))
        if max_abs_cci > 0:
            cci = cci / max_abs_cci
        else:
            cci = 0
            
        # 3. MACD (Moving Average Convergence Divergence)
        macd, macd_signal, macd_hist = self._calculate_macd(df, fast=12, slow=26, signal=9)
        # Normalize MACD values
        max_val = max(abs(macd.max() if not np.isnan(macd.max()) else 1), 
                      abs(macd.min() if not np.isnan(macd.min()) else 1),
                      abs(macd_signal.max() if not np.isnan(macd_signal.max()) else 1), 
                      abs(macd_signal.min() if not np.isnan(macd_signal.min()) else 1),
                      abs(macd_hist.max() if not np.isnan(macd_hist.max()) else 1), 
                      abs(macd_hist.min() if not np.isnan(macd_hist.min()) else 1))
        if max_val > 0:
            macd = macd / max_val
            macd_signal = macd_signal / max_val
            macd_hist = macd_hist / max_val
        
        # 4. ATR (Average True Range)
        atr = self._calculate_atr(df, length=14)
        # Normalize ATR by close price
        atr = atr / df['close'].iloc[-1]
        
        # Create observation array based on environment.py structure
        # Format: [close_norm, technical_indicators..., position]
        # The most recent values from each indicator
        obs = np.array([
            close_norm,                  # Normalized close price
            rsi.iloc[-1],                # RSI
            cci.iloc[-1],                # CCI
            macd.iloc[-1],               # MACD
            macd_signal.iloc[-1],        # MACD Signal
            macd_hist.iloc[-1],          # MACD Histogram
            atr.iloc[-1],                # ATR
            float(self.current_position)  # Current position
        ], dtype=np.float32)
        
        # Clip observation values to [-1, 1] range except close_norm [0, 1]
        # First element (close_norm) should be clipped to [0, 1]
        obs[0] = np.clip(obs[0], 0, 1)
        # Other elements should be clipped to [-1, 1]
        obs[1:] = np.clip(obs[1:], -1, 1)
        
        return obs
    
    def _calculate_rsi(self, df, length=14):
        """Calculate RSI indicator"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=length).mean()
        avg_loss = loss.rolling(window=length).mean()
        
        rs = avg_gain / avg_loss.where(avg_loss != 0, 1)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_cci(self, df, length=20):
        """Calculate CCI indicator"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_sma = tp.rolling(window=length).mean()
        md = tp.rolling(window=length).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - tp_sma) / (0.015 * md)
        return cci.fillna(0)
    
    def _calculate_macd(self, df, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        close = df['close']
        exp1 = close.ewm(span=fast, adjust=False).mean()
        exp2 = close.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd.fillna(0), signal_line.fillna(0), histogram.fillna(0)
    
    def _calculate_atr(self, df, length=14):
        """Calculate ATR indicator"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Get previous close
        prev_close = close.shift(1)
        
        # Calculate True Range
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=length).mean()
        return atr.fillna(0)
    
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
        
        # Verify position before trading
        self.verify_position()
        
        current_price = bar['close']
        logger.info(f"Current price: {current_price}, Action: {action}, Current position: {self.current_position}")
        
        # Cancel any existing orders
        if self.active_order:
            self.ib.cancelOrder(self.active_order)
            self.active_order = None
        
        # Long signal
        if action == 0 and self.current_position != 1:
            # Exit any existing short position
            if self.current_position == -1:
                self._exit_position()
            
            # Enter long position
            self._enter_position(1, current_price)
        
        # Short signal
        elif action == 1 and self.current_position != -1:
            # Exit any existing long position
            if self.current_position == 1:
                self._exit_position()
            
            # Enter short position
            self._enter_position(-1, current_price)
        
        # Hold signal - do nothing
        elif action == 2:
            logger.info("Hold signal - maintaining current position")
    
    def _enter_position(self, direction, price):
        """
        Enter a new position with optional stop loss and take profit.
        
        Args:
            direction: 1 for long, -1 for short
            price: Current price for order execution
        """
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
        
        # Update current position
        self.current_position = 0
    
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
        point_value = 20.0  # $20 per point for NQ futures
        
        # For bracket orders, we need price levels rather than percentages
        if self.take_profit_pct is not None:
            # Convert take profit percentage to points
            tp_points = (price * (self.take_profit_pct / 100)) / point_value
            
            # Calculate take profit price
            if direction == 1:  # Long position
                take_profit_price = price + tp_points
            else:  # Short position
                take_profit_price = price - tp_points
            
            logger.info(f"Take profit set at {take_profit_price} ({self.take_profit_pct}% = {tp_points} points)")
        else:
            take_profit_price = None
            logger.info("Take profit is disabled")
        
        if self.stop_loss_pct is not None:
            # Convert stop loss percentage to points
            sl_points = (price * (self.stop_loss_pct / 100)) / point_value
            
            # Calculate stop loss price
            if direction == 1:  # Long position
                stop_loss_price = price - sl_points
            else:  # Short position
                stop_loss_price = price + sl_points
            
            logger.info(f"Stop loss set at {stop_loss_price} ({self.stop_loss_pct}% = {sl_points} points)")
        else:
            stop_loss_price = None
            logger.info("Stop loss is disabled")
            
        # Check if trailing stop is enabled
        if self.trailing_stop_pct is not None:
            ts_points = (price * (self.trailing_stop_pct / 100)) / point_value
            logger.info(f"Trailing stop is enabled: {self.trailing_stop_pct}% = {ts_points} points")
        else:
            logger.info("Trailing stop is disabled")
        
        # Create parent order
        parent_order = Order()
        parent_order.action = action_text
        parent_order.orderType = "MKT"
        parent_order.totalQuantity = quantity
        parent_order.transmit = False
        
        # Create take profit order
        if take_profit_price is not None:
            take_profit_order = Order()
            take_profit_order.action = opposite_action
            take_profit_order.orderType = "LMT"
            take_profit_order.totalQuantity = quantity
            take_profit_order.lmtPrice = round(take_profit_price, 2)
            take_profit_order.parentId = parent_order.orderId
            take_profit_order.transmit = stop_loss_price is None  # Only transmit if no stop loss
        else:
            take_profit_order = None
        
        # Create stop loss order
        if stop_loss_price is not None:
            stop_loss_order = Order()
            stop_loss_order.action = opposite_action
            stop_loss_order.orderType = "STP"
            stop_loss_order.totalQuantity = quantity
            stop_loss_order.auxPrice = round(stop_loss_price, 2)
            stop_loss_order.parentId = parent_order.orderId
            stop_loss_order.transmit = True  # Always transmit the last order
        else:
            stop_loss_order = None
        
        # Place the bracket order
        bracket_orders = [parent_order]
        if take_profit_order is not None:
            bracket_orders.append(take_profit_order)
        if stop_loss_order is not None:
            bracket_orders.append(stop_loss_order)
        
        for order in bracket_orders:
            trade = self.ib.placeOrder(self.active_contract, order)
        
        self.active_order = parent_order

def onBar(bars, hasNewBar):
    """
    Callback function triggered on each new real-time bar.
    'bars' is a list of bars, and 'hasNewBar' is a boolean indicating if there's a new bar.
    """
    global bar_buffer, model_trader, last_data_timestamp, is_data_flowing
    
    # Update heartbeat timestamp
    last_data_timestamp = datetime.now()
    is_data_flowing = True
    
    if not bars or len(bars) == 0:
        return
    
    # Get the latest bar from the list
    latest_bar = bars[-1]
    
    # Print information about the new 5-sec bar.
    print(f"5-sec Bar: Open={latest_bar.open_} High={latest_bar.high} Low={latest_bar.low} Close={latest_bar.close} Volume={latest_bar.volume}")
    
    # Only append if it's a new bar
    if hasNewBar:
        bar_buffer.append(latest_bar)
        
        # When 60 bars are accumulated, aggregate them into one 5-minute bar.
        if len(bar_buffer) == 60:
            five_min_bar = aggregate_bars(bar_buffer)
            print("Aggregated 5-Minute Bar:")
            print(five_min_bar)
            
            # Use the model trader to make a prediction and execute a trade
            if model_trader is not None:
                # Preprocess the bar for model input
                observation = model_trader.preprocess_bar(five_min_bar)
                
                # If we have enough data to make a prediction
                if observation is not None:
                    # Get model prediction
                    action = model_trader.get_prediction(observation)
                    print(f"Model prediction: {action} (0=long, 1=short, 2=hold)")
                    
                    # Execute trade based on prediction
                    model_trader.execute_trade(action, five_min_bar)
            
            # Reset the buffer for the next aggregation.
            bar_buffer = []
        else:
            print(f"Bar buffer: {len(bar_buffer)}/60 bars collected")

def heartbeat_monitor():
    """Monitor for data flow and connection health."""
    global last_data_timestamp, is_data_flowing, reconnection_attempts, model_trader
    
    while True:
        time.sleep(5)  # Check every 5 seconds
        
        # Check time since last data
        now = datetime.now()
        time_since_last_data = (now - last_data_timestamp).total_seconds()
        
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
    
    # Load previous state if exists
    model_trader.load_state()
    
    # Set the active contract
    model_trader.set_active_contract(active_contract)
    
    # Start the heartbeat monitoring thread
    heartbeat_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
    heartbeat_thread.start()
    print("Heartbeat monitoring started")
    
    # Request real-time bars using the active contract.
    try:
        bars = ib.reqRealTimeBars(active_contract, barSize=5, whatToShow='TRADES', useRTH=False)
        
        # Register to receive order status updates for all orders
        ib.connectHandler += lambda: ib.reqAllOpenOrders()
        
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
        ib.disconnect()
    except Exception as e:
        print(f"Error in main loop: {e}")
        if model_trader:
            model_trader.save_state()
        ib.disconnect()
