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
        self.use_risk_management = use_risk_management
        
        # Store bars for feature calculation
        self.bar_history = []
        self.min_bars_needed = 20  # Minimum number of bars needed for feature calculation
        
        # Risk management settings from config
        self.risk_config = config.get("risk_management", {})
        self.stop_loss_pct = None
        self.take_profit_pct = None
        self.trailing_stop_pct = None
        self.position_size = 1
        
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
        """Initialize risk management parameters from config."""
        risk_enabled = self.risk_config.get("enabled", True)
        
        if risk_enabled:
            # Stop loss configuration
            stop_loss_config = self.risk_config.get("stop_loss", {})
            if stop_loss_config.get("enabled", False):
                self.stop_loss_pct = stop_loss_config.get("percentage", 1.0)
            
            # Take profit configuration
            take_profit_config = self.risk_config.get("take_profit", {})
            if take_profit_config.get("enabled", False):
                self.take_profit_pct = take_profit_config.get("percentage", 2.0)
            
            # Trailing stop configuration
            trailing_stop_config = self.risk_config.get("trailing_stop", {})
            if trailing_stop_config.get("enabled", False):
                self.trailing_stop_pct = trailing_stop_config.get("percentage", 0.5)
            
            # Position sizing configuration
            position_sizing_config = self.risk_config.get("position_sizing", {})
            if position_sizing_config.get("enabled", False):
                self.position_size = position_sizing_config.get("size_multiplier", 1.0)
    
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
        
        # TODO: Add calculation of other technical indicators here if needed
        # For now, we'll just use the normalized close price as our only feature
        
        # Create observation array
        # Format: [close_norm, indicator1, indicator2, ..., position]
        obs = np.array([close_norm, float(self.current_position)], dtype=np.float32)
        
        return obs
    
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
    global bar_buffer, model_trader
    
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
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first using train.py")
        sys.exit(1)
    
    ib = IB()
    
    # Connect to IB Gateway/TWS; update host/port/clientId if needed.
    ib.connect('127.0.0.1', 7496, clientId=1)
    print("Connected")
    
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
    
    # Request real-time bars using the active contract.
    try:
        bars = ib.reqRealTimeBars(active_contract, barSize=5, whatToShow='TRADES', useRTH=False)
        #print(f"Bars object type: {type(bars)}")
        
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
            ib.sleep(60*60)  # Run for an hour or until interrupted.
            
        except Exception as e2:
            print(f"Error with delayed data as well: {e2}")
            print("Please check your IB account market data permissions.")
            ib.disconnect()
            sys.exit(1)
    
    # Start the IB event loop to begin receiving real-time data.
    print("Starting IB event loop. Data should begin flowing shortly.")
    print("If you don't see any data within 10-15 seconds, make sure your TWS/IB Gateway is running and properly connected.")
    ib.run()
