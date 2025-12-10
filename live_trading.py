from ib_insync import IB, Future, util
import datetime
import sys
import pytz
import time
import logging
import os
import threading
from datetime import datetime, timedelta
import argparse

# Import utility functions
from trading.utils import get_most_recent_contract

# Import bar handling functions and variables
from trading.bar_handler import (
    aggregate_bars, get_interval_key, end_of_interval, synchronize_bars,
    bar_buckets, BARS_PER_FIVE_MIN, FIVE_SEC_PER_BAR, ROUND_TO, UTC as BH_UTC
)

# Import config and constants
from config import config
import constants

# Import ModelTrader class
from trading.model_trader import ModelTrader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File to store last trading day
LAST_TRADING_DAY_FILE = "last_trading_day.txt"

def save_last_trading_day(date):
    """Save the last trading day to a file."""
    try:
        with open(LAST_TRADING_DAY_FILE, 'w') as f:
            f.write(date.strftime('%Y-%m-%d'))
        logger.info(f"Saved last trading day: {date}")
    except Exception as e:
        logger.error(f"Error saving last trading day: {e}")

def load_last_trading_day():
    """Load the last trading day from file."""
    try:
        if os.path.exists(LAST_TRADING_DAY_FILE):
            with open(LAST_TRADING_DAY_FILE, 'r') as f:
                date_str = f.read().strip()
                return datetime.strptime(date_str, '%Y-%m-%d').date()
    except Exception as e:
        logger.error(f"Error loading last trading day: {e}")
    return None

# Global variables for model trader and execution state
model_trader = None
last_execution_time = None
last_data_timestamp = datetime.now()
is_data_flowing = False
reconnection_attempts = 0

# Daily PnL tracking
daily_pnl = 0.0
last_trading_day = None
daily_trading_stopped = False
realized_pnl = 0.0
unrealized_pnl = 0.0

# Heartbeat monitoring globals
heartbeat_interval = 60  # Seconds between heartbeat checks
state_file = "trader_state.pkl"
shutdown_requested = False  # Add this flag for coordinated shutdown

def check_daily_limits():
    """Check if daily PnL limits have been hit."""
    global daily_trading_stopped, daily_pnl, realized_pnl, unrealized_pnl, model_trader, shutdown_requested

    try:
        if (config["risk_management"]["enabled"] and 
            config["risk_management"]["daily_risk_limit"]["enabled"]):
            
            max_loss = config["risk_management"]["daily_risk_limit"]["max_daily_loss"]
            take_profit = config["risk_management"]["daily_risk_limit"]["daily_take_profit"]
            
            # Log limit check details
            logger.info(f"Checking limits - Max Loss: ${max_loss}, "
                      f"Take Profit: ${take_profit}, "
                      f"Current PnL: ${daily_pnl:.2f}, "
                      f"Trading Stopped: {daily_trading_stopped}")
            
            # Check if we've already hit the limit
            if daily_trading_stopped:
                logger.info(f"Daily trading already stopped. Current PnL: ${daily_pnl:.2f}")
                if not shutdown_requested:
                    initiate_shutdown("Daily trading stopped")
                return
            
            # Check for max loss - use abs() to handle negative values correctly
            if abs(daily_pnl) >= max_loss and daily_pnl < 0:
                logger.warning(f"!!! DAILY MAX LOSS TRIGGERED !!! "
                             f"Limit: ${max_loss}, Current Loss: ${abs(daily_pnl):.2f}")
                daily_trading_stopped = True
                # Close any open positions
                if model_trader:
                    logger.warning("Initiating emergency position closure due to max loss")
                    # Cancel all existing orders first
                    _cancel_all_orders_safe()
                    # Then close positions
                    model_trader.close_all_positions()
                    logger.warning("Emergency position closure completed")
                # Initiate shutdown
                initiate_shutdown("Daily max loss limit reached")
            elif daily_pnl >= take_profit:
                logger.warning(f"!!! DAILY TAKE PROFIT TRIGGERED !!! "
                             f"Target: ${take_profit}, Current Profit: ${daily_pnl:.2f}")
                daily_trading_stopped = True
                # Close any open positions
                if model_trader:
                    logger.warning("Initiating position closure due to take profit")
                    # Cancel all existing orders first
                    _cancel_all_orders_safe()
                    # Then close positions
                    model_trader.close_all_positions()
                    logger.warning("Take profit position closure completed")
                # Initiate shutdown
                initiate_shutdown("Daily take profit target reached")
    except Exception as e:
        logger.error(f"Error checking daily limits: {e}")
        import traceback
        logger.error(traceback.format_exc())

def _cancel_all_orders_safe():
    """
    Safely cancel all orders without trying to cancel order ID 0.
    """
    try:
        logger.info("Safely cancelling all orders")
        if not model_trader or not model_trader.ib:
            logger.warning("Cannot cancel orders - model trader or IB connection not available")
            return

        # Get all open trades
        open_trades = model_trader.ib.trades()
        cancelled = 0

        for trade in open_trades:
            if not trade.order or not trade.contract:
                continue
                
            # Skip orders with ID 0
            if trade.order.orderId == 0:
                continue
                
            try:
                logger.info(f"Cancelling order {trade.order.orderId}")
                model_trader.ib.cancelOrder(trade.order)
                cancelled += 1
            except Exception as e:
                logger.error(f"Error cancelling order {trade.order.orderId}: {e}")
                
        logger.info(f"Cancelled {cancelled} orders")
        
        # Clear internal order tracking
        model_trader.order_ids = []
        model_trader.active_orders = {}
        
    except Exception as e:
        logger.error(f"Error in _cancel_all_orders_safe: {e}")
        import traceback
        logger.error(traceback.format_exc())

def onPnL(pnl):
    """Handle position PnL updates."""
    global unrealized_pnl, daily_pnl, realized_pnl, daily_trading_stopped
    
    try:
        # Validate PnL values before updating
        if hasattr(pnl, 'dailyPnL') and isinstance(pnl.dailyPnL, (int, float)) and abs(pnl.dailyPnL) < 1e6:
            daily_pnl = pnl.dailyPnL
            # Immediately check limits after updating daily PnL
            if not daily_trading_stopped:
                check_daily_limits()
        if hasattr(pnl, 'unrealizedPnL') and isinstance(pnl.unrealizedPnL, (int, float)) and abs(pnl.unrealizedPnL) < 1e6:
            unrealized_pnl = pnl.unrealizedPnL
        if hasattr(pnl, 'realizedPnL') and isinstance(pnl.realizedPnL, (int, float)) and abs(pnl.realizedPnL) < 1e6:
            realized_pnl = pnl.realizedPnL
        
        logger.info(f"Account PnL Update - Daily PnL: ${daily_pnl:.2f}, "
                   f"Realized: ${realized_pnl:.2f}, "
                   f"Unrealized: ${unrealized_pnl:.2f}")
    except Exception as e:
        logger.error(f"Error in onPnL: {e}")
        import traceback
        logger.error(traceback.format_exc())

def onPnLSingle(pnl):
    """Handle single position PnL updates."""
    global unrealized_pnl, daily_pnl, realized_pnl, daily_trading_stopped
    
    try:
        # Only process if this is for our active contract
        if model_trader and model_trader.active_contract and hasattr(pnl, 'conId') and pnl.conId == model_trader.active_contract.conId:
            # Validate PnL values before updating
            if hasattr(pnl, 'unrealizedPnL') and isinstance(pnl.unrealizedPnL, (int, float)) and abs(pnl.unrealizedPnL) < 1e6:
                unrealized_pnl = pnl.unrealizedPnL
            if hasattr(pnl, 'realizedPnL') and isinstance(pnl.realizedPnL, (int, float)) and abs(pnl.realizedPnL) < 1e6:
                realized_pnl = pnl.realizedPnL
            if hasattr(pnl, 'dailyPnL') and isinstance(pnl.dailyPnL, (int, float)) and abs(pnl.dailyPnL) < 1e6:
                daily_pnl = pnl.dailyPnL
                # Immediately check limits after updating daily PnL
                if not daily_trading_stopped:
                    check_daily_limits()
            else:
                # Calculate daily PnL as sum of realized and unrealized
                daily_pnl = realized_pnl + unrealized_pnl
                # Check limits with calculated value
                if not daily_trading_stopped:
                    check_daily_limits()
            
            logger.info(f"Position PnL Update - Daily PnL: ${daily_pnl:.2f}, "
                       f"Realized: ${realized_pnl:.2f}, "
                       f"Unrealized: ${unrealized_pnl:.2f}")
    except Exception as e:
        logger.error(f"Error in onPnLSingle: {e}")
        import traceback
        logger.error(traceback.format_exc())

def onExecDetails(trade, fill):
    """Handle execution details updates."""
    global realized_pnl, daily_pnl
    if trade.contract == model_trader.active_contract:
        # Update realized PnL when trades are executed
        if hasattr(fill, 'realizedPNL'):
            realized_pnl = fill.realizedPNL
            daily_pnl = realized_pnl + unrealized_pnl
            logger.info(f"Trade Execution - Realized PnL: ${realized_pnl:.2f}, "
                       f"Total PnL: ${daily_pnl:.2f}")
            check_daily_limits()

def onBar(bars, hasNewBar):
    """
    Callback function for new bars from Interactive Brokers.
    This is called by IBApi when new bars arrive.
    
    Args:
        bars: RealTimeBarList object from IB API
        hasNewBar: Whether the update contains a new completed bar
    """
    global model_trader, last_execution_time, last_data_timestamp, is_data_flowing
    global daily_pnl, last_trading_day, daily_trading_stopped, realized_pnl, unrealized_pnl
    
    # Update heartbeat monitoring
    last_data_timestamp = datetime.now()
    is_data_flowing = True
    
    try:
        # Check if it's a new trading day
        current_day = datetime.now().date()
        stored_last_day = load_last_trading_day()
        
        if stored_last_day != current_day:
            # Reset daily tracking variables
            daily_pnl = 0.0
            realized_pnl = 0.0
            unrealized_pnl = 0.0
            daily_trading_stopped = False
            last_trading_day = current_day
            save_last_trading_day(current_day)
            logger.info(f"New trading day started: {current_day}")

            # Reset LSTM states for recurrent models at day boundary
            if model_trader and model_trader.is_recurrent_model:
                model_trader.reset_lstm_state()
                logger.info("Reset LSTM hidden states for new trading day")

        # Only process bar completion logic if we have a new bar
        if not hasNewBar:
            return
            
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
        
        # Skip trading logic if daily limits have been hit
        if daily_trading_stopped:
            logger.info(f"Daily trading stopped. Current PnL: ${daily_pnl:.2f}")
            return
            
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
            f"bucket {bucket_key} size={len(bar_buckets[bucket_key])} last_bar={bar_time} now={datetime.now(BH_UTC)}"
        )
        
        # Check if we have a complete 5-minute bar to process
        five_min_bar = synchronize_bars()
        
        if five_min_bar:
            # Process the 5-minute bar
            logger.info(f"Processing complete 5-minute bar: {five_min_bar}")
            
            # Check if enough time has passed since last execution
            now = datetime.now()
            if last_execution_time and now - last_execution_time < timedelta(seconds=constants.MIN_EXECUTION_INTERVAL):
                time_since_last = (now - last_execution_time).total_seconds()
                logger.info(f"Skipping execution - only {time_since_last:.1f} seconds since last execution (min {constants.MIN_EXECUTION_INTERVAL}s)")
                return
            
            if model_trader is None:
                logger.error("ModelTrader not initialized, cannot process bar.")
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
    global ib, model_trader, reconnection_attempts, contract, is_data_flowing, shutdown_requested
    
    logger.info("Starting heartbeat monitor thread")
    
    while not shutdown_requested:  # Add shutdown check
        try:
            current_time = datetime.now()
            
            # Check if trading has been stopped
            if daily_trading_stopped and not shutdown_requested:
                logger.info("Trading stopped detected in heartbeat monitor")
                initiate_shutdown("Trading stopped detected in heartbeat monitor")
                break
            
            # Check if data is still flowing
            if is_data_flowing:
                time_since_last = (current_time - last_data_timestamp).total_seconds()
                
                # If no data for over constants.DATA_FLOW_THRESHOLD seconds, try to reconnect
                if time_since_last > constants.DATA_FLOW_THRESHOLD:
                    logger.error(f"No data received for {time_since_last:.1f} seconds. Attempting reconnection.")
                    reconnected = reconnect_to_ib()
                    
                    if reconnected and contract:
                        if ib.isConnected():
                            try:
                                logger.info(f"Re-subscribing to real-time bars for {contract.localSymbol}")
                                bars_subscription = ib.reqRealTimeBars(contract, barSize=5, whatToShow='TRADES', useRTH=False)
                                bars_subscription.updateEvent += onBar
                                logger.info(f"Re-subscribed to real-time bars for {contract.localSymbol}")
                                is_data_flowing = True
                            except Exception as e:
                                logger.error(f"Error re-subscribing to market data: {e}")
                        else:
                            logger.error("IB is not connected after reconnection attempt, cannot re-subscribe.")
            
            # Sleep before next check
            time.sleep(15)
            
        except Exception as e:
            logger.error(f"Error in heartbeat monitor: {e}")
            import traceback
            logger.error(traceback.format_exc())
            time.sleep(15)
    
    logger.info("Heartbeat monitor thread stopping...")

def reconnect_to_ib():
    """Attempt to reconnect to IB and restore the trading state."""
    global ib, model_trader, reconnection_attempts
    
    reconnection_attempts += 1
    if reconnection_attempts > constants.MAX_RECONNECTION_ATTEMPTS:
        logger.error("Max reconnection attempts reached. Exiting.")
        # sys.exit(1) # Consider how to handle this; exiting thread might not stop main
        return False

    try:
        # Disconnect if currently connected
        if ib.isConnected():
            logger.info("Disconnecting from IB for reconnection attempt...")
            ib.disconnect()
            time.sleep(1) # Brief pause after disconnect
        
        # Wait before reconnecting
        logger.info(f"Attempting reconnection {reconnection_attempts}/{constants.MAX_RECONNECTION_ATTEMPTS}...")
        time.sleep(5)
        
        if not ib.isConnected():
            logger.info("Attempting to connect to IB...")
            ib.connect(config.get('ib_host', '127.0.0.1'), 
                       config.get('ib_port', 7496), # Default to live port if not specified, or make it rely on initial setup
                       clientId=config.get('ib_client_id', 1))
            if ib.isConnected():
                 logger.info("Reconnected to IB successfully.")
                 reconnection_attempts = 0 # Reset on success
            else:
                logger.error("Failed to reconnect to IB.")
                return False
        else:
            logger.info("IB is already connected. Skipping reconnection logic inside reconnect_to_ib.")

        # Save the current state before reinitializing - model_trader might be None if initial setup failed
        if model_trader:
            model_trader.save_state() # Uses self.state_file_path
        
        # Restore the trading state
        if model_trader:
            model_trader.load_state() # Uses self.state_file_path
            model_trader.verify_position(force=True)
        
        return True # Successfully reconnected (or was already connected) and attempted state restoration
    except Exception as e:
        logger.error(f"Reconnection attempt failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def _cancel_all_existing_orders():
    """
    Cancel ALL existing orders for the current contract.
    This is more robust than trying to track individual orders.
    """
    try:
        logger.info("Cancelling ALL existing orders to ensure clean slate")
        
        # Get all open trades from IB (trades contain both order and contract)
        open_trades = ib.trades()
        if not open_trades:
            logger.info("No open orders found to cancel")
            return
            
        # Count orders for our contract
        contract_orders = 0
        
        # Cancel all orders for our contract
        if model_trader and model_trader.active_contract:
            for trade in open_trades:
                # Check if this order is for our contract
                if (trade.contract.symbol == model_trader.active_contract.symbol and 
                    trade.contract.secType == model_trader.active_contract.secType and
                    trade.order.orderId != 0):  # Skip orders with ID 0
                    contract_orders += 1
                    try:
                        logger.info(f"Cancelling order: ID={trade.order.orderId}, Action={trade.order.action}, Type={trade.order.orderType}")
                        ib.cancelOrder(trade.order)
                    except Exception as e:
                        logger.error(f"Error cancelling order {trade.order.orderId}: {e}")
            logger.info(f"Cancelled {contract_orders} orders for {model_trader.active_contract.symbol}")
        else:
            logger.warning("Cannot cancel orders as no active contract is set in ModelTrader.")

        # Clear our internal order tracking
        if model_trader:
            model_trader.order_ids = []
            model_trader.active_orders = {}
    except Exception as e:
        logger.error(f"Error in _cancel_all_existing_orders: {e}")
        import traceback
        logger.error(traceback.format_exc())

def initiate_shutdown(reason: str):
    """
    Initiate a graceful shutdown of the trading system.
    
    Args:
        reason: The reason for shutdown
    """
    global shutdown_requested
    
    try:
        shutdown_requested = True
        logger.warning(f"Initiating trading system shutdown. Reason: {reason}")
        
        # Save final state
        if model_trader:
            logger.info("Saving final trading state...")
            model_trader.save_state()
        
        # Cancel all pending orders
        _cancel_all_orders_safe()
        
        # Disconnect from IB
        if ib and ib.isConnected():
            logger.info("Disconnecting from Interactive Brokers...")
            ib.disconnect()
        
        logger.info("=== Trading Session Summary ===")
        logger.info(f"Final Daily PnL: ${daily_pnl:.2f}")
        logger.info(f"Realized PnL: ${realized_pnl:.2f}")
        logger.info(f"Unrealized PnL: ${unrealized_pnl:.2f}")
        logger.info(f"Shutdown reason: {reason}")
        logger.info("==============================")
        
        # Exit the script
        logger.info("Exiting trading system...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live trading script for executing model predictions via IB")
    parser.add_argument("--model", type=str, default="best_model", help="Path to the model directory")
    parser.add_argument("--contract", type=str, default="MNQ", help="Contract symbol to trade")
    parser.add_argument("--paper", action="store_true", help="Connect to paper trading")
    parser.add_argument("--no_risk", action="store_true", help="Disable risk management")
    args = parser.parse_args()
    
    # Set up logging to file and console
    log_file = f"live_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Add handlers to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    # Log platform information
    logger.info(f"Starting live trading with model {args.model}")
    logger.info(f"Python version: {sys.version}")
    import platform
    logger.info(f"Operating system: {platform.system()} {platform.release()}")
    
    # Log detailed configuration for transparency
    logger.info("====== Configuration ======")
    logger.info(f"- Model path: {args.model}")
    logger.info(f"- Contract: {args.contract}")
    logger.info(f"- Paper trading: {args.paper}")
    logger.info(f"- Risk management: {not args.no_risk}")
    logger.info(f"- Transaction costs in config: {config.get('environment', {}).get('transaction_cost', 0.0)}")
    logger.info(f"- Position size in config: {config.get('environment', {}).get('position_size', 1)}")
    
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
    
    # Define contract here so heartbeat_monitor can access it if needed for re-subscription
    contract_obj = None # Changed name to avoid conflict with args.contract string
    
    # Try connecting to IB
    try:
        port = 7497 if args.paper else 7496
        client_id = 1 # Example client ID
        host = '127.0.0.1'
        logger.info(f"Connecting to IB ({'Paper' if args.paper else 'Live'}) on {host}:{port} with ClientID {client_id}")
        ib.connect(host, port, clientId=client_id)
        
        # Log connection status    
        logger.info(f"IB connection established: {ib.isConnected()}")
        
        # Create model trader instance (use the global variable)
        model_trader = ModelTrader(ib, 
                                 model_path=args.model, 
                                 state_file_path=state_file, # Pass state_file global
                                 use_risk_management=not args.no_risk)
        
        # Get the most liquid contract for the chosen symbol
        if args.contract == "MNQ":
            logger.info("Setting up MNQ Futures contract")
            contracts_details = ib.reqContractDetails(Future(symbol='MNQ', exchange='CME'))
            if contracts_details:
                contract_obj = get_most_recent_contract(contracts_details)
            else:
                logger.error("No contract details received for MNQ.")
                sys.exit(1)
        elif args.contract == "NQ":
            logger.info("Setting up NQ Futures contract")
            contracts_details = ib.reqContractDetails(Future(symbol='NQ', exchange='CME'))
            if contracts_details:
                contract_obj = get_most_recent_contract(contracts_details)
            else:
                logger.error("No contract details received for NQ.")
                sys.exit(1)
        elif args.contract == "ES":
            logger.info("Setting up ES Futures contract")
            contracts_details = ib.reqContractDetails(Future(symbol='ES', exchange='CME'))
            if contracts_details:
                contract_obj = get_most_recent_contract(contracts_details)
            else:
                logger.error("No contract details received for ES.")
                sys.exit(1)
        else:
            logger.error(f"Unsupported contract symbol: {args.contract}")
            sys.exit(1)
        
        if contract_obj:
            logger.info(f"Selected contract: {contract_obj.localSymbol}")
            model_trader.set_active_contract(contract_obj)
            # Make contract_obj accessible to heartbeat_monitor (it is already a global variable `contract` in that function)
            # To make it truly accessible, it should be a global in this module, or passed differently.
            # The heartbeat_monitor declares `global contract`. So we need a global `contract` here.
            # Assign to global `contract` for heartbeat_monitor
            globals()['contract'] = contract_obj
        else:
            logger.error(f"No valid contract found for {args.contract}!")
            sys.exit(1)
        
        # Fetch historical bars to initialize the model
        historical_data_success = model_trader.fetch_historical_bars(days_back=30)
        if not historical_data_success:
            logger.error("Failed to fetch historical data")
            # sys.exit(1) # Allow to continue if historical data fails but model might not work
        
        # Test loading the model to ensure it's working
        logger.info("Verifying model is loaded correctly")
        if model_trader.model is None: # model_trader.load_model() returns True/False, check self.model directly
            logger.error("Model failed to load! Check model path and integrity.")
            sys.exit(1)
        
        # Set up real-time bar subscription (5 second bars)
        logger.info(f"Setting up real-time bar subscription for {contract_obj.localSymbol}")
        bars_subscription = ib.reqRealTimeBars(contract_obj, 5, 'TRADES', False)
        bars_subscription.updateEvent += onBar
        
        # Set up PnL monitoring
        logger.info("Setting up PnL monitoring")
        ib.pnlEvent += onPnL
        ib.pnlSingleEvent += onPnLSingle
        ib.execDetailsEvent += onExecDetails
        
        # Request PnL updates
        if contract_obj:
            account = ib.managedAccounts()[0]  # Get the first managed account
            logger.info(f"Requesting PnL updates for account {account}")
            ib.reqPnL(account, '')  # Request account-wide PnL
            ib.reqPnLSingle(account, '', contract_obj.conId)  # Request PnL for specific contract
        
        # Log the bar aggregation setup for clarity
        logger.info("====== Bar Aggregation Setup ======")
        logger.info("- Raw bar interval: 5 seconds (from IB API)")
        logger.info("- Aggregated bar interval: 5 minutes")
        logger.info("- Aggregation function: aggregate_bars (from bar_handler)")
        logger.info("- Buffer: bar_buckets (from bar_handler, organized by 5-minute intervals)")
        logger.info("- Processing: synchronize_bars (from bar_handler, checks for completed intervals)")
        
        # Start heartbeat monitoring thread
        heartbeat_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
        heartbeat_thread.start()
        logger.info("Heartbeat monitoring started")
        
        # Load saved state if available (ModelTrader handles this in its init or via load_state method)
        # model_trader.load_state() # Already called in init and after reconnect attempts if needed
        
        # Add force reconnect timer
        ib.setTimeout(constants.IB_TIMEOUT)
        
        # Main loop - just keep the connection alive
        logger.info("Entering main loop, waiting for bar events and IB events...")
        ib.run()
        
    except ConnectionRefusedError:
        logger.error(f"IB connection refused. Ensure TWS/Gateway is running and API connections are enabled on port {port}.")
    except Exception as e:
        logger.error(f"Error in main thread: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Attempt to save state before exit
        if model_trader:
            logger.info("Attempting to save final state...")
            model_trader.save_state()
            
        # Ensure IB connection is closed
        if ib and ib.isConnected():
            logger.info("Disconnecting IB connection...")
            ib.disconnect()
            logger.info("IB connection closed")
        logger.info("Live trading system shut down.")

