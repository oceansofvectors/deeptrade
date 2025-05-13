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

# Import config
from config import config

# Import ModelTrader class
from trading.model_trader import ModelTrader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Global variables for model trader and execution state
model_trader = None
last_execution_time = None
last_data_timestamp = datetime.now()
is_data_flowing = False
reconnection_attempts = 0
MAX_RECONNECTION_ATTEMPTS = 3
DATA_FLOW_THRESHOLD = 60  # seconds
MIN_EXECUTION_INTERVAL = 10  # seconds

# Heartbeat monitoring globals
heartbeat_interval = 60  # Seconds between heartbeat checks
state_file = "trader_state.pkl"

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
        
    global model_trader, last_execution_time, last_data_timestamp, is_data_flowing
    
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
            f"bucket {bucket_key} size={len(bar_buckets[bucket_key])} last_bar={bar_time} now={datetime.now(BH_UTC)}"
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
    global ib, model_trader, reconnection_attempts, contract, is_data_flowing
    
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
                    # Call reconnect_to_ib which handles ib and model_trader globals
                    reconnected = reconnect_to_ib() 
                    
                    if reconnected and contract: # Check if reconnected and contract exists
                        # Re-subscribe to data - this needs access to the 'contract' object from main
                        if ib.isConnected():
                            try:
                                logger.info(f"Re-subscribing to real-time bars for {contract.localSymbol}")
                                bars_subscription = ib.reqRealTimeBars(contract, barSize=5, whatToShow='TRADES', useRTH=False)
                                bars_subscription.updateEvent += onBar # Re-attach event handler
                                logger.info(f"Re-subscribed to real-time bars for {contract.localSymbol}")
                                is_data_flowing = True # Reset flag optimistically
                            except Exception as e:
                                logger.error(f"Error re-subscribing to market data: {e}")
                        else:
                            logger.error("IB is not connected after reconnection attempt, cannot re-subscribe.")
            # else:
                # logger.debug(f"Data flowing, {time_since_last:.1f}s since last data.") # Optional: log if data is flowing
            
            # Sleep before next check
            time.sleep(15)
            
        except Exception as e:
            logger.error(f"Error in heartbeat monitor: {e}")
            import traceback
            logger.error(traceback.format_exc())
            time.sleep(15) # Ensure sleep even on error

def reconnect_to_ib():
    """Attempt to reconnect to IB and restore the trading state."""
    global ib, model_trader, reconnection_attempts
    
    reconnection_attempts += 1
    if reconnection_attempts > MAX_RECONNECTION_ATTEMPTS:
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
        logger.info(f"Attempting reconnection {reconnection_attempts}/{MAX_RECONNECTION_ATTEMPTS}...")
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
        ib.setTimeout(120)  # 2-minute timeout for IB API calls
        
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

