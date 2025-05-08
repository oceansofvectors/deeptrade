import logging
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, time, timedelta
import pytz
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import optuna
import math
from decimal import Decimal
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import pickle

from environment import TradingEnv
from get_data import get_data
from train import evaluate_agent, plot_results, train_walk_forward_model
from trade import trade_with_risk_management, save_trade_history  # Import trade_with_risk_management and save_trade_history
from config import config
import money
from utils.seeding import seed_worker  # Import the seed_worker function
from normalization import scale_window, get_standardized_column_names  # Import both functions from normalization

# Setup logging to save to file and console
os.makedirs('models/logs', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'models/logs/walk_forward_{timestamp}.log'

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle pandas Timestamp objects
class TimestampJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        # Handle numpy types
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        # Fix for deprecated np.float
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Handle Decimal objects
        if isinstance(obj, Decimal):
            return float(obj)
        return super(TimestampJSONEncoder, self).default(obj)

# Function to safely save JSON data
def save_json(data, filepath):
    """Save data to JSON file with custom encoder for timestamps."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, cls=TimestampJSONEncoder)
    logger.info(f"Saved JSON data to {filepath}")

def load_tradingview_data(csv_filepath: str = "data/data/NQ_2024_unix.csv") -> pd.DataFrame:
    """
    Load and process data from a TradingView CSV export file.
    
    Args:
        csv_filepath: Path to the TradingView CSV file
        
    Returns:
        DataFrame: Processed data with technical indicators
    """
    logger.info(f"Loading TradingView data from {csv_filepath}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_filepath)
        
        # Debug information
        logger.info(f"Raw TradingView data columns: {df.columns.tolist()}")
        
        # Check if we have the required columns
        required_cols = ['open', 'high', 'low', 'close']
        time_cols = ['time', 'timestamp']  # Accept either 'time' or 'timestamp'
        available_cols = [col.lower() for col in df.columns]
        
        # Check if all required columns are present (case insensitive)
        if not all(col.lower() in available_cols for col in required_cols):
            logger.error(f"Missing required columns in TradingView data. Available columns: {df.columns.tolist()}")
            return None
            
        # Check if at least one of the time columns is present
        if not any(col.lower() in available_cols for col in time_cols):
            logger.error(f"Missing time/timestamp column in TradingView data. Available columns: {df.columns.tolist()}")
            return None
        
        # Create mapping from available columns to required columns (case insensitive)
        col_mapping = {}
        for req_col in required_cols:
            for avail_col in df.columns:
                if avail_col.lower() == req_col:
                    col_mapping[avail_col] = req_col
        
        # Handle time column mapping
        time_col_found = None
        for time_col in time_cols:
            for avail_col in df.columns:
                if avail_col.lower() == time_col:
                    time_col_found = avail_col
                    col_mapping[avail_col] = 'time'  # Map to standard 'time' name
                    break
            if time_col_found:
                break
        
        # Rename columns to lowercase standard format
        df = df.rename(columns=col_mapping)
        
        # Convert time column to datetime
        try:
            # Try different approaches to convert time to datetime depending on its current format
            if pd.api.types.is_numeric_dtype(df['time']):
                # If time is already numeric (timestamp), convert to datetime
                logger.info("Converting numeric timestamp to datetime")
                df['time'] = pd.to_datetime(df['time'], unit='s')
            else:
                # If time is string or already datetime
                logger.info("Converting string or datetime to datetime")
                df['time'] = pd.to_datetime(df['time'])
                
            # Set time as index
            df = df.set_index('time')
            logger.info("Successfully converted time column to datetime index")
        except Exception as e:
            logger.error(f"Error converting time column: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
        # Extract available indicator columns from TradingView if they exist
        # Map TradingView column names to our expected format
        indicator_mapping = {
            'ema': 'EMA',
            'volume': 'Volume',
            'histogram': 'Histogram', 
            'macd': 'MACD',
            'signal': 'Signal'
        }
        
        # Rename any matching indicator columns
        for tv_col, our_col in indicator_mapping.items():
            for col in df.columns:
                if col.lower() == tv_col.lower():
                    df[our_col] = df[col]
        
        # Process technical indicators using the same logic as in get_data
        from get_data import process_technical_indicators
        
        # Process indicators 
        df = process_technical_indicators(df)
        
        logger.info(f"TradingView data loaded and processed. Shape: {df.shape}")
        logger.info(f"Final columns: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading TradingView data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def filter_market_hours(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the data to only include NYSE market hours (9:30 AM to 4:00 PM ET, Monday to Friday).
    
    Args:
        data: DataFrame with DatetimeIndex in UTC
        
    Returns:
        DataFrame: Filtered data containing only market hours
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.error("Data index is not a DatetimeIndex, cannot filter market hours")
        return data
    
    # Make a copy to avoid modifying the original
    filtered_data = data.copy()
    
    # Convert UTC times to Eastern Time
    eastern = pytz.timezone('US/Eastern')
    
    # Ensure the index is timezone-aware
    if filtered_data.index.tz is None:
        filtered_data.index = filtered_data.index.tz_localize('UTC')
    
    # Convert to Eastern Time
    filtered_data.index = filtered_data.index.tz_convert(eastern)
    
    # Filter for weekdays (Monday=0, Friday=4)
    weekday_mask = (filtered_data.index.dayofweek >= 0) & (filtered_data.index.dayofweek <= 4)
    
    # Filter for market hours (9:30 AM to 4:00 PM ET)
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    hours_mask = (
        (filtered_data.index.time >= market_open) & 
        (filtered_data.index.time <= market_close)
    )
    
    # Apply both filters
    market_hours_mask = weekday_mask & hours_mask
    filtered_data = filtered_data.loc[market_hours_mask]
    
    # Convert back to UTC for consistency with the rest of the system
    filtered_data.index = filtered_data.index.tz_convert('UTC')
    
    # Log filtering results
    filtered_pct = (len(filtered_data) / len(data)) * 100
    logger.info(f"Filtered data to market hours only: {len(filtered_data)} / {len(data)} rows ({filtered_pct:.2f}%)")
    
    return filtered_data

def get_trading_days(data: pd.DataFrame) -> List[str]:
    """
    Extract unique trading days from a DataFrame with filtered market hours.
    
    Args:
        data: DataFrame with DatetimeIndex in UTC, already filtered to market hours
        
    Returns:
        List[str]: List of unique trading days in YYYY-MM-DD format
    """
    # Ensure index is timezone-aware and convert to Eastern Time for day counting
    if data.index.tz is None:
        data_index = data.index.tz_localize('UTC')
    else:
        data_index = data.index.copy()
    
    eastern = pytz.timezone('US/Eastern')
    data_index = data_index.tz_convert(eastern)
    
    # Extract date part (without time) and get unique days
    unique_days = sorted(set(data_index.date.astype(str)))
    logger.info(f"Found {len(unique_days)} unique trading days in the dataset")
    
    return unique_days

def calculate_hit_rate_from_trade_results(results: Dict) -> Dict:
    """
    Calculate hit rate and profitable trades from trade_with_risk_management results.
    
    Args:
        results: Results dictionary from trade_with_risk_management
        
    Returns:
        Dict: Updated results with hit rate metrics
    """
    # Initialize metrics
    profitable_trades = 0
    total_trades = results.get("trade_count", 0)
    hit_rate = 0.0
    trades_with_profit = []  # Initialize this variable to avoid UnboundLocalError
    
    # Process trade history to count profitable trades
    if 'trade_history' in results and len(results['trade_history']) > 0:
        # Filter for complete trades (those with 'profit' field)
        # In trade_with_risk_management, the trade history includes both entries and exits
        # Only exits have a 'profit' field
        trades_with_profit = [trade for trade in results['trade_history'] if 'profit' in trade]
        
        # Count profitable trades (those with positive profit)
        profitable_trades = sum(1 for trade in trades_with_profit if float(trade.get('profit', 0)) > 0)
        
        # Calculate hit rate if we have trades with profit info
        total_completed_trades = len(trades_with_profit)
        if total_completed_trades > 0:
            hit_rate = (profitable_trades / total_completed_trades) * 100
            
            # Log trade profitability breakdown
            logger.debug(f"Trade profitability: {profitable_trades} profitable out of {total_completed_trades} completed trades")
            if len(trades_with_profit) > 0:
                avg_profit = sum(float(trade.get('profit', 0)) for trade in trades_with_profit) / len(trades_with_profit)
                logger.debug(f"Average profit per trade: ${avg_profit:.2f}")
    
    # Update results dictionary with hit rate metrics
    results["hit_rate"] = hit_rate
    results["profitable_trades"] = profitable_trades
    results["completed_trades"] = len(trades_with_profit)
    
    return results

def evaluate_agent_prediction_accuracy(model, test_data, verbose=0, deterministic=True):
    """
    Evaluate a trained agent's prediction accuracy on test data.
    
    This function focuses on measuring how accurately the model predicts price direction
    (up or down) in the next candle, rather than measuring returns.
    
    Args:
        model: Trained PPO model
        test_data: Test data DataFrame
        verbose: Verbosity level (0=silent, 1=info)
        deterministic: Whether to make deterministic predictions
        
    Returns:
        Dict: Results including prediction accuracy metrics
    """
    # Determine which case is used for price columns
    if 'Close' in test_data.columns:
        close_col = 'Close'
    elif 'CLOSE' in test_data.columns:
        close_col = 'CLOSE'
    else:
        close_col = 'close'
        
    # Create evaluation environment
    env = TradingEnv(
        test_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 0.0),
        position_size=config["environment"].get("position_size", 1)
    )
    
    # Initialize tracking variables
    total_predictions = 0
    correct_predictions = 0
    
    # Portfolio tracking
    initial_balance = config["environment"]["initial_balance"]
    current_portfolio = money.to_decimal(initial_balance)
    
    # Position tracking
    current_position = 0  # 0 = no position, 1 = long, -1 = short
    
    # Trade tracking
    trade_history = []
    trade_count = 0
    
    # Track action history for plotting
    action_history = []
    portfolio_history = [float(initial_balance)]
    
    # Reset environment to start evaluation
    obs, _ = env.reset()
    done = False
    
    if verbose > 0:
        logger.info("Starting model evaluation with prediction accuracy tracking")
    
    # Step through the environment until done
    while not done:
        # Get current step and price before taking action
        current_step = env.current_step
        current_price = money.to_decimal(test_data.iloc[current_step][close_col])
        
        # Get model's action
        action, _ = model.predict(obs, deterministic=deterministic)
        action_history.append(int(action))
        
        # Convert action to position (0 = long/buy, 1 = short/sell, 2 = hold)
        if action == 0:  # Long
            new_position = 1
        elif action == 1:  # Short
            new_position = -1
        else:  # Hold - maintain current position
            new_position = current_position
        
        # Check if we're at the last step or not
        # We need the next price to determine if prediction was correct
        if current_step < len(test_data) - 1:
            # Get next price
            next_price = money.to_decimal(test_data.iloc[current_step + 1][close_col])
            price_change = next_price - current_price
            
            # Evaluate prediction accuracy based on action type
            prediction_correct = False
            
            if action == 0:  # Long - prediction is correct if price goes up
                prediction_correct = price_change > 0
            elif action == 1:  # Short - prediction is correct if price goes down
                prediction_correct = price_change < 0
            elif action == 2:  # Hold - prediction is correct if price doesn't change much
                # For hold, we'll define "correct" as the price not changing significantly
                # Using a threshold of 0.1% of the current price
                threshold = current_price * money.to_decimal(0.001)
                prediction_correct = abs(price_change) <= threshold
            
            # Record prediction
            total_predictions += 1
            if prediction_correct:
                correct_predictions += 1
                
            # Log information if verbose
            if verbose > 0 and total_predictions % 100 == 0:
                accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
                logger.info(f"Step {current_step}: Predictions so far - {correct_predictions}/{total_predictions} correct ({accuracy:.2f}%)")
        
        # Record every action, including HOLD actions
        trade_count += 1
        trade_info = {
            "step": current_step,
            "action": "BUY" if action == 0 else "SELL" if action == 1 else "HOLD",
            "price": float(current_price),
            "timestamp": test_data.index[current_step].strftime('%Y-%m-%d %H:%M:%S') if hasattr(test_data.index[current_step], 'strftime') else str(test_data.index[current_step]),
            "position_change": new_position != current_position
        }
        trade_history.append(trade_info)
        
        # Update position
        current_position = new_position
        
        # Take step in environment
        new_obs, reward, done, truncated, info = env.step(action)
        obs = new_obs
        done = done or truncated
        
        # Update portfolio from environment
        current_portfolio = env.net_worth
        portfolio_history.append(float(current_portfolio))
    
    # Calculate final metrics
    prediction_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    total_return_pct = money.calculate_return_pct(current_portfolio, initial_balance)
    
    # Calculate hit rate (percentage of profitable trades)
    hit_rate = 0
    profitable_trades = 0
    if trade_count > 0:
        # For simplicity, estimate profitable trades based on overall performance
        # In a more detailed implementation, we'd track each trade's profitability
        profitable_trades = int(trade_count * (prediction_accuracy / 100))
        hit_rate = (profitable_trades / trade_count) * 100
    
    # Create results dictionary
    results = {
        "prediction_accuracy": prediction_accuracy,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions,
        "total_return_pct": float(total_return_pct),
        "final_portfolio_value": float(current_portfolio),
        "initial_portfolio_value": float(initial_balance),
        "trade_count": trade_count,
        "trade_history": trade_history,
        "hit_rate": hit_rate,
        "profitable_trades": profitable_trades,
        "final_position": current_position,
        "portfolio_history": portfolio_history,
        "action_history": action_history
    }
    
    # Log summary
    if verbose > 0:
        logger.info(f"Evaluation complete: {correct_predictions}/{total_predictions} correct predictions ({prediction_accuracy:.2f}%)")
        logger.info(f"Return: {float(total_return_pct):.2f}%, Final portfolio: ${float(current_portfolio):.2f}")
        logger.info(f"Total trades: {trade_count}")
    
    return results

def export_consolidated_trade_history(all_window_results: List[Dict], session_folder: str) -> None:
    """
    Consolidate trade histories from all windows into a single CSV file.
    
    Args:
        all_window_results: List of results dictionaries from each walk-forward window
        session_folder: Folder to save the consolidated trade history
    """
    # Check if we have trade histories to consolidate
    windows_with_history = [res for res in all_window_results if 'trade_history' in res and res['trade_history'] and len(res['trade_history']) > 0]
    
    if not windows_with_history:
        logger.warning("No non-empty trade histories found in any window. Skipping consolidated export.")
        return
    
    # Create empty list to store all trades
    all_trades = []
    
    # Process each window's trade history
    for res in all_window_results:
        window_num = res.get("window", 0)
        
        if 'trade_history' in res and res['trade_history'] and len(res['trade_history']) > 0:
            # Add window number to each trade
            for trade in res['trade_history']:
                trade_copy = trade.copy()
                trade_copy['window'] = window_num
                trade_copy['test_start'] = res.get('test_start', '')
                trade_copy['test_end'] = res.get('test_end', '')
                all_trades.append(trade_copy)
        elif 'trade_count' in res and res['trade_count'] > 0:
            # Log warning for windows with trades but no trade history
            logger.warning(f"Window {window_num} has {res['trade_count']} trades but empty trade history")
    
    if not all_trades:
        logger.warning("No trades found in any window. Skipping consolidated export.")
        return
    
    # Convert to DataFrame
    consolidated_df = pd.DataFrame(all_trades)
    
    # Sort by date
    if 'date' in consolidated_df.columns:
        consolidated_df.sort_values('date', inplace=True)
    
    # Save to CSV
    export_path = f'{session_folder}/reports/all_windows_trade_history.csv'
    consolidated_df.to_csv(export_path, index=False)
    
    logger.info(f"Exported consolidated trade history from {len(windows_with_history)} windows "
               f"with {len(all_trades)} trades to {export_path}")

def process_single_window(
    window_idx: int,
    num_windows: int,
    window_data: pd.DataFrame,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    window_folder: str,
    initial_timesteps: int,
    additional_timesteps: int,
    max_iterations: int,
    n_stagnant_loops: int,
    improvement_threshold: float,
    run_hyperparameter_tuning: bool,
    tuning_trials: int
) -> Dict:
    """
    Process a single window in the walk-forward analysis.
    """
    # Create a logger for this window
    window_logger = logging.getLogger(f"walk_forward.window_{window_idx}")
    
    # Save window periods
    window_periods = {
        "train_start": train_data.index[0],
        "train_end": train_data.index[-1],
        "validation_start": validation_data.index[0],
        "validation_end": validation_data.index[-1],
        "test_start": test_data.index[0],
        "test_end": test_data.index[-1]
    }
    save_json(window_periods, f"{window_folder}/window_periods.json")
    
    # Scale technical indicator columns within this window to prevent data leakage
    window_logger.info(f"Window {window_idx}: Scaling technical indicators to prevent data leakage")
    
    # Get columns to scale using the helper function from normalization module
    cols_to_scale = get_standardized_column_names(train_data)
    
    # Scale the data
    scaler, train_data, validation_data, test_data = scale_window(
        train_data=train_data,
        val_data=validation_data,
        test_data=test_data,
        cols_to_scale=cols_to_scale,
        feature_range=(-1, 1),
        window_folder=window_folder
    )
    
    # Training the model with the scaled data
    model, training_stats = train_walk_forward_model(
        train_data=train_data,
        validation_data=validation_data,
        initial_timesteps=initial_timesteps,
        additional_timesteps=additional_timesteps,
        max_iterations=max_iterations,
        n_stagnant_loops=n_stagnant_loops,
        improvement_threshold=improvement_threshold,
        window_folder=window_folder,
        run_hyperparameter_tuning=run_hyperparameter_tuning,
        tuning_trials=tuning_trials
    )
    
    # Plot training progress
    plot_training_progress(training_stats, window_folder)
    
    # Load the best model for testing
    model = PPO.load(f"{window_folder}/model")
    
    # Get risk management parameters from config
    risk_config = config.get("risk_management", {})
    risk_enabled = risk_config.get("enabled", False)
    
    # Initialize risk parameters with default values (disabled)
    stop_loss_pct = None
    take_profit_pct = None
    trailing_stop_pct = None
    position_size = 1.0
    max_risk_per_trade_pct = 0.0
    daily_risk_limit = None
    
    # Only set risk parameters if risk management is enabled
    if risk_enabled:
        # Daily risk limit configuration
        daily_risk_config = risk_config.get("daily_risk_limit", {})
        if daily_risk_config.get("enabled", False):
            daily_risk_limit = daily_risk_config.get("max_daily_loss", 1000.0)
        
        # Stop loss configuration
        stop_loss_config = risk_config.get("stop_loss", {})
        if stop_loss_config.get("enabled", False):
            stop_loss_pct = stop_loss_config.get("percentage", 0.0)
        
        # Take profit configuration
        take_profit_config = risk_config.get("take_profit", {})
        if take_profit_config.get("enabled", False):
            take_profit_pct = take_profit_config.get("percentage", 0.0)
        
        # Trailing stop configuration
        trailing_stop_config = risk_config.get("trailing_stop", {})
        if trailing_stop_config.get("enabled", False):
            trailing_stop_pct = trailing_stop_config.get("percentage", 0.0)
        
        # Position sizing configuration
        position_sizing_config = risk_config.get("position_sizing", {})
        if position_sizing_config.get("enabled", False):
            position_size = position_sizing_config.get("size_multiplier", 1.0)
            max_risk_per_trade_pct = position_sizing_config.get("max_risk_per_trade_percentage", 0.0)
    
    # Log risk management settings
    window_logger.info("Risk Management Settings:")
    window_logger.info(f"  Risk Management Enabled: {risk_enabled}")
    if risk_enabled:
        window_logger.info(f"  Daily Risk Limit: {'Enabled' if daily_risk_limit is not None else 'Disabled'}" + 
                         (f" (${daily_risk_limit})" if daily_risk_limit is not None else ""))
        window_logger.info(f"  Stop Loss: {'Enabled' if stop_loss_pct is not None else 'Disabled'}" + 
                         (f" ({stop_loss_pct}%)" if stop_loss_pct is not None else ""))
        window_logger.info(f"  Take Profit: {'Enabled' if take_profit_pct is not None else 'Disabled'}" + 
                         (f" ({take_profit_pct}%)" if take_profit_pct is not None else ""))
        window_logger.info(f"  Trailing Stop: {'Enabled' if trailing_stop_pct is not None else 'Disabled'}" + 
                         (f" ({trailing_stop_pct}%)" if trailing_stop_pct is not None else ""))
        window_logger.info(f"  Position Size: {position_size}")
        window_logger.info(f"  Max Risk per Trade: {max_risk_per_trade_pct}%")
    
    # Evaluate with risk management if enabled
    if risk_enabled:
        window_logger.info(f"Evaluating window {window_idx} with risk management")
        
        # Convert all numeric parameters to Decimal before passing to trade_with_risk_management
        import money
        
        test_results = trade_with_risk_management(
            model_path=f"{window_folder}/model",
            test_data=test_data,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trailing_stop_pct=trailing_stop_pct,
            position_size=position_size,
            max_risk_per_trade_pct=max_risk_per_trade_pct,
            initial_balance=config["environment"]["initial_balance"],
            transaction_cost=config["environment"].get("transaction_cost", 0.0),
            verbose=1,
            daily_risk_limit=daily_risk_limit
        )
    else:
        # Evaluate without risk management
        window_logger.info(f"Evaluating window {window_idx} without risk management")
        test_env = TradingEnv(
            test_data,
            initial_balance=config["environment"]["initial_balance"],
            transaction_cost=0.0
        )
        
        test_results = evaluate_agent(
            model, 
            test_data,
            deterministic=True
        )
    
    # Plot results
    plot_window_performance(test_data, test_results, window_folder, window_idx)
    
    # Save test results
    test_results_path = f'{window_folder}/test_results.json'
    save_json(test_results, test_results_path)
    window_logger.info(f"Test results saved to {test_results_path}")
    
    # Compile window result
    window_result = {
        "window": window_idx,
        "window_folder": window_folder,
        "return": test_results["total_return_pct"],
        "portfolio_value": test_results["final_portfolio_value"],
        "trade_count": test_results["trade_count"],
        "final_position": test_results["final_position"],
        "train_start": train_data.index[0],
        "train_end": train_data.index[-1],
        "test_start": test_data.index[0],
        "test_end": test_data.index[-1]
    }
    
    # Add additional results if available
    if "hit_rate" in test_results:
        window_result["hit_rate"] = test_results["hit_rate"]
        window_result["profitable_trades"] = test_results.get("profitable_trades", 0)
    
    if "prediction_accuracy" in test_results:
        window_result["prediction_accuracy"] = test_results["prediction_accuracy"]
        window_result["correct_predictions"] = test_results.get("correct_predictions", 0)
        window_result["total_predictions"] = test_results.get("total_predictions", 0)
    
    window_logger.info(f"Window {window_idx} Return: {test_results['total_return_pct']:.2f}%, "
                      f"Portfolio Value: ${test_results['final_portfolio_value']:.2f}, "
                      f"Trades: {test_results['trade_count']}")
    
    if "trade_history" in test_results:
        window_result["has_trade_history"] = True
        
        # Only save trade history if there are actual trades in the history
        if test_results["trade_history"] and len(test_results["trade_history"]) > 0:
            trade_history_path = f'{window_folder}/trade_history.csv'
            save_trade_history(test_results["trade_history"], trade_history_path)
            window_logger.info(f"Trade history saved to {trade_history_path}")
        else:
            window_logger.warning(f"Empty trade history even though trade_count is {test_results['trade_count']}. Not saving empty CSV.")
    else:
        window_result["has_trade_history"] = False
    
    return window_result

def walk_forward_testing(
    data: pd.DataFrame,
    window_size: int,
    step_size: int,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    initial_timesteps: int = 10000,
    additional_timesteps: int = 5000,
    max_iterations: int = 10,
    n_stagnant_loops: int = 3,
    improvement_threshold: float = 0.1,
    run_hyperparameter_tuning: bool = False,
    tuning_trials: int = 30
) -> Dict:
    """
    Perform walk-forward testing with anchored walk-forward analysis.
    
    Args:
        data: Full dataset with all technical indicators
        window_size: Size of each walk-forward window in trading days (not calendar days)
        step_size: Number of trading days to step forward at each iteration
        train_ratio: Proportion of window to use for training
        validation_ratio: Proportion of window to use for validation
        initial_timesteps: Initial number of training timesteps
        additional_timesteps: Number of additional timesteps for each training iteration
        max_iterations: Maximum number of training iterations
        n_stagnant_loops: Number of consecutive iterations without improvement before stopping
        improvement_threshold: Minimum percentage improvement considered significant
        run_hyperparameter_tuning: Whether to run hyperparameter tuning before training
        tuning_trials: Number of trials for hyperparameter tuning
        
    Returns:
        Dict: Results of walk-forward testing
    """
    # Create session folder within models directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_folder = f'models/session_{timestamp}'
    os.makedirs(f'{session_folder}/models', exist_ok=True)
    os.makedirs(f'{session_folder}/plots', exist_ok=True)
    os.makedirs(f'{session_folder}/reports', exist_ok=True)
    
    logger.info(f"Created session folder: {session_folder}")
    
    # Check if data is empty
    if data is None or len(data) == 0:
        logger.error("Empty dataset provided for walk-forward testing")
        error_report = {
            "all_window_results": [],
            "avg_return": 0,
            "avg_portfolio": 0,
            "avg_trades": 0,
            "num_windows": 0,
            "error": "Empty dataset"
        }
        save_json(error_report, f'{session_folder}/reports/error_report.json')
        return error_report
    
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.error("Data index is not a DatetimeIndex, cannot perform walk-forward testing")
        error_report = {
            "all_window_results": [],
            "avg_return": 0,
            "avg_portfolio": 0,
            "avg_trades": 0,
            "num_windows": 0,
            "error": "Invalid index type"
        }
        save_json(error_report, f'{session_folder}/reports/error_report.json')
        return error_report
    
    # Filter data to include only market hours
    logger.info("Filtering data to include only NYSE market hours")
    data = filter_market_hours(data)
    
    # Get list of unique trading days in the dataset
    trading_days = get_trading_days(data)
    logger.info(f"Total number of trading days in dataset: {len(trading_days)}")
    
    # Verify we have enough data for at least one window
    if len(trading_days) < window_size:
        logger.error(f"Not enough trading days in dataset ({len(trading_days)}) for window size ({window_size})")
        error_report = {
            "all_window_results": [],
            "avg_return": 0,
            "avg_portfolio": 0,
            "avg_trades": 0,
            "num_windows": 0,
            "error": "Insufficient trading days"
        }
        save_json(error_report, f'{session_folder}/reports/error_report.json')
        return error_report
    
    # Calculate number of windows
    num_windows = max(1, (len(trading_days) - window_size) // step_size + 1)
    logger.info(f"Number of walk-forward windows: {num_windows}")
    
    # Store results for each window
    all_window_results = []
    
    # Get evaluation metric from config
    eval_metric = config.get("training", {}).get("evaluation", {}).get("metric", "return")
    
    # Get parallel processing configuration
    parallel_config = config.get("walk_forward", {}).get("parallel_processing", {})
    use_parallel = parallel_config.get("enabled", False)
    n_processes = parallel_config.get("n_processes", 0)
    max_workers = parallel_config.get("max_workers", 0)
    
    # If n_processes is 0, use the number of available CPU cores
    if n_processes <= 0:
        n_processes = multiprocessing.cpu_count()
    
    # If max_workers is 0, use n_processes
    if max_workers <= 0:
        max_workers = n_processes
    
    # Log parallelization settings
    if use_parallel:
        logger.info(f"Parallel processing enabled with {max_workers} workers (out of {n_processes} CPU cores)")
    else:
        logger.info(f"Parallel processing disabled. Processing {num_windows} windows sequentially.")
    
    # Save session parameters
    session_params = {
        "timestamp": timestamp,
        "window_size_trading_days": window_size,
        "step_size_trading_days": step_size,
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "initial_timesteps": initial_timesteps,
        "additional_timesteps": additional_timesteps,
        "max_iterations": max_iterations,
        "n_stagnant_loops": n_stagnant_loops,
        "improvement_threshold": improvement_threshold,
        "num_windows": num_windows,
        "data_start": data.index[0],
        "data_end": data.index[-1],
        "data_length": len(data),
        "total_trading_days": len(trading_days),
        "market_hours_only": True,
        "evaluation_metric": eval_metric,
        "parallel_processing": use_parallel,
        "n_processes": n_processes,
        "max_workers": max_workers
    }
    
    # Add enabled indicators from config
    indicators_config = config.get("indicators", {})
    enabled_indicators = {}
    
    for indicator_name, indicator_config in indicators_config.items():
        if indicator_config.get("enabled", False):
            enabled_indicators[indicator_name] = indicator_config
    
    # Add enabled indicators to session parameters
    session_params["enabled_indicators"] = enabled_indicators
    
    save_json(session_params, f'{session_folder}/reports/session_parameters.json')
    
    # Prepare window data for all windows
    window_data_list = []
    for i in range(num_windows):
        # Create window folder
        window_folder = f'{session_folder}/models/window_{i+1}'
        os.makedirs(window_folder, exist_ok=True)
        
        # Calculate start and end trading days for this window
        start_day_idx = i * step_size
        end_day_idx = start_day_idx + window_size
        if end_day_idx > len(trading_days):
            end_day_idx = len(trading_days)
        
        start_day = trading_days[start_day_idx]
        end_day = trading_days[end_day_idx - 1]  # -1 because end_day_idx is exclusive
        
        # Convert to Eastern timezone for proper day-based filtering
        eastern = pytz.timezone('US/Eastern')
        data_eastern = data.copy()
        data_eastern.index = data_eastern.index.tz_convert(eastern)
        
        # Extract data for this window by trading days
        window_mask = (data_eastern.index.date.astype(str) >= start_day) & (data_eastern.index.date.astype(str) <= end_day)
        window_data = data_eastern[window_mask].copy()
        
        # Convert back to UTC
        window_data.index = window_data.index.tz_convert('UTC')
        
        # Log window data range
        logger.info(f"Window {i+1} data range: {window_data.index[0]} to {window_data.index[-1]} (UTC)")
        logger.info(f"Window {i+1} data points: {len(window_data)}")
        
        # Split window into train, validation, and test sets
        train_idx = int(len(window_data) * train_ratio)
        validation_idx = train_idx + int(len(window_data) * validation_ratio)
        
        train_data = window_data.iloc[:train_idx].copy()
        validation_data = window_data.iloc[train_idx:validation_idx].copy()
        test_data = window_data.iloc[validation_idx:].copy()
        
        logger.info(f"Window {i+1} - Train period: {train_data.index[0]} to {train_data.index[-1]}")
        logger.info(f"Window {i+1} - Validation period: {validation_data.index[0]} to {validation_data.index[-1]}")
        logger.info(f"Window {i+1} - Test period: {test_data.index[0]} to {test_data.index[-1]}")
        
        # Store window data
        window_data_list.append({
            "window_idx": i+1,
            "window_data": window_data,
            "train_data": train_data,
            "validation_data": validation_data,
            "test_data": test_data,
            "window_folder": window_folder
        })
    
    # Process windows - either in parallel or sequentially
    if use_parallel and num_windows > 1:
        # Process windows in parallel
        logger.info(f"Processing {num_windows} windows in parallel with {max_workers} workers")
        
        seed_value = config.get("seed", 42)
        logger.info(f"Using seed {seed_value} for worker initialization")
        
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=seed_worker, 
            initargs=(seed_value,)
        ) as executor:
            futures = []
            
            # Submit all windows for processing
            for window_data_dict in window_data_list:
                futures.append(
                    executor.submit(
                        process_single_window,
                        window_data_dict["window_idx"],
                        num_windows,
                        window_data_dict["window_data"],
                        window_data_dict["train_data"],
                        window_data_dict["validation_data"],
                        window_data_dict["test_data"],
                        window_data_dict["window_folder"],
                        initial_timesteps,
                        additional_timesteps,
                        max_iterations,
                        n_stagnant_loops,
                        improvement_threshold,
                        run_hyperparameter_tuning,
                        tuning_trials
                    )
                )
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    window_result = future.result()
                    all_window_results.append(window_result)
                    logger.info(f"Completed window {window_result['window']}/{num_windows}")
                except Exception as e:
                    logger.error(f"Error processing window: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        
        # Sort results by window number
        all_window_results.sort(key=lambda x: x["window"])
    else:
        # Process windows sequentially
        logger.info(f"Processing {num_windows} windows sequentially")
        
        for window_data_dict in window_data_list:
            try:
                window_result = process_single_window(
                    window_data_dict["window_idx"],
                    num_windows,
                    window_data_dict["window_data"],
                    window_data_dict["train_data"],
                    window_data_dict["validation_data"],
                    window_data_dict["test_data"],
                    window_data_dict["window_folder"],
                    initial_timesteps,
                    additional_timesteps,
                    max_iterations,
                    n_stagnant_loops,
                    improvement_threshold,
                    run_hyperparameter_tuning,
                    tuning_trials
                )
                all_window_results.append(window_result)
            except Exception as e:
                logger.error(f"Error processing window {window_data_dict['window_idx']}: {e}")
                import traceback
                logger.error(traceback.format_exc())

    # Aggregate results across all windows
    returns = [res["return"] for res in all_window_results]
    portfolio_values = [res["portfolio_value"] for res in all_window_results]
    trade_counts = [res["trade_count"] for res in all_window_results]
    
    # Also aggregate hit rates if that metric is used
    hit_rates = [res.get("hit_rate", 0) for res in all_window_results]
    profitable_trades = [res.get("profitable_trades", 0) for res in all_window_results]
    
    # Also aggregate prediction accuracies if that metric is used
    prediction_accuracies = [res.get("prediction_accuracy", 0) for res in all_window_results]
    correct_predictions = [res.get("correct_predictions", 0) for res in all_window_results]
    total_predictions = [res.get("total_predictions", 0) for res in all_window_results]
    
    # Calculate average metrics
    avg_return = np.mean(returns)
    avg_portfolio = np.mean(portfolio_values)
    avg_trades = np.mean(trade_counts)
    avg_hit_rate = np.mean(hit_rates) if hit_rates else 0
    avg_profitable_trades = np.mean(profitable_trades) if profitable_trades else 0
    avg_prediction_accuracy = np.mean(prediction_accuracies) if prediction_accuracies else 0
    avg_correct_predictions = np.mean(correct_predictions) if correct_predictions else 0
    avg_total_predictions = np.mean(total_predictions) if total_predictions else 0
    
    logger.info(f"\n{'='*80}\nWalk-Forward Testing Summary\n{'='*80}")
    logger.info(f"Number of windows: {num_windows}")
    logger.info(f"Average return: {avg_return:.2f}%")
    
    if eval_metric == "hit_rate":
        logger.info(f"Average hit rate: {avg_hit_rate:.2f}%")
        logger.info(f"Average profitable trades: {avg_profitable_trades:.2f} out of {avg_trades:.2f}")
    elif eval_metric == "prediction_accuracy":
        logger.info(f"Average prediction accuracy: {avg_prediction_accuracy:.2f}%")
        logger.info(f"Average correct predictions: {avg_correct_predictions:.2f} out of {avg_total_predictions:.2f}")
    
    logger.info(f"Average final portfolio: ${avg_portfolio:.2f}")
    logger.info(f"Average trade count: {avg_trades:.2f}")
    
    # Save summary results
    summary_results = {
        "avg_return": avg_return,
        "avg_hit_rate": avg_hit_rate,
        "avg_prediction_accuracy": avg_prediction_accuracy,
        "avg_portfolio": avg_portfolio,
        "avg_trades": avg_trades,
        "avg_profitable_trades": avg_profitable_trades,
        "avg_correct_predictions": avg_correct_predictions,
        "avg_total_predictions": avg_total_predictions,
        "num_windows": num_windows,
        "timestamp": timestamp,
        "evaluation_metric": eval_metric
    }
    
    # Don't include all window results in the summary JSON (they may contain non-serializable objects)
    save_json(summary_results, f'{session_folder}/reports/summary_results.json')
    
    # Add window results back for the return value (not for JSON serialization)
    summary_results["all_window_results"] = all_window_results
    
    # Plot results
    plot_walk_forward_results(all_window_results, session_folder, eval_metric)
    
    # Export consolidated trade history
    export_consolidated_trade_history(all_window_results, session_folder)
    
    return summary_results

def hyperparameter_tuning(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    n_trials: int = 30,
    window_folder: str = None,
    eval_metric: str = "return",
    hit_rate_min_trades: int = 5
) -> Dict:
    """
    Run hyperparameter tuning using Optuna.
    
    Args:
        train_data: Training data DataFrame
        validation_data: Validation data DataFrame
        n_trials: Number of trials for Optuna optimization
        window_folder: Folder to save visualization results
        eval_metric: Evaluation metric to optimize for
        hit_rate_min_trades: Minimum number of trades required for hit rate to be meaningful
        
    Returns:
        Dict: Best hyperparameters and optimization results
    """
    logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
    logger.info(f"Evaluation metric: {eval_metric}")
    
    # Get minimum predictions for prediction accuracy metric
    min_predictions = config.get("training", {}).get("evaluation", {}).get("min_predictions", 10)
    
    # Get risk management parameters from config
    risk_enabled = config.get("risk_management", {}).get("enabled", False)
    
    # Set up risk management parameters if enabled
    stop_loss_pct = 0
    take_profit_pct = 0
    trailing_stop_pct = 0
    position_size = 1
    max_risk_per_trade_pct = 1.0
    
    if risk_enabled:
        # Get stop loss configuration
        stop_loss_config = config.get("risk_management", {}).get("stop_loss", {})
        if stop_loss_config.get("enabled", False):
            stop_loss_pct = stop_loss_config.get("percentage", 0)
        else:
            stop_loss_pct = 0
            
        # Get take profit configuration
        take_profit_config = config.get("risk_management", {}).get("take_profit", {})
        if take_profit_config.get("enabled", False):
            take_profit_pct = take_profit_config.get("percentage", 0)
        else:
            take_profit_pct = 0
            
        # Get trailing stop configuration - respect the 'enabled' flag
        trailing_stop_config = config.get("risk_management", {}).get("trailing_stop", {})
        if trailing_stop_config.get("enabled", False):
            trailing_stop_pct = trailing_stop_config.get("percentage", 0)
        else:
            trailing_stop_pct = 0  # Set to 0 to disable if not enabled
            
        # Get position size configuration
        position_size = config["environment"].get("position_size", 1)
        max_risk_per_trade_pct = config.get("risk_management", {}).get("position_sizing", {}).get("max_risk_per_trade_percentage", 1.0)
        
        logger.info(f"Risk management is enabled: SL={stop_loss_pct}% (enabled={stop_loss_config.get('enabled', False)}), " +
                   f"TP={take_profit_pct}% (enabled={take_profit_config.get('enabled', False)}), " +
                   f"TS={trailing_stop_pct}% (enabled={trailing_stop_config.get('enabled', False)})")
    else:
        logger.info("Risk management is disabled")
    
    # Create Optuna study
    study_name = f"wf_hyperparam_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    
    def objective(trial):
        # Define hyperparameters to tune
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        n_steps = trial.suggest_int("n_steps", 128, 2048, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.5, log=True)
        batch_size = trial.suggest_int("batch_size", 8, 128, log=True)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.999)
        
        # Create environment
        train_env = TradingEnv(
            train_data,
            initial_balance=config["environment"]["initial_balance"],
            transaction_cost=0.0,
            position_size=config["environment"].get("position_size", 1)
        )
        
        # Create model with trial hyperparameters
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            learning_rate=learning_rate,
            n_steps=n_steps,
            ent_coef=ent_coef,
            batch_size=batch_size,
            gamma=gamma,
            seed=config.get('seed'),
            gae_lambda=gae_lambda
        )
        
        # Train the model
        total_timesteps = config["training"].get("total_timesteps", 10000)
        model.learn(total_timesteps=total_timesteps)
        
        # Evaluate on validation data
        if risk_enabled and window_folder and eval_metric != "prediction_accuracy":
            # Save the model temporarily for evaluation with risk management
            temp_model_path = f"{window_folder}/temp_model_trial_{trial.number}"
            model.save(temp_model_path)
            
            # Evaluate with risk management
            results = trade_with_risk_management(
                model_path=temp_model_path,
                test_data=validation_data,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                trailing_stop_pct=trailing_stop_pct,
                position_size=position_size,
                max_risk_per_trade_pct=max_risk_per_trade_pct,
                initial_balance=config["environment"]["initial_balance"],
                transaction_cost=config["environment"].get("transaction_cost", 0.0),
                verbose=0,
                deterministic=True,
                close_at_end_of_day=True
            )
            
            # Clean up temporary model file
            if os.path.exists(f"{temp_model_path}.zip"):
                os.remove(f"{temp_model_path}.zip")
                
            # Calculate hit rate from trade results
            results = calculate_hit_rate_from_trade_results(results)
        elif eval_metric == "prediction_accuracy":
            # Use prediction accuracy evaluation
            results = evaluate_agent_prediction_accuracy(model, validation_data, verbose=0, deterministic=True)
        else:
            # Use standard evaluation without risk management
            results = evaluate_agent(model, validation_data, verbose=0, deterministic=True)
        
        # Determine metric value to optimize
        if eval_metric == "hit_rate" and results["trade_count"] >= hit_rate_min_trades:
            metric_value = results["hit_rate"]
            logger.info(f"Trial {trial.number}: Hit Rate = {metric_value:.2f}% ({results.get('profitable_trades', 0)}/{results['trade_count']} trades)")
        elif eval_metric == "prediction_accuracy" and results.get("total_predictions", 0) >= min_predictions:
            metric_value = results["prediction_accuracy"]
            logger.info(f"Trial {trial.number}: Prediction Accuracy = {metric_value:.2f}% ({results.get('correct_predictions', 0)}/{results.get('total_predictions', 0)} predictions)")
        else:
            metric_value = results["total_return_pct"]
            logger.info(f"Trial {trial.number}: Return = {metric_value:.2f}%")
            
            if eval_metric == "hit_rate" and results["trade_count"] < hit_rate_min_trades:
                logger.warning(f"Not enough trades ({results['trade_count']}) for hit rate metric. Using return instead.")
            elif eval_metric == "prediction_accuracy" and results.get("total_predictions", 0) < min_predictions:
                logger.warning(f"Not enough predictions ({results.get('total_predictions', 0)}) for prediction accuracy metric. Using return instead.")
        
        # Log all hyperparameters
        logger.info(f"Parameters: lr={learning_rate:.6f}, n_steps={n_steps}, ent_coef={ent_coef:.6f}, "
                  f"batch_size={batch_size}, gamma={gamma:.4f}, gae_lambda={gae_lambda:.4f}")
        
        return metric_value
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info("\n" + "="*80)
    logger.info("Hyperparameter Tuning Results:")
    logger.info(f"Best {eval_metric}: {best_value:.2f}%")
    logger.info("Best parameters:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    
    # Save visualization if folder is provided
    if window_folder:
        try:
            # Create optimization visualization plots
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.write_image(f'{window_folder}/optimization_history.png')
            
            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.write_image(f'{window_folder}/param_importances.png')
            
            logger.info(f"Saved optimization visualizations to {window_folder}")
        except Exception as e:
            logger.warning(f"Could not save optimization visualizations: {e}")
    
    return {
        "best_params": best_params,
        "best_value": best_value,
        "study": study
    }

def plot_training_progress(training_stats: List[Dict], window_folder: str) -> None:
    """
    Plot the training progress over iterations.
    
    Args:
        training_stats: List of training statistics dictionaries
        window_folder: Folder to save the plot
    """
    # Extract data from training statistics
    iterations = [stat["iteration"] for stat in training_stats]
    returns = [stat["return_pct"] for stat in training_stats]
    portfolio_values = [stat["portfolio_value"] for stat in training_stats]
    is_best = [stat["is_best"] for stat in training_stats]
    
    # Determine which metric to highlight
    metric_name = training_stats[0]["metric_used"]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Set metric-specific values
    if metric_name == "hit_rate":
        metric_values = [stat["hit_rate"] for stat in training_stats]
        y_label = 'Hit Rate (%)'
        title_prefix = 'Hit Rate'
    elif metric_name == "prediction_accuracy":
        metric_values = [stat["prediction_accuracy"] for stat in training_stats]
        y_label = 'Prediction Accuracy (%)'
        title_prefix = 'Prediction Accuracy'
    else:
        metric_values = [stat["return_pct"] for stat in training_stats]
        y_label = 'Return (%)'
        title_prefix = 'Return'
    
    # Plot metric values
    for i, (iteration, value, best) in enumerate(zip(iterations, metric_values, is_best)):
        color = 'green' if best else 'blue'
        plt.bar(iteration, value, color=color, alpha=0.7)
        
        # Add text label for value
        plt.text(iteration, value, f"{value:.1f}%", 
                ha='center', va='bottom', fontsize=8)
        
        # Add star for best models
        if best:
            plt.text(iteration, value, '⭐', 
                    ha='center', va='top', fontsize=12)
    
    plt.xlabel('Training Iteration')
    plt.ylabel(y_label)
    plt.title(f'{title_prefix} by Training Iteration')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Use integer x-axis ticks
    plt.xticks(iterations)
    
    plt.tight_layout()
    plt.savefig(f'{window_folder}/training_progress.png')
    plt.close()
    
    # Second plot for portfolio value
    plt.figure(figsize=(10, 6))
    plt.bar(iterations, portfolio_values, color='purple', alpha=0.7)
    
    for i, (iteration, value) in enumerate(zip(iterations, portfolio_values)):
        plt.text(iteration, value, f"${value:.0f}", 
                ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Training Iteration')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value by Training Iteration')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(iterations)
    
    plt.tight_layout()
    plt.savefig(f'{window_folder}/portfolio_progress.png')
    plt.close()
    
    # Additional plot for trade count if using hit rate
    if metric_name == "hit_rate":
        plt.figure(figsize=(10, 6))
        trade_counts = [stat["trade_count"] for stat in training_stats]
        profitable_trades = [stat.get("profitable_trades", 0) for stat in training_stats]
        
        plt.bar(iterations, trade_counts, color='blue', alpha=0.7, label='Total Trades')
        plt.bar(iterations, profitable_trades, color='green', alpha=0.7, label='Profitable Trades')
        
        plt.xlabel('Training Iteration')
        plt.ylabel('Number of Trades')
        plt.title('Trade Performance by Iteration')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{window_folder}/trade_performance.png')
        plt.close()
    
    # Additional plot for prediction accuracy if using that metric
    elif metric_name == "prediction_accuracy":
        plt.figure(figsize=(10, 6))
        total_predictions = [stat["total_predictions"] for stat in training_stats]
        correct_predictions = [stat.get("correct_predictions", 0) for stat in training_stats]
        
        plt.bar(iterations, total_predictions, color='blue', alpha=0.7, label='Total Predictions')
        plt.bar(iterations, correct_predictions, color='green', alpha=0.7, label='Correct Predictions')
        
        plt.xlabel('Training Iteration')
        plt.ylabel('Number of Predictions')
        plt.title('Prediction Performance by Iteration')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{window_folder}/prediction_performance.png')
        plt.close()

def plot_window_performance(test_data: pd.DataFrame, test_results: Dict, window_folder: str, window_num: int) -> None:
    """
    Plot the performance of a window on test data.
    
    Args:
        test_data: Test data for this window
        test_results: Results dictionary from evaluation
        window_folder: Folder to save the plot
        window_num: Window number
    """
    if 'portfolio_history' not in test_results:
        logger.warning(f"No portfolio history available for window {window_num}. Skipping performance plot.")
        return
    
    portfolio_history = test_results['portfolio_history']
    action_history = test_results.get('action_history', [])
    
    # Ensure action_history is always an array
    action_history = np.atleast_1d(action_history)
    
    plt.figure(figsize=(12, 8))
    
    # Plot portfolio value
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_history, color='blue', label='Portfolio Value')
    plt.title(f'Window {window_num} Test Performance')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot price and buy/sell signals
    plt.subplot(2, 1, 2)
    
    # Plot price line
    if 'CLOSE' in test_data.columns:
        plt.plot(test_data.index, test_data['CLOSE'], color='gray', label='Price')
    elif 'Close' in test_data.columns:
        plt.plot(test_data.index, test_data['Close'], color='gray', label='Price')
    else:
        plt.plot(test_data.index, test_data['close'], color='gray', label='Price')
    
    # Plot buy signals
    buy_indices = np.where(action_history == 0)[0]
    if buy_indices.size > 0:
        # Ensure buy_indices are within range
        buy_dates = test_data.index[buy_indices[buy_indices < len(test_data)]]
        
        # Get buy prices with proper column name handling
        if 'CLOSE' in test_data.columns:
            buy_prices = [test_data['CLOSE'].iloc[i] for i in buy_indices if i < len(test_data)]
        elif 'Close' in test_data.columns:
            buy_prices = [test_data['Close'].iloc[i] for i in buy_indices if i < len(test_data)]
        else:
            buy_prices = [test_data['close'].iloc[i] for i in buy_indices if i < len(test_data)]
            
        plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy')
    
    # Plot sell signals
    sell_indices = np.where(action_history == 1)[0]
    if sell_indices.size > 0:
        # Ensure sell_indices are within range
        sell_dates = test_data.index[sell_indices[sell_indices < len(test_data)]]
        
        # Get sell prices with proper column name handling
        if 'CLOSE' in test_data.columns:
            sell_prices = [test_data['CLOSE'].iloc[i] for i in sell_indices if i < len(test_data)]
        elif 'Close' in test_data.columns:
            sell_prices = [test_data['Close'].iloc[i] for i in sell_indices if i < len(test_data)]
        else:
            sell_prices = [test_data['close'].iloc[i] for i in sell_indices if i < len(test_data)]
            
        plt.scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='Sell')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{window_folder}/test_performance.png')
    plt.close()

def plot_walk_forward_results(all_window_results: List[Dict], session_folder: str, eval_metric: str) -> None:
    """
    Plot the results of walk-forward testing.
    
    Args:
        all_window_results: List of results dictionaries from each walk-forward window
        session_folder: Folder to save the plot
        eval_metric: Evaluation metric used for the walk-forward testing
    """
    windows = [res["window"] for res in all_window_results]
    returns = [res["return"] for res in all_window_results]
    portfolio_values = [res["portfolio_value"] for res in all_window_results]
    trade_counts = [res["trade_count"] for res in all_window_results]
    
    # Also get hit rates if that metric is used
    hit_rates = [res.get("hit_rate", 0) for res in all_window_results]
    profitable_trades = [res.get("profitable_trades", 0) for res in all_window_results]
    
    # Get prediction accuracy metrics if that metric is used
    prediction_accuracies = [res.get("prediction_accuracy", 0) for res in all_window_results]
    correct_predictions = [res.get("correct_predictions", 0) for res in all_window_results]
    total_predictions = [res.get("total_predictions", 0) for res in all_window_results]
    
    # Number of subplots depends on which metric is being used
    num_plots = 4 if eval_metric in ["hit_rate", "prediction_accuracy"] else 3
    
    # Create figure with appropriate number of subplots
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots), sharex=True)
    
    plot_idx = 0
    
    # Plot returns
    axs[plot_idx].bar(windows, returns, color='blue')
    axs[plot_idx].set_ylabel('Return (%)')
    axs[plot_idx].set_title('Returns by Walk-Forward Window')
    axs[plot_idx].grid(True, linestyle='--', alpha=0.7)
    plot_idx += 1
    
    # Plot hit rates if that metric is used
    if eval_metric == "hit_rate":
        axs[plot_idx].bar(windows, hit_rates, color='green')
        axs[plot_idx].set_ylabel('Hit Rate (%)')
        axs[plot_idx].set_title('Hit Rates by Walk-Forward Window')
        axs[plot_idx].grid(True, linestyle='--', alpha=0.7)
        
        # Add text annotations showing number of trades
        for i, (hr, pt, tc) in enumerate(zip(hit_rates, profitable_trades, trade_counts)):
            if tc > 0:  # Only annotate windows with trades
                axs[plot_idx].annotate(f"{int(pt)}/{int(tc)}", 
                                     (windows[i], hr),
                                     textcoords="offset points", 
                                     xytext=(0,5), 
                                     ha='center')
        plot_idx += 1
    
    # Plot prediction accuracy if that metric is used
    elif eval_metric == "prediction_accuracy":
        axs[plot_idx].bar(windows, prediction_accuracies, color='green')
        axs[plot_idx].set_ylabel('Prediction Accuracy (%)')
        axs[plot_idx].set_title('Prediction Accuracies by Walk-Forward Window')
        axs[plot_idx].grid(True, linestyle='--', alpha=0.7)
        
        # Add text annotations showing number of predictions
        for i, (pa, cp, tp) in enumerate(zip(prediction_accuracies, correct_predictions, total_predictions)):
            if tp > 0:  # Only annotate windows with predictions
                axs[plot_idx].annotate(f"{int(cp)}/{int(tp)}", 
                                     (windows[i], pa),
                                     textcoords="offset points", 
                                     xytext=(0,5), 
                                     ha='center')
        plot_idx += 1
    
    # Plot portfolio values
    axs[plot_idx].bar(windows, portfolio_values, color='purple')
    axs[plot_idx].set_ylabel('Final Portfolio Value ($)')
    axs[plot_idx].set_title('Final Portfolio Values by Walk-Forward Window')
    axs[plot_idx].grid(True, linestyle='--', alpha=0.7)
    plot_idx += 1
    
    # Plot trade counts
    axs[plot_idx].bar(windows, trade_counts, color='red')
    axs[plot_idx].set_xlabel('Walk-Forward Window')
    axs[plot_idx].set_ylabel('Trade Count')
    axs[plot_idx].set_title('Trade Counts by Walk-Forward Window')
    axs[plot_idx].grid(True, linestyle='--', alpha=0.7)
    
    # If using hit rate, add profitable trades as a second bar
    if eval_metric == "hit_rate":
        axs[plot_idx].bar(windows, profitable_trades, color='green', alpha=0.7)
        axs[plot_idx].legend(['Total Trades', 'Profitable Trades'])
    # If using prediction accuracy, add correct predictions as a second bar
    elif eval_metric == "prediction_accuracy":
        axs[plot_idx].bar(windows, correct_predictions, color='green', alpha=0.7)
        axs[plot_idx].legend(['Total Predictions', 'Correct Predictions'])
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'{session_folder}/plots/walk_forward_results_{eval_metric}.png')
    plt.close()
    
    # Also create and save cumulative charts
    plt.figure(figsize=(12, 6))
    
    # Cumulative returns
    cumulative_returns = np.cumsum(returns)
    plt.plot(windows, cumulative_returns, marker='o', linestyle='-', color='blue', label='Cumulative Return (%)')
    
    # If using hit rate, also plot cumulative hit rate
    if eval_metric == "hit_rate":
        # Calculate cumulative hit rate (cumulative profitable trades / cumulative total trades)
        cum_trades = np.cumsum(trade_counts)
        cum_profitable = np.cumsum(profitable_trades)
        cum_hit_rate = [100 * cum_profitable[i] / cum_trades[i] if cum_trades[i] > 0 else 0 for i in range(len(cum_trades))]
        
        # Add to plot with secondary y-axis
        ax2 = plt.gca().twinx()
        ax2.plot(windows, cum_hit_rate, marker='s', linestyle='-', color='green', label='Cumulative Hit Rate (%)')
        ax2.set_ylabel('Cumulative Hit Rate (%)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
    
    # If using prediction accuracy, also plot cumulative prediction accuracy
    elif eval_metric == "prediction_accuracy":
        # Calculate cumulative prediction accuracy (cumulative correct predictions / cumulative total predictions)
        cum_predictions = np.cumsum(total_predictions)
        cum_correct = np.cumsum(correct_predictions)
        cum_accuracy = [100 * cum_correct[i] / cum_predictions[i] if cum_predictions[i] > 0 else 0 for i in range(len(cum_predictions))]
        
        # Add to plot with secondary y-axis
        ax2 = plt.gca().twinx()
        ax2.plot(windows, cum_accuracy, marker='s', linestyle='-', color='green', label='Cumulative Prediction Accuracy (%)')
        ax2.set_ylabel('Cumulative Prediction Accuracy (%)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
    
    plt.xlabel('Walk-Forward Window')
    plt.ylabel('Cumulative Return (%)', color='blue')
    plt.tick_params(axis='y', labelcolor='blue')
    plt.title('Cumulative Performance Across Walk-Forward Windows')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create legend that incorporates both axes
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    if eval_metric in ["hit_rate", "prediction_accuracy"]:
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(f'{session_folder}/plots/cumulative_performance_{eval_metric}.png')
    plt.close()

def main():
    # Create model directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/logs', exist_ok=True)
    
    # Load data from TradingView CSV instead of Yahoo Finance
    full_data = load_tradingview_data("data/NQ_2024_unix.csv")
    
    # Check if data loading was successful
    if full_data is None or len(full_data) == 0:
        logger.error("Failed to load TradingView data or dataset is empty. Check your data file.")
        print("\nERROR: Data loading failed. Please check the TradingView data file and ensure it contains valid data.")
        return
    
    logger.info(f"Loaded TradingView dataset with {len(full_data)} rows spanning from {full_data.index[0]} to {full_data.index[-1]}")
    logger.info("Note: Dataset will be filtered to include only NYSE market hours (9:30 AM to 4:00 PM ET, Monday to Friday)")
    
    # Get walk-forward parameters from config or use defaults
    wf_config = config.get("walk_forward", {})
    window_size = wf_config.get("window_size", 14)  # 14 trading days default
    step_size = wf_config.get("step_size", 7)       # 7 trading days default
    
    # Log information about how days are counted
    logger.info(f"Window size: {window_size} trading days (NYSE business days, not calendar days)")
    logger.info(f"Step size: {step_size} trading days (NYSE business days, not calendar days)")
    
    # Get evaluation metric from config
    eval_metric = config.get("training", {}).get("evaluation", {}).get("metric", "return")
    
    # Check if risk management is enabled
    risk_config = config.get("risk_management", {})
    risk_enabled = risk_config.get("enabled", False)
    
    if risk_enabled:
        # Log risk management settings
        logger.info("Risk management is ENABLED for walk-forward testing")
        
        # Stop loss configuration
        stop_loss_config = risk_config.get("stop_loss", {})
        if stop_loss_config.get("enabled", False):
            stop_loss_pct = stop_loss_config.get("percentage", 1.0)
            logger.info(f"  - Stop loss: {stop_loss_pct}%")
        else:
            logger.info("  - Stop loss: Disabled")
        
        # Take profit configuration
        take_profit_config = risk_config.get("take_profit", {})
        if take_profit_config.get("enabled", False):
            take_profit_pct = take_profit_config.get("percentage", 2.0)
            logger.info(f"  - Take profit: {take_profit_pct}%")
        else:
            logger.info("  - Take profit: Disabled")
        
        # Trailing stop configuration
        trailing_stop_config = risk_config.get("trailing_stop", {})
        if trailing_stop_config.get("enabled", False):
            trailing_stop_pct = trailing_stop_config.get("percentage", 0.5)
            logger.info(f"  - Trailing Stop: {trailing_stop_pct}%")
        else:
            logger.info("  - Trailing Stop: Disabled")
        
        # Position sizing configuration
        position_sizing_config = risk_config.get("position_sizing", {})
        if position_sizing_config.get("enabled", False):
            position_size = position_sizing_config.get("size_multiplier", 1.0)
            max_risk_per_trade_pct = position_sizing_config.get("max_risk_per_trade_percentage", 2.0)
    else:
        logger.info("Risk management is DISABLED for walk-forward testing")
    
    # Check if hyperparameter tuning is enabled in config
    hyperparameter_tuning_enabled = config.get("hyperparameter_tuning", {}).get("enabled", False)
    tuning_trials = config.get("hyperparameter_tuning", {}).get("n_trials", 30)
    
    # Run walk-forward testing with hyperparameter tuning if enabled
    results = walk_forward_testing(
        data=full_data,
        window_size=window_size,
        step_size=step_size,
        train_ratio=config["data"].get("train_ratio", 0.7),
        validation_ratio=config["data"].get("validation_ratio", 0.15),
        initial_timesteps=config["training"].get("total_timesteps", 10000),
        additional_timesteps=config["training"].get("additional_timesteps", 5000),
        max_iterations=config["training"].get("max_iterations", 10),
        n_stagnant_loops=config["training"].get("n_stagnant_loops", 3),
        improvement_threshold=config["training"].get("improvement_threshold", 0.1),
        run_hyperparameter_tuning=hyperparameter_tuning_enabled,
        tuning_trials=tuning_trials
    )
    
    # Check if we have results
    if "error" in results:
        logger.error(f"Walk-forward testing failed: {results.get('error', 'Unknown error')}")
        print(f"\nERROR: Walk-forward testing failed: {results.get('error', 'Unknown error')}")
        return
    
    if results["num_windows"] == 0:
        logger.warning("No walk-forward windows were processed")
        print("\nWARNING: No walk-forward windows were processed. Check your window_size and step_size settings.")
        return
        
    # Print summary
    print("\nWalk-Forward Testing Summary:")
    print(f"Number of windows: {results['num_windows']}")
    print(f"Average return: {results['avg_return']:.2f}%")
    
    # Display hit rate metrics if that evaluation metric was used
    if eval_metric == "hit_rate":
        print(f"Average hit rate: {results['avg_hit_rate']:.2f}%")
        print(f"Average profitable trades: {results['avg_profitable_trades']:.2f} out of {results['avg_trades']:.2f}")
    
    print(f"Average final portfolio: ${results['avg_portfolio']:.2f}")
    print(f"Average trade count: {results['avg_trades']:.2f}")
    print(f"Results saved to models/session_{results['timestamp']}")
    print(f"Evaluation metric used: {eval_metric}")
    
    # Print risk management information
    if risk_enabled:
        print("\nRisk Management Settings:")
        # Stop loss
        if stop_loss_config.get("enabled", False):
            print(f"  Stop Loss: {stop_loss_pct}%")
        else:
            print("  Stop Loss: Disabled")
        
        # Take profit
        if take_profit_config.get("enabled", False):
            print(f"  Take Profit: {take_profit_pct}%")
        else:
            print("  Take Profit: Disabled")
        
        # Trailing stop
        if trailing_stop_config.get("enabled", False):
            print(f"  Trailing Stop: {trailing_stop_pct}%")
        else:
            print("  Trailing Stop: Disabled")
    
    
    print("Note: All references to 'days' now indicate trading days (NYSE business days), not calendar days")
    print("Note: Training was performed only on NYSE market hours data (9:30 AM to 4:00 PM ET, Monday to Friday)")

if __name__ == "__main__":
    main() 