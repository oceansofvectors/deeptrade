import logging
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, time
import pytz
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from environment import TradingEnv
from get_data import get_data
from train import evaluate_agent, plot_results
from trade import trade_with_risk_management, save_trade_history  # Import trade_with_risk_management and save_trade_history
from config import config
import money

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
        if isinstance(obj, (np.float, np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(TimestampJSONEncoder, self).default(obj)

# Function to safely save JSON data
def save_json(data, filepath):
    """
    Safely save data to JSON file with proper timestamp handling.
    
    Args:
        data: The data to save
        filepath: Path to save the JSON file
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4, cls=TimestampJSONEncoder)
        logger.info(f"Successfully saved JSON data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving JSON data to {filepath}: {e}")

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
    results["completed_trades"] = len(trades_with_profit) if 'trade_history' in results else 0
    
    return results

def export_consolidated_trade_history(all_window_results: List[Dict], session_folder: str) -> None:
    """
    Consolidate trade histories from all windows into a single CSV file.
    
    Args:
        all_window_results: List of results dictionaries from each walk-forward window
        session_folder: Folder to save the consolidated trade history
    """
    # Check if we have trade histories to consolidate
    windows_with_history = [res for res in all_window_results if 'trade_history' in res and res['trade_history']]
    
    if not windows_with_history:
        logger.warning("No trade histories found in any window. Skipping consolidated export.")
        return
    
    # Create empty list to store all trades
    all_trades = []
    
    # Process each window's trade history
    for res in all_window_results:
        window_num = res.get("window", 0)
        
        if 'trade_history' in res and res['trade_history']:
            # Add window number to each trade
            for trade in res['trade_history']:
                trade_copy = trade.copy()
                trade_copy['window'] = window_num
                trade_copy['test_start'] = res.get('test_start', '')
                trade_copy['test_end'] = res.get('test_end', '')
                all_trades.append(trade_copy)
    
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
    improvement_threshold: float = 0.1
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
        "evaluation_metric": eval_metric
    }
    
    save_json(session_params, f'{session_folder}/reports/session_parameters.json')
    
    # Walk-forward testing
    for i in range(num_windows):
        logger.info(f"\n{'='*80}\nStarting walk-forward window {i+1}/{num_windows}\n{'='*80}")
        
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
        
        logger.info(f"Window {i+1} trading days: {start_day} to {end_day}")
        
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
        
        logger.info(f"Train period: {train_data.index[0]} to {train_data.index[-1]}")
        logger.info(f"Validation period: {validation_data.index[0]} to {validation_data.index[-1]}")
        logger.info(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
        
        # Save window periods
        window_periods = {
            "window": i+1,
            "train_start": train_data.index[0],
            "train_end": train_data.index[-1],
            "validation_start": validation_data.index[0],
            "validation_end": validation_data.index[-1],
            "test_start": test_data.index[0],
            "test_end": test_data.index[-1]
        }
        
        save_json(window_periods, f'{window_folder}/window_periods.json')
        
        # Create and train model
        model, training_stats = train_walk_forward_model(
            train_data, 
            validation_data,
            initial_timesteps=initial_timesteps,
            additional_timesteps=additional_timesteps,
            max_iterations=max_iterations,
            n_stagnant_loops=n_stagnant_loops,
            improvement_threshold=improvement_threshold,
            window_folder=window_folder
        )
        
        # Evaluate on test data
        # Get risk management configuration from config
        risk_config = config.get("risk_management", {})
        risk_enabled = risk_config.get("enabled", False)
        
        if risk_enabled:
            # Initialize risk parameters
            stop_loss_pct = None
            take_profit_pct = None
            trailing_stop_pct = None
            position_size = 1.0
            max_risk_per_trade_pct = 2.0
            
            # Apply risk management configuration
            # Stop loss configuration
            stop_loss_config = risk_config.get("stop_loss", {})
            if stop_loss_config.get("enabled", False):
                stop_loss_pct = stop_loss_config.get("percentage", 1.0)
            
            # Take profit configuration
            take_profit_config = risk_config.get("take_profit", {})
            if take_profit_config.get("enabled", False):
                take_profit_pct = take_profit_config.get("percentage", 2.0)
            
            # Trailing stop configuration
            trailing_stop_config = risk_config.get("trailing_stop", {})
            if trailing_stop_config.get("enabled", False):
                trailing_stop_pct = trailing_stop_config.get("percentage", 0.5)
            
            # Position sizing configuration
            position_sizing_config = risk_config.get("position_sizing", {})
            if position_sizing_config.get("enabled", False):
                position_size = position_sizing_config.get("size_multiplier", 1.0)
                max_risk_per_trade_pct = position_sizing_config.get("max_risk_per_trade_percentage", 2.0)
                logger.info(f"Position sizing enabled with multiplier {position_size} and max risk {max_risk_per_trade_pct}%")
            else:
                logger.info("  - Position sizing: Disabled")
            
            # Save the model temporarily for evaluation
            temp_model_path = f"{window_folder}/temp_test_model"
            model.save(temp_model_path)
            
            # Evaluate with risk management
            test_results = trade_with_risk_management(
                model_path=temp_model_path,
                test_data=test_data,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                trailing_stop_pct=trailing_stop_pct,
                position_size=position_size,
                max_risk_per_trade_pct=max_risk_per_trade_pct,
                initial_balance=config["environment"]["initial_balance"],
                transaction_cost=config["environment"].get("transaction_cost", 0.0),
                verbose=1,
                deterministic=True
            )
            
            # Clean up temporary model file
            if os.path.exists(f"{temp_model_path}.zip"):
                os.remove(f"{temp_model_path}.zip")
                
            # Calculate hit rate from trade results
            test_results = calculate_hit_rate_from_trade_results(test_results)
            
            logger.info(f"Evaluated model with risk management: stop_loss={stop_loss_pct}%, " +
                      f"take_profit={take_profit_pct}%, trailing_stop={trailing_stop_pct}%")
        else:
            # Use standard evaluation without risk management
            test_results = evaluate_agent(model, test_data, deterministic=True)
            logger.info("Evaluated model without risk management")
        
        # Log appropriate metrics based on evaluation metric
        if eval_metric == "hit_rate" and test_results["trade_count"] > 0:
            logger.info(f"Test Results - Hit Rate: {test_results['hit_rate']:.2f}% " +
                       f"({test_results['profitable_trades']}/{test_results['trade_count']} trades), " +
                       f"Portfolio: ${test_results['final_portfolio_value']:.2f}")
        else:
            logger.info(f"Test Results - Return: {test_results['total_return_pct']:.2f}%, " +
                      f"Portfolio: ${test_results['final_portfolio_value']:.2f}")
        
        # Save trade history to CSV if available
        if 'trade_history' in test_results and test_results['trade_history']:
            # Create CSV file path for this window's trade history
            trade_history_file = f'{window_folder}/test_trade_history.csv'
            
            # Save trade history
            save_trade_history(test_results['trade_history'], trade_history_file)
            logger.info(f"Saved test trade history to {trade_history_file}")
            
            # Create a summary of trade performance stats
            trade_summary = {}
            
            # Extract stop loss, take profit, and trailing stop info if available
            if risk_enabled:
                exit_reasons = test_results.get('exit_reasons', {})
                
                trade_summary['total_trades'] = test_results['trade_count']
                trade_summary['profitable_trades'] = test_results.get('profitable_trades', 0)
                trade_summary['hit_rate'] = test_results.get('hit_rate', 0)
                
                # Count trades by exit reason
                trade_summary['stop_loss_exits'] = exit_reasons.get('stop_loss', 0)
                trade_summary['take_profit_exits'] = exit_reasons.get('take_profit', 0)
                trade_summary['trailing_stop_exits'] = exit_reasons.get('trailing_stop', 0)
                trade_summary['model_signal_exits'] = exit_reasons.get('model_signal', 0)
                
                # Calculate percentages of exit reasons
                total_exits = sum(exit_reasons.values())
                if total_exits > 0:
                    trade_summary['stop_loss_pct'] = (exit_reasons.get('stop_loss', 0) / total_exits) * 100
                    trade_summary['take_profit_pct'] = (exit_reasons.get('take_profit', 0) / total_exits) * 100
                    trade_summary['trailing_stop_pct'] = (exit_reasons.get('trailing_stop', 0) / total_exits) * 100
                    trade_summary['model_signal_pct'] = (exit_reasons.get('model_signal', 0) / total_exits) * 100
                
                # Save trade summary
                save_json(trade_summary, f'{window_folder}/trade_summary.json')
                logger.info(f"Saved trade summary to {window_folder}/trade_summary.json")
        
        # Save results for this window
        test_results["window"] = i + 1
        test_results["train_start"] = train_data.index[0]
        test_results["train_end"] = train_data.index[-1]
        test_results["validation_start"] = validation_data.index[0]
        test_results["validation_end"] = validation_data.index[-1]
        test_results["test_start"] = test_data.index[0]
        test_results["test_end"] = test_data.index[-1]
        test_results["training_stats"] = training_stats
        
        all_window_results.append(test_results)
        
        # Save test results
        save_json(test_results, f'{window_folder}/test_results.json')
        
        # Save model for this window
        model.save(f"{window_folder}/model")
        logger.info(f"Saved model for window {i+1} to {window_folder}/model")
        
        # Plot window performance
        plot_window_performance(test_data, test_results, window_folder, i+1)
        
    # Aggregate results across all windows
    returns = [res["total_return_pct"] for res in all_window_results]
    portfolio_values = [res["final_portfolio_value"] for res in all_window_results]
    trade_counts = [res["trade_count"] for res in all_window_results]
    
    # Also aggregate hit rates if that metric is used
    hit_rates = [res.get("hit_rate", 0) for res in all_window_results]
    profitable_trades = [res.get("profitable_trades", 0) for res in all_window_results]
    
    # Calculate average metrics
    avg_return = np.mean(returns)
    avg_portfolio = np.mean(portfolio_values)
    avg_trades = np.mean(trade_counts)
    avg_hit_rate = np.mean(hit_rates) if hit_rates else 0
    avg_profitable_trades = np.mean(profitable_trades) if profitable_trades else 0
    
    logger.info(f"\n{'='*80}\nWalk-Forward Testing Summary\n{'='*80}")
    logger.info(f"Number of windows: {num_windows}")
    logger.info(f"Average return: {avg_return:.2f}%")
    
    if eval_metric == "hit_rate":
        logger.info(f"Average hit rate: {avg_hit_rate:.2f}%")
        logger.info(f"Average profitable trades: {avg_profitable_trades:.2f} out of {avg_trades:.2f}")
    
    logger.info(f"Average final portfolio: ${avg_portfolio:.2f}")
    logger.info(f"Average trade count: {avg_trades:.2f}")
    
    # Save summary results
    summary_results = {
        "avg_return": avg_return,
        "avg_hit_rate": avg_hit_rate,
        "avg_portfolio": avg_portfolio,
        "avg_trades": avg_trades,
        "avg_profitable_trades": avg_profitable_trades,
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

def train_walk_forward_model(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    initial_timesteps: int = 10000,
    additional_timesteps: int = 5000,
    max_iterations: int = 10,
    n_stagnant_loops: int = 3,
    improvement_threshold: float = 0.1,
    window_folder: str = None
) -> Tuple[PPO, List[Dict]]:
    """
    Train a model for a single walk-forward window.
    
    Args:
        train_data: Training data for this window
        validation_data: Validation data for this window
        initial_timesteps: Initial number of training timesteps
        additional_timesteps: Number of additional timesteps for each training iteration
        max_iterations: Maximum number of training iterations
        n_stagnant_loops: Number of consecutive iterations without improvement before stopping
        improvement_threshold: Minimum percentage improvement considered significant
        window_folder: Folder to save training statistics
        
    Returns:
        Tuple[PPO, List[Dict]]: Trained model and training statistics
    """
    # Initialize training environment
    train_env = TradingEnv(
        train_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=0.0,
        position_size=config["environment"].get("position_size", 1)
    )
    
    # Initialize model
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=0,
        ent_coef=config["model"].get("ent_coef", 0.01),
        learning_rate=config["model"].get("learning_rate", 0.0003),
        n_steps=config["model"].get("n_steps", 2048),
        batch_size=config["model"].get("batch_size", 64),
        gamma=0.99,
        gae_lambda=0.95,
    )
    
    # Initialize training statistics list
    training_stats = []
    
    # Get evaluation metric from config
    eval_metric = config.get("training", {}).get("evaluation", {}).get("metric", "return")
    hit_rate_min_trades = config.get("training", {}).get("evaluation", {}).get("hit_rate_min_trades", 5)
    
    # Log which metric we're using for evaluation
    if eval_metric == "hit_rate":
        logger.info(f"Using hit rate as evaluation metric with minimum {hit_rate_min_trades} trades")
    else:
        logger.info(f"Using return percentage as evaluation metric")
    
    # Get risk management configuration from config
    risk_config = config.get("risk_management", {})
    risk_enabled = risk_config.get("enabled", False)
    
    # Initialize risk parameters
    stop_loss_pct = None
    take_profit_pct = None
    trailing_stop_pct = None
    position_size = 1.0
    max_risk_per_trade_pct = 2.0
    
    # Apply risk management configuration if enabled
    if risk_enabled:
        logger.info("Risk management is enabled for model evaluation")
        # Stop loss configuration
        stop_loss_config = risk_config.get("stop_loss", {})
        if stop_loss_config.get("enabled", False):
            stop_loss_pct = stop_loss_config.get("percentage", 1.0)
            logger.info(f"Stop loss enabled at {stop_loss_pct}%")
        
        # Take profit configuration
        take_profit_config = risk_config.get("take_profit", {})
        if take_profit_config.get("enabled", False):
            take_profit_pct = take_profit_config.get("percentage", 2.0)
            logger.info(f"Take profit enabled at {take_profit_pct}%")
        
        # Trailing stop configuration
        trailing_stop_config = risk_config.get("trailing_stop", {})
        if trailing_stop_config.get("enabled", False):
            trailing_stop_pct = trailing_stop_config.get("percentage", 0.5)
            logger.info(f"Trailing stop enabled at {trailing_stop_pct}%")
        
        # Position sizing configuration
        position_sizing_config = risk_config.get("position_sizing", {})
        if position_sizing_config.get("enabled", False):
            position_size = position_sizing_config.get("size_multiplier", 1.0)
            max_risk_per_trade_pct = position_sizing_config.get("max_risk_per_trade_percentage", 2.0)
            logger.info(f"Position sizing enabled with multiplier {position_size} and max risk {max_risk_per_trade_pct}%")
        else:
            logger.info("  - Position sizing: Disabled")
    else:
        logger.info("Risk management is disabled for model evaluation")
    
    # Initial training
    logger.info(f"Starting initial training for {initial_timesteps} timesteps")
    model.learn(total_timesteps=initial_timesteps)
    
    # Evaluate initial model on validation data
    if risk_enabled and window_folder:
        # Save the model temporarily for evaluation with risk management
        temp_model_path = f"{window_folder}/temp_model_initial"
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
            deterministic=True
        )
        
        # Clean up temporary model file
        if os.path.exists(f"{temp_model_path}.zip"):
            os.remove(f"{temp_model_path}.zip")
        
        # Calculate hit rate from trade results
        results = calculate_hit_rate_from_trade_results(results)
    else:
        # Use standard evaluation without risk management
        results = evaluate_agent(model, validation_data, verbose=0, deterministic=True)
    
    # Log detailed trade statistics for debugging
    logger.info(f"Initial evaluation - Trade count: {results['trade_count']}, " +
               f"Profitable trades: {results.get('profitable_trades', 0)}, " +
               f"Hit rate: {results.get('hit_rate', 0):.2f}%")

    # Print a sample of trade history for debugging
    if 'trade_history' in results and results['trade_history']:
        sample_size = min(5, len(results['trade_history']))
        logger.info(f"Sample of first {sample_size} trades:")
        for i, trade in enumerate(results['trade_history'][:sample_size]):
            logger.info(f"Trade {i+1}: {trade.get('action', 'N/A')}, " +
                        f"Price: {trade.get('price', 'N/A')}, " +
                        f"Profit: {trade.get('profit', 'N/A')}")
    
    # Determine best metric value depending on config
    if eval_metric == "hit_rate" and results["trade_count"] >= hit_rate_min_trades:
        best_metric_value = results["hit_rate"]
        best_metric_name = "hit rate"
    else:
        # Default to return if hit_rate is selected but not enough trades, or if return is selected
        best_metric_value = results["total_return_pct"]
        best_metric_name = "return"
        if eval_metric == "hit_rate" and results["trade_count"] < hit_rate_min_trades:
            logger.warning(f"Not enough trades ({results['trade_count']}) for hit rate metric. Using return instead.")
    
    best_model = model
    
    # Record initial training statistics
    initial_stats = {
        "iteration": 0,
        "timesteps": initial_timesteps,
        "return_pct": results["total_return_pct"],
        "hit_rate": results.get("hit_rate", 0),
        "portfolio_value": results["final_portfolio_value"],
        "trade_count": results["trade_count"],
        "profitable_trades": results.get("profitable_trades", 0),
        "metric_used": best_metric_name,
        "metric_value": best_metric_value,
        "is_best": True
    }
    training_stats.append(initial_stats)
    
    logger.info(f"Initial validation {best_metric_name}: {best_metric_value:.2f}" + 
               (f"%" if best_metric_name == "return" else "% (profitable trades)"))
    
    # Counter for consecutive iterations without significant improvement
    stagnant_counter = 0
    
    # Continue training until max_iterations or n_stagnant_loops consecutive iterations without improvement
    for iteration in range(1, max_iterations + 1):
        # Train for additional timesteps
        model.learn(total_timesteps=additional_timesteps)
        
        # Evaluate on validation data
        if risk_enabled and window_folder:
            # Save the model temporarily for evaluation with risk management
            temp_model_path = f"{window_folder}/temp_model_iteration_{iteration}"
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
                deterministic=True
            )
            
            # Clean up temporary model file
            if os.path.exists(f"{temp_model_path}.zip"):
                os.remove(f"{temp_model_path}.zip")
            
            # Calculate hit rate from trade results
            results = calculate_hit_rate_from_trade_results(results)
        else:
            # Use standard evaluation without risk management
            results = evaluate_agent(model, validation_data, verbose=0, deterministic=True)
        
        # Log detailed trade statistics for debugging
        logger.info(f"Iteration {iteration} - Trade count: {results['trade_count']}, " +
                   f"Profitable trades: {results.get('profitable_trades', 0)}, " +
                   f"Hit rate: {results.get('hit_rate', 0):.2f}%")
        
        # Determine current metric value depending on config
        if eval_metric == "hit_rate" and results["trade_count"] >= hit_rate_min_trades:
            current_metric_value = results["hit_rate"]
            current_metric_name = "hit rate"
        else:
            # Default to return if hit_rate is selected but not enough trades
            current_metric_value = results["total_return_pct"]
            current_metric_name = "return"
            if eval_metric == "hit_rate" and results["trade_count"] < hit_rate_min_trades:
                logger.warning(f"Not enough trades ({results['trade_count']}) for hit rate metric. Using return instead.")
        
        # Calculate improvement
        improvement = current_metric_value - best_metric_value
        
        # Log different formats based on metric type
        if current_metric_name == "hit_rate":
            logger.info(f"Iteration {iteration} - Validation {current_metric_name}: {current_metric_value:.2f}% " +
                       f"({results['profitable_trades']}/{results['trade_count']} trades), " +
                       f"Improvement: {improvement:.2f}%")
        else:
            logger.info(f"Iteration {iteration} - Validation {current_metric_name}: {current_metric_value:.2f}%, " +
                       f"Improvement: {improvement:.2f}%")
        
        # Record training statistics
        is_best = current_metric_value > best_metric_value + improvement_threshold
        iteration_stats = {
            "iteration": iteration,
            "timesteps": additional_timesteps,
            "return_pct": results["total_return_pct"],
            "hit_rate": results.get("hit_rate", 0),
            "portfolio_value": results["final_portfolio_value"],
            "trade_count": results["trade_count"],
            "profitable_trades": results.get("profitable_trades", 0),
            "metric_used": current_metric_name,
            "metric_value": current_metric_value,
            "improvement": improvement,
            "is_best": is_best
        }
        training_stats.append(iteration_stats)
        
        # Check if this is the best model so far
        if is_best:
            best_metric_value = current_metric_value
            best_metric_name = current_metric_name
            best_model = model
            
            # Format the log message based on the metric
            if best_metric_name == "hit_rate":
                logger.info(f"New best model found! Validation {best_metric_name}: {best_metric_value:.2f}% " +
                           f"({results['profitable_trades']}/{results['trade_count']} trades)")
            else:
                logger.info(f"New best model found! Validation {best_metric_name}: {best_metric_value:.2f}%")
            
            # Save intermediate best model
            if window_folder:
                best_model.save(f"{window_folder}/best_model_iteration_{iteration}")
                logger.info(f"Saved best model at iteration {iteration}")
            
            # Reset stagnant counter since we found improvement
            stagnant_counter = 0
        else:
            # Increment stagnant counter if no significant improvement
            stagnant_counter += 1
            logger.info(f"No significant improvement. Stagnant iterations: {stagnant_counter}/{n_stagnant_loops}")
        
        # Stop if we've had n_stagnant_loops consecutive iterations without improvement
        if stagnant_counter >= n_stagnant_loops:
            logger.info(f"Stopping training after {n_stagnant_loops} consecutive iterations without significant improvement")
            break
    
    logger.info(f"Training completed. Best validation {best_metric_name}: {best_metric_value:.2f}" + 
               (f"%" if best_metric_name == "return" else "% (profitable trades)"))
    
    # Plot training progress
    if window_folder:
        plot_training_progress(training_stats, window_folder)
        
        # Save training statistics
        save_json(training_stats, f'{window_folder}/training_stats.json')
    
    # Final evaluation for logging trade examples
    if risk_enabled and window_folder:
        # Save the model temporarily for evaluation with risk management
        temp_model_path = f"{window_folder}/temp_best_model_final"
        best_model.save(temp_model_path)
        
        # Evaluate with risk management
        final_results = trade_with_risk_management(
            model_path=temp_model_path,
            test_data=validation_data,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trailing_stop_pct=trailing_stop_pct,
            position_size=position_size,
            max_risk_per_trade_pct=max_risk_per_trade_pct,
            initial_balance=config["environment"]["initial_balance"],
            transaction_cost=config["environment"].get("transaction_cost", 0.0),
            verbose=1,
            deterministic=True
        )
        
        # Clean up temporary model file
        if os.path.exists(f"{temp_model_path}.zip"):
            os.remove(f"{temp_model_path}.zip")
        
        # Calculate hit rate from final results
        final_results = calculate_hit_rate_from_trade_results(final_results)
    else:
        # Use standard evaluation without risk management
        final_results = evaluate_agent(best_model, validation_data, verbose=1, deterministic=True)
    
    # Save trade history to CSV
    if 'trade_history' in final_results and final_results['trade_history']:
        # Convert to DataFrame and save
        validation_trade_history_path = f'{window_folder}/validation_trade_history.csv'
        save_trade_history(final_results['trade_history'], validation_trade_history_path)
        logger.info(f"Saved validation trade history to {validation_trade_history_path}")
        
        # Create a summary of validation trade performance stats
        if risk_enabled and window_folder:
            validation_trade_summary = {}
            exit_reasons = final_results.get('exit_reasons', {})
            
            validation_trade_summary['total_trades'] = final_results['trade_count']
            validation_trade_summary['profitable_trades'] = final_results.get('profitable_trades', 0)
            validation_trade_summary['hit_rate'] = final_results.get('hit_rate', 0)
            
            # Count trades by exit reason
            validation_trade_summary['stop_loss_exits'] = exit_reasons.get('stop_loss', 0)
            validation_trade_summary['take_profit_exits'] = exit_reasons.get('take_profit', 0)
            validation_trade_summary['trailing_stop_exits'] = exit_reasons.get('trailing_stop', 0)
            validation_trade_summary['model_signal_exits'] = exit_reasons.get('model_signal', 0)
            
            # Calculate percentages of exit reasons
            total_exits = sum(exit_reasons.values())
            if total_exits > 0:
                validation_trade_summary['stop_loss_pct'] = (exit_reasons.get('stop_loss', 0) / total_exits) * 100
                validation_trade_summary['take_profit_pct'] = (exit_reasons.get('take_profit', 0) / total_exits) * 100
                validation_trade_summary['trailing_stop_pct'] = (exit_reasons.get('trailing_stop', 0) / total_exits) * 100
                validation_trade_summary['model_signal_pct'] = (exit_reasons.get('model_signal', 0) / total_exits) * 100
            
            # Save validation trade summary
            save_json(validation_trade_summary, f'{window_folder}/validation_trade_summary.json')
            logger.info(f"Saved validation trade summary to {window_folder}/validation_trade_summary.json")
    
    return best_model, training_stats

def plot_training_progress(training_stats: List[Dict], window_folder: str) -> None:
    """
    Plot the training progress for a window.
    
    Args:
        training_stats: List of training statistics dictionaries
        window_folder: Folder to save the plot
    """
    iterations = [stat["iteration"] for stat in training_stats]
    is_best = [stat["is_best"] for stat in training_stats]
    
    # Determine which metric to plot
    # Get the metric used in the most recent iteration
    if training_stats:
        metric_name = training_stats[-1].get("metric_used", "return")
    else:
        metric_name = "return"  # Default if no stats
    
    if metric_name == "hit rate":
        metric_values = [stat["hit_rate"] for stat in training_stats]
        y_label = 'Hit Rate (%)'
        title_prefix = 'Hit Rate'
    else:
        metric_values = [stat["return_pct"] for stat in training_stats]
        y_label = 'Return (%)'
        title_prefix = 'Return'
    
    plt.figure(figsize=(10, 6))
    
    # Plot metric values
    plt.plot(iterations, metric_values, marker='o', linestyle='-', color='blue', 
             label=f'Validation {metric_name.title()}')
    
    # Highlight best models
    best_iterations = [iterations[i] for i in range(len(iterations)) if is_best[i]]
    best_metrics = [metric_values[i] for i in range(len(metric_values)) if is_best[i]]
    plt.scatter(best_iterations, best_metrics, color='green', s=100, marker='*', label='Best Model')
    
    plt.xlabel('Training Iteration')
    plt.ylabel(y_label)
    plt.title(f'Training Progress - {title_prefix}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{window_folder}/training_progress.png')
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
    
    plt.figure(figsize=(12, 8))
    
    # Plot portfolio value
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_history, color='blue', label='Portfolio Value')
    plt.title(f'Window {window_num} Test Performance')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot price and trades if available
    if 'close' in test_data.columns and len(action_history) > 0:
        plt.subplot(2, 1, 2)
        plt.plot(test_data.index, test_data['close'], color='gray', label='Price')
        
        # Plot buy and sell points
        buy_indices = [i for i, action in enumerate(action_history) if action == 1]
        sell_indices = [i for i, action in enumerate(action_history) if action == 2]
        
        if buy_indices:
            buy_dates = [test_data.index[i] for i in buy_indices if i < len(test_data)]
            buy_prices = [test_data['close'].iloc[i] for i in buy_indices if i < len(test_data)]
            plt.scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='Buy')
            
        if sell_indices:
            sell_dates = [test_data.index[i] for i in sell_indices if i < len(test_data)]
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
    returns = [res["total_return_pct"] for res in all_window_results]
    portfolio_values = [res["final_portfolio_value"] for res in all_window_results]
    trade_counts = [res["trade_count"] for res in all_window_results]
    
    # Also get hit rates if that metric is used
    hit_rates = [res.get("hit_rate", 0) for res in all_window_results]
    profitable_trades = [res.get("profitable_trades", 0) for res in all_window_results]
    
    # Number of subplots depends on which metric is being used
    num_plots = 4 if eval_metric == "hit_rate" else 3
    
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
    
    plt.xlabel('Walk-Forward Window')
    plt.ylabel('Cumulative Return (%)', color='blue')
    plt.tick_params(axis='y', labelcolor='blue')
    plt.title('Cumulative Performance Across Walk-Forward Windows')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Create legend that incorporates both axes
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    if eval_metric == "hit_rate":
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
    
    # Load all data
    train_data, validation_data, test_data = get_data(
        symbol=config["data"]["symbol"],
        period=config["data"]["period"],
        interval=config["data"]["interval"],
        train_ratio=0.0,  # Don't split data here, we'll do it in walk-forward testing
        validation_ratio=0.0
    )
    
    # Check if data loading was successful
    if train_data is None or len(train_data) == 0:
        logger.error("Failed to load data or dataset is empty. Check your data file and configuration.")
        print("\nERROR: Data loading failed. Please check the data file and ensure it contains valid data.")
        return
    
    full_data = train_data  # Just get the first part (full dataset)
    logger.info(f"Loaded dataset with {len(full_data)} rows spanning from {full_data.index[0]} to {full_data.index[-1]}")
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
            logger.info(f"  - Trailing stop: {trailing_stop_pct}%")
        else:
            logger.info("  - Trailing stop: Disabled")
        
        # Position sizing configuration
        position_sizing_config = risk_config.get("position_sizing", {})
        if position_sizing_config.get("enabled", False):
            position_size = position_sizing_config.get("size_multiplier", 1.0)
            max_risk_per_trade_pct = position_sizing_config.get("max_risk_per_trade_percentage", 2.0)
            logger.info(f"Position sizing enabled with multiplier {position_size} and max risk {max_risk_per_trade_pct}%")
        else:
            logger.info("  - Position sizing: Disabled")
    else:
        logger.info("Risk management is DISABLED for walk-forward testing")
    
    # Run walk-forward testing
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
        improvement_threshold=config["training"].get("improvement_threshold", 0.1)
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
        
        # Position sizing
        if position_sizing_config.get("enabled", False):
            print(f"  Position Sizing: Multiplier {position_size}, Max Risk {max_risk_per_trade_pct}%")
        else:
            print("  Position Sizing: Disabled")
    
    print("Note: All references to 'days' now indicate trading days (NYSE business days), not calendar days")
    print("Note: Training was performed only on NYSE market hours data (9:30 AM to 4:00 PM ET, Monday to Friday)")

if __name__ == "__main__":
    main() 