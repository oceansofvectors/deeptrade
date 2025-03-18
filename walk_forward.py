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
        window_size: Size of each walk-forward window in days
        step_size: Number of days to step forward at each iteration
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
    
    # Calculate number of samples per day safely
    try:
        time_span = (data.index[-1] - data.index[0]).days
        if time_span <= 0:
            # If data spans less than a day, use number of hours instead
            hours_span = (data.index[-1] - data.index[0]).total_seconds() / 3600
            if hours_span <= 0:
                samples_per_day = len(data)  # Use all samples if time span is too small
            else:
                samples_per_day = int(len(data) / hours_span * 24)  # Convert hours to days
        else:
            samples_per_day = int(len(data) / time_span)
            
        logger.info(f"Calculated {samples_per_day} samples per day")
    except (IndexError, ZeroDivisionError, AttributeError) as e:
        logger.error(f"Error calculating samples per day: {e}. Using default value of 78 (NYSE trading period 6.5 hours with 5-minute intervals).")
        samples_per_day = 78  # Default for 5-minute data in NYSE hours (78 samples per day)
    
    # Convert window_size and step_size from days to samples
    window_samples = window_size * samples_per_day
    step_samples = step_size * samples_per_day
    
    # Number of walk-forward windows
    num_windows = (len(data) - window_samples) // step_samples + 1
    
    # Store results for each window
    all_window_results = []
    
    # Save session parameters
    session_params = {
        "timestamp": timestamp,
        "window_size_days": window_size,
        "step_size_days": step_size,
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
        "samples_per_day": samples_per_day,
        "market_hours_only": True
    }
    
    save_json(session_params, f'{session_folder}/reports/session_parameters.json')
    
    # Walk-forward testing
    for i in range(num_windows):
        logger.info(f"\n{'='*80}\nStarting walk-forward window {i+1}/{num_windows}\n{'='*80}")
        
        # Create window folder
        window_folder = f'{session_folder}/models/window_{i+1}'
        os.makedirs(window_folder, exist_ok=True)
        
        # Extract data for this window
        start_idx = i * step_samples
        end_idx = start_idx + window_samples
        window_data = data.iloc[start_idx:end_idx].copy()
        
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
        test_results = evaluate_agent(model, test_data, deterministic=True)
        logger.info(f"Test Results - Return: {test_results['total_return_pct']:.2f}%, "
                   f"Portfolio: ${test_results['final_portfolio_value']:.2f}")
        
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
    
    # Calculate average metrics
    avg_return = np.mean(returns)
    avg_portfolio = np.mean(portfolio_values)
    avg_trades = np.mean(trade_counts)
    
    logger.info(f"\n{'='*80}\nWalk-Forward Testing Summary\n{'='*80}")
    logger.info(f"Number of windows: {num_windows}")
    logger.info(f"Average return: {avg_return:.2f}%")
    logger.info(f"Average final portfolio: ${avg_portfolio:.2f}")
    logger.info(f"Average trade count: {avg_trades:.2f}")
    
    # Save summary results
    summary_results = {
        "avg_return": avg_return,
        "avg_portfolio": avg_portfolio,
        "avg_trades": avg_trades,
        "num_windows": num_windows,
        "timestamp": timestamp
    }
    
    # Don't include all window results in the summary JSON (they may contain non-serializable objects)
    save_json(summary_results, f'{session_folder}/reports/summary_results.json')
    
    # Add window results back for the return value (not for JSON serialization)
    summary_results["all_window_results"] = all_window_results
    
    # Plot results
    plot_walk_forward_results(all_window_results, session_folder)
    
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
    
    # Initial training
    logger.info(f"Starting initial training for {initial_timesteps} timesteps")
    model.learn(total_timesteps=initial_timesteps)
    
    # Evaluate initial model on validation data
    results = evaluate_agent(model, validation_data, verbose=0, deterministic=True)
    best_return = results["total_return_pct"]
    best_model = model
    
    # Record initial training statistics
    initial_stats = {
        "iteration": 0,
        "timesteps": initial_timesteps,
        "return_pct": best_return,
        "portfolio_value": results["final_portfolio_value"],
        "trade_count": results["trade_count"],
        "is_best": True
    }
    training_stats.append(initial_stats)
    
    logger.info(f"Initial validation return: {best_return:.2f}%")
    
    # Counter for consecutive iterations without significant improvement
    stagnant_counter = 0
    
    # Continue training until max_iterations or n_stagnant_loops consecutive iterations without improvement
    for iteration in range(1, max_iterations + 1):
        # Train for additional timesteps
        model.learn(total_timesteps=additional_timesteps)
        
        # Evaluate on validation data
        results = evaluate_agent(model, validation_data, verbose=0, deterministic=True)
        current_return = results["total_return_pct"]
        
        # Calculate improvement
        improvement = current_return - best_return
        logger.info(f"Iteration {iteration} - Validation return: {current_return:.2f}%, Improvement: {improvement:.2f}%")
        
        # Record training statistics
        is_best = current_return > best_return + improvement_threshold
        iteration_stats = {
            "iteration": iteration,
            "timesteps": additional_timesteps,
            "return_pct": current_return,
            "portfolio_value": results["final_portfolio_value"],
            "trade_count": results["trade_count"],
            "improvement": improvement,
            "is_best": is_best
        }
        training_stats.append(iteration_stats)
        
        # Check if this is the best model so far
        if is_best:
            best_return = current_return
            best_model = model
            logger.info(f"New best model found! Validation return: {best_return:.2f}%")
            
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
    
    logger.info(f"Training completed. Best validation return: {best_return:.2f}%")
    
    # Plot training progress
    if window_folder:
        plot_training_progress(training_stats, window_folder)
        
        # Save training statistics
        save_json(training_stats, f'{window_folder}/training_stats.json')
    
    return best_model, training_stats

def plot_training_progress(training_stats: List[Dict], window_folder: str) -> None:
    """
    Plot the training progress for a window.
    
    Args:
        training_stats: List of training statistics dictionaries
        window_folder: Folder to save the plot
    """
    iterations = [stat["iteration"] for stat in training_stats]
    returns = [stat["return_pct"] for stat in training_stats]
    is_best = [stat["is_best"] for stat in training_stats]
    
    plt.figure(figsize=(10, 6))
    
    # Plot returns
    plt.plot(iterations, returns, marker='o', linestyle='-', color='blue', label='Validation Return')
    
    # Highlight best models
    best_iterations = [iterations[i] for i in range(len(iterations)) if is_best[i]]
    best_returns = [returns[i] for i in range(len(returns)) if is_best[i]]
    plt.scatter(best_iterations, best_returns, color='green', s=100, marker='*', label='Best Model')
    
    plt.xlabel('Training Iteration')
    plt.ylabel('Validation Return (%)')
    plt.title('Training Progress')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{window_folder}/training_progress.png')
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

def plot_walk_forward_results(all_window_results: List[Dict], session_folder: str) -> None:
    """
    Plot the results of walk-forward testing.
    
    Args:
        all_window_results: List of results dictionaries from each walk-forward window
        session_folder: Folder to save the plot
    """
    windows = [res["window"] for res in all_window_results]
    returns = [res["total_return_pct"] for res in all_window_results]
    portfolio_values = [res["final_portfolio_value"] for res in all_window_results]
    trade_counts = [res["trade_count"] for res in all_window_results]
    
    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot returns
    axs[0].bar(windows, returns, color='blue')
    axs[0].set_ylabel('Return (%)')
    axs[0].set_title('Returns by Walk-Forward Window')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot portfolio values
    axs[1].bar(windows, portfolio_values, color='green')
    axs[1].set_ylabel('Final Portfolio Value ($)')
    axs[1].set_title('Final Portfolio Values by Walk-Forward Window')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot trade counts
    axs[2].bar(windows, trade_counts, color='red')
    axs[2].set_xlabel('Walk-Forward Window')
    axs[2].set_ylabel('Trade Count')
    axs[2].set_title('Trade Counts by Walk-Forward Window')
    axs[2].grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'{session_folder}/plots/walk_forward_results.png')
    plt.close()
    
    # Also create and save a cumulative return plot
    plt.figure(figsize=(12, 6))
    cumulative_returns = np.cumsum(returns)
    plt.plot(windows, cumulative_returns, marker='o', linestyle='-', color='blue')
    plt.xlabel('Walk-Forward Window')
    plt.ylabel('Cumulative Return (%)')
    plt.title('Cumulative Returns Across Walk-Forward Windows')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{session_folder}/plots/cumulative_returns.png')
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
    window_size = wf_config.get("window_size", 14)  # 14 days default
    step_size = wf_config.get("step_size", 7)       # 7 days default
    
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
    print(f"Average final portfolio: ${results['avg_portfolio']:.2f}")
    print(f"Average trade count: {results['avg_trades']:.2f}")
    print(f"Results saved to models/session_{results['timestamp']}")
    print("Note: Training was performed only on NYSE market hours data (9:30 AM to 4:00 PM ET, Monday to Friday)")

if __name__ == "__main__":
    main() 