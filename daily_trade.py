import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import os
from typing import Tuple, Dict, List
from dateutil.relativedelta import relativedelta

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from environment import TradingEnv
from config import config
import money
from trade import trade_with_risk_management, plot_results, save_trade_history
from get_data import filter_market_hours, get_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators and prepare data for training.
    
    Args:
        df: Raw DataFrame with OHLCV data
        
    Returns:
        DataFrame: Processed data with technical indicators
    """
    # Call get_data with use_yfinance=True to download fresh data from Yahoo Finance
    train_data, validation_data, test_data = get_data(
        train_ratio=1.0,  # Return all data as training data
        validation_ratio=0.0,  # No validation split needed here
        use_yfinance=True  # Use Yahoo Finance directly
    )
    
    # If data is None, something went wrong
    if train_data is None:
        logger.error("Failed to process data")
        return None
        
    return train_data

def split_today_and_training_window(data: pd.DataFrame, window_days: int = 59, 
                                   train_ratio: float = 0.7, 
                                   validation_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training, validation, and test data where test data is one day after validation data.
    
    Args:
        data: Processed DataFrame with DateTime index
        window_days: Number of days to use for training and validation window
        train_ratio: Proportion of window to use for training
        validation_ratio: Proportion of window to use for validation
        
    Returns:
        tuple: (training_data, validation_data, test_data)
    """
    # Get the most recent date in the data
    last_date = data.index.max()
    
    # Calculate the start of the training window
    window_start = last_date - timedelta(days=window_days)
    
    # Get historical data for the window
    historical_data = data[data.index >= window_start].copy()
    
    # Split historical data into training, validation and test by date
    # First determine size of training and validation periods in terms of data points
    historical_size = len(historical_data)
    train_size = int(historical_size * train_ratio)
    validation_size = int(historical_size * validation_ratio)
    
    # Get the data for each period
    train_data = historical_data.iloc[:train_size].copy()
    validation_data = historical_data.iloc[train_size:train_size + validation_size].copy()
    
    # Get the end date of validation period
    validation_end_date = validation_data.index.max()
    
    # Get all data for the day after validation ends as test data
    # First get the next calendar day
    next_day = validation_end_date + timedelta(days=1)
    # Get the data from that day
    test_data = data[data.index.date >= next_day.date()].copy()
    
    # If test data is empty (no data for the next day), try to find the next available day
    if len(test_data) == 0:
        remaining_data = data[data.index > validation_end_date].copy()
        if len(remaining_data) > 0:
            # Get the first available day after validation
            next_available_date = remaining_data.index.min().date()
            test_data = data[data.index.date == next_available_date].copy()
            logger.info(f"No data found exactly one day after validation, using next available day: {next_available_date}")
    
    # In case there's still no test data, use the most recent day
    if len(test_data) == 0:
        logger.warning("No test data found after validation period. Using most recent day instead.")
        most_recent_date = data.index.max().date()
        test_data = data[data.index.date == most_recent_date].copy()
    
    logger.info(f"Created training window with {len(train_data)} rows from {train_data.index.min()} to {train_data.index.max()}")
    logger.info(f"Created validation window with {len(validation_data)} rows from {validation_data.index.min()} to {validation_data.index.max()}")
    logger.info(f"Test data has {len(test_data)} rows from {test_data.index.min()} to {test_data.index.max()}")
    
    return train_data, validation_data, test_data

def train_model(train_data: pd.DataFrame, validation_data: pd.DataFrame,
               initial_timesteps: int = None,
               additional_timesteps: int = None,
               max_iterations: int = None,
               n_stagnant_loops: int = None,
               improvement_threshold: float = None) -> Tuple[PPO, List[Dict], str]:
    """
    Train a PPO model on the training data with iterative validation.
    
    Args:
        train_data: Training data with features
        validation_data: Validation data with features
        initial_timesteps: Initial number of training timesteps
        additional_timesteps: Additional timesteps for each training iteration
        max_iterations: Maximum number of training iterations
        n_stagnant_loops: Number of consecutive iterations without improvement before stopping
        improvement_threshold: Minimum percentage improvement considered significant
        
    Returns:
        Tuple[PPO, List[Dict], str]: Best trained model based on validation performance and training statistics and model path
    """
    # Get training parameters from config
    if initial_timesteps is None:
        initial_timesteps = config["training"].get("total_timesteps", 10000)
    if additional_timesteps is None:
        additional_timesteps = config["training"].get("additional_timesteps", 5000)
    if max_iterations is None:
        max_iterations = config["training"].get("max_iterations", 10)
    if n_stagnant_loops is None:
        n_stagnant_loops = config["training"].get("n_stagnant_loops", 3)
    if improvement_threshold is None:
        improvement_threshold = config["training"].get("improvement_threshold", 0.1)
    
    # Create model folder if it doesn't exist
    os.makedirs('models/daily', exist_ok=True)
    
    # Initialize training environment
    train_env = TradingEnv(
        train_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=0.0,
        position_size=config["environment"].get("position_size", 1)
    )
    
    # Validate environment
    check_env(train_env, skip_render_check=True)
    
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
    
    # Save initial model for validation
    initial_model_path = "models/daily/initial_model"
    model.save(initial_model_path)
    
    # Evaluate initial model on validation data
    results = trade_with_risk_management(
        model_path=initial_model_path,
        test_data=validation_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 0.0),
        verbose=0,
        deterministic=True
    )
    
    best_return = results["total_return_pct"]
    best_model = model
    best_model_path = initial_model_path
    
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
        
        # Save current iteration model
        current_model_path = f"models/daily/model_iteration_{iteration}"
        model.save(current_model_path)
        
        # Evaluate on validation data
        results = trade_with_risk_management(
            model_path=current_model_path,
            test_data=validation_data,
            initial_balance=config["environment"]["initial_balance"],
            transaction_cost=config["environment"].get("transaction_cost", 0.0),
            verbose=0,
            deterministic=True
        )
        
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
            best_model_path = current_model_path
            logger.info(f"New best model found! Validation return: {best_return:.2f}%")
            
            # Save intermediate best model
            best_model.save(f"models/daily/best_model_iteration_{iteration}")
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
    
    # Save the final best model to the standard location
    final_model_path = "models/daily_model"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    
    # Copy the best model to the final location
    import shutil
    shutil.copy2(best_model_path + ".zip", final_model_path + ".zip")
    logger.info(f"Best model copied from {best_model_path} to {final_model_path}")
    
    # Save training statistics to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('models/logs', exist_ok=True)
    training_log = f"models/logs/daily_training_{timestamp}.log"
    
    try:
        with open(training_log, 'w') as f:
            import json
            json.dump(training_stats, f, indent=4, default=str)
        logger.info(f"Training statistics saved to {training_log}")
    except Exception as e:
        logger.error(f"Error saving training statistics: {e}")
    
    # Return the best model, training statistics, and path to best model
    return best_model, training_stats, final_model_path

def execute_test_trade(model: PPO, model_path: str, test_data: pd.DataFrame) -> Dict:
    """
    Execute a test trade on the test data using the trained model with risk management.
    
    Args:
        model: Trained PPO model (kept for compatibility)
        model_path: Path to the saved model file
        test_data: Test market data
        
    Returns:
        Dict: Trade results
    """
    # Get risk management configuration from config.yaml
    risk_config = config.get("risk_management", {})
    risk_enabled = risk_config.get("enabled", True)
    
    # Initialize risk parameters
    stop_loss_pct = None
    take_profit_pct = None
    trailing_stop_pct = None
    position_size = 1.0
    max_risk_per_trade_pct = 2.0
    
    # Apply risk management configuration if enabled
    if risk_enabled:
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
    
    logger.info(f"Executing test trade with risk management settings:")
    logger.info(f"  Stop Loss: {stop_loss_pct}%")
    logger.info(f"  Take Profit: {take_profit_pct}%")
    logger.info(f"  Trailing Stop: {trailing_stop_pct}%")
    logger.info(f"  Position Size: {position_size}")
    logger.info(f"  Close at End of Day: True")
    
    # Run trade with risk management
    results = trade_with_risk_management(
        model_path=model_path,
        test_data=test_data,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_stop_pct=trailing_stop_pct,
        position_size=position_size,
        max_risk_per_trade_pct=max_risk_per_trade_pct,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 0.0),
        verbose=1,
        deterministic=True,
        close_at_end_of_day=True  # Always close positions at the end of trading day
    )
    
    return results

def plot_training_progress(training_stats: List[Dict]) -> None:
    """
    Plot the training progress.
    
    Args:
        training_stats: List of training statistics dictionaries
    """
    import matplotlib.pyplot as plt
    
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
    plt.title('Daily Training Progress')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.tight_layout()
    plt.savefig(f'plots/daily_training_progress_{timestamp}.png')
    plt.close()
    
    logger.info(f"Training progress plot saved to plots/daily_training_progress_{timestamp}.png")

def main():
    """
    Main function to run the daily trading process.
    """
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/daily', exist_ok=True)
    os.makedirs('models/logs', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Record session timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info(f"Starting daily trading session at {timestamp}")
    
    # Process data with technical indicators directly from Yahoo Finance
    processed_data = prepare_data_for_training(None)  # None because we're downloading directly
    
    if processed_data is None:
        logger.error("Failed to process data. Exiting.")
        return
    
    # Filter to market hours if configured
    if config["data"].get("market_hours_only", True):
        try:
            # Let's import the filter_market_hours function from walk_forward
            from walk_forward import filter_market_hours
            processed_data = filter_market_hours(processed_data)
        except Exception as e:
            logger.error(f"Error filtering market hours: {e}")
    
    # Split into training, validation, and test data
    train_data, validation_data, test_data = split_today_and_training_window(
        processed_data, 
        window_days=config["walk_forward"].get("window_size", 14),
        train_ratio=config["data"].get("train_ratio", 0.7),
        validation_ratio=config["data"].get("validation_ratio", 0.15)
    )
    
    if len(test_data) == 0:
        logger.error("No test data available. Exiting.")
        return
    
    if len(train_data) == 0:
        logger.error("No training data available. Exiting.")
        return
    
    if len(validation_data) == 0:
        logger.error("No validation data available. Exiting.")
        return
    
    # Get iterative training parameters from config
    training_config = config.get("training", {})
    initial_timesteps = training_config.get("total_timesteps", 10000)
    additional_timesteps = training_config.get("additional_timesteps", 5000)
    max_iterations = training_config.get("max_iterations", 10)
    n_stagnant_loops = training_config.get("n_stagnant_loops", 3)
    improvement_threshold = training_config.get("improvement_threshold", 0.1)
    
    # Train model on the window with iterative validation
    logger.info(f"Starting iterative training with initial {initial_timesteps} timesteps and max {max_iterations} iterations")
    model, training_stats, best_model_path = train_model(
        train_data,
        validation_data,
        initial_timesteps=initial_timesteps,
        additional_timesteps=additional_timesteps,
        max_iterations=max_iterations,
        n_stagnant_loops=n_stagnant_loops,
        improvement_threshold=improvement_threshold
    )
    
    # Plot training progress
    plot_training_progress(training_stats)
    
    # Execute test trade using the best model
    logger.info("Running trade on test data using the best model")
    results = execute_test_trade(model, best_model_path, test_data)
    
    # Plot results
    plot_results(results)
    
    # Save trade history
    trade_history_file = f"daily_trade_history_{timestamp}.csv"
    save_trade_history(results["trade_history"], trade_history_file)
    logger.info(f"Trade history saved to {trade_history_file}")
    
    # Print summary of results
    logger.info(f"\n{'='*80}\nDaily Trade Results\n{'='*80}")
    logger.info(f"Final Portfolio Value: ${results['final_portfolio_value']:.2f}")
    logger.info(f"Total Return: {results['total_return_pct']:.2f}%")
    logger.info(f"Total Trades: {results['trade_count']}")
    logger.info(f"Final Position: {results['final_position']}")
    
    # Print a warning if no trades were made
    if results['trade_count'] == 0:
        logger.warning("No trades were executed in test period. The model may need more training data or different parameters.")
    
    logger.info(f"{'='*80}\nDaily trading session completed at {datetime.now().strftime('%Y%m%d_%H%M%S')}\n{'='*80}")

if __name__ == "__main__":
    main() 