import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from environment import TradingEnv
from get_data import get_data
from train import evaluate_agent, plot_results
from config import config
import money

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    # Check if data is empty
    if data is None or len(data) == 0:
        logger.error("Empty dataset provided for walk-forward testing")
        return {
            "all_window_results": [],
            "avg_return": 0,
            "avg_portfolio": 0,
            "avg_trades": 0,
            "num_windows": 0,
            "error": "Empty dataset"
        }
    
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.error("Data index is not a DatetimeIndex, cannot perform walk-forward testing")
        return {
            "all_window_results": [],
            "avg_return": 0,
            "avg_portfolio": 0,
            "avg_trades": 0,
            "num_windows": 0,
            "error": "Invalid index type"
        }
    
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
        logger.error(f"Error calculating samples per day: {e}. Using default value of 288 (5-minute intervals).")
        samples_per_day = 288  # Default for 5-minute data (288 samples per day)
    
    # Convert window_size and step_size from days to samples
    window_samples = window_size * samples_per_day
    step_samples = step_size * samples_per_day
    
    # Number of walk-forward windows
    num_windows = (len(data) - window_samples) // step_samples + 1
    
    # Store results for each window
    all_window_results = []
    
    # Walk-forward testing
    for i in range(num_windows):
        logger.info(f"\n{'='*80}\nStarting walk-forward window {i+1}/{num_windows}\n{'='*80}")
        
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
        
        # Create and train model
        model = train_walk_forward_model(
            train_data, 
            validation_data,
            initial_timesteps=initial_timesteps,
            additional_timesteps=additional_timesteps,
            max_iterations=max_iterations,
            n_stagnant_loops=n_stagnant_loops,
            improvement_threshold=improvement_threshold
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
        
        all_window_results.append(test_results)
        
        # Save model for this window
        model.save(f"walk_forward_model_{i+1}")
        
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
    
    # Plot results
    plot_walk_forward_results(all_window_results)
    
    return {
        "all_window_results": all_window_results,
        "avg_return": avg_return,
        "avg_portfolio": avg_portfolio,
        "avg_trades": avg_trades,
        "num_windows": num_windows
    }

def train_walk_forward_model(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    initial_timesteps: int = 10000,
    additional_timesteps: int = 5000,
    max_iterations: int = 10,
    n_stagnant_loops: int = 3,
    improvement_threshold: float = 0.1
) -> PPO:
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
        
    Returns:
        PPO: Trained model
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
    
    # Initial training
    logger.info(f"Starting initial training for {initial_timesteps} timesteps")
    model.learn(total_timesteps=initial_timesteps)
    
    # Evaluate initial model on validation data
    results = evaluate_agent(model, validation_data, verbose=0, deterministic=True)
    best_return = results["total_return_pct"]
    best_model = model
    
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
        
        # Check if this is the best model so far
        if current_return > best_return + improvement_threshold:
            best_return = current_return
            best_model = model
            logger.info(f"New best model found! Validation return: {best_return:.2f}%")
            
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
    return best_model

def plot_walk_forward_results(all_window_results: List[Dict]) -> None:
    """
    Plot the results of walk-forward testing.
    
    Args:
        all_window_results: List of results dictionaries from each walk-forward window
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
    plt.savefig('walk_forward_results.png')
    plt.show()

def main():
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

if __name__ == "__main__":
    main() 