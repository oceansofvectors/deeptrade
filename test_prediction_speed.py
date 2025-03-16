import time
import logging
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from environment import TradingEnv
from get_data import get_data
from config import config
import yaml
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def test_prediction_speed(model_path, test_data, num_predictions=1000):
    """
    Test the prediction speed of a model.
    
    Args:
        model_path (str): Path to the model file.
        test_data (pd.DataFrame): Test data for predictions.
        num_predictions (int): Number of predictions to make.
        
    Returns:
        dict: Dictionary with timing results.
    """
    # Load the model
    logger.info(f"Loading model from {model_path}")
    start_time = time.time()
    model = PPO.load(model_path)
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.4f} seconds")
    
    # Create environment
    env = TradingEnv(
        test_data,
        initial_balance=config["environment"]["initial_balance"],
        position_size=config["environment"].get("position_size", 1)
    )
    
    # Reset environment
    obs, _ = env.reset()
    
    # Warm-up prediction (first prediction can be slower)
    model.predict(obs, deterministic=True)
    
    # Test prediction speed
    logger.info(f"Testing prediction speed with {num_predictions} predictions")
    start_time = time.time()
    
    for _ in range(num_predictions):
        action, _ = model.predict(obs, deterministic=True)
    
    total_time = time.time() - start_time
    avg_time_per_prediction = total_time / num_predictions
    predictions_per_second = num_predictions / total_time
    
    results = {
        "model_path": model_path,
        "model_number": int(model_path.split("_")[-1].split(".")[0]) if "walk_forward_model" in model_path else 0,
        "num_predictions": num_predictions,
        "total_time": total_time,
        "avg_time_per_prediction": avg_time_per_prediction,
        "predictions_per_second": predictions_per_second,
        "load_time": load_time
    }
    
    logger.info(f"Results:")
    logger.info(f"  Total time: {total_time:.4f} seconds")
    logger.info(f"  Average time per prediction: {avg_time_per_prediction*1000:.4f} ms")
    logger.info(f"  Predictions per second: {predictions_per_second:.2f}")
    
    return results

def plot_comparison(results_list):
    """
    Plot comparison of prediction speeds across models.
    
    Args:
        results_list (list): List of result dictionaries from test_prediction_speed.
    """
    # Sort results by model number
    results_list.sort(key=lambda x: x["model_number"])
    
    # Extract data for plotting
    model_numbers = [result["model_number"] for result in results_list]
    predictions_per_second = [result["predictions_per_second"] for result in results_list]
    avg_time_ms = [result["avg_time_per_prediction"] * 1000 for result in results_list]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot predictions per second
    ax1.bar(model_numbers, predictions_per_second, color='blue', alpha=0.7)
    ax1.set_xlabel('Model Number')
    ax1.set_ylabel('Predictions per Second')
    ax1.set_title('Prediction Speed Comparison (Higher is Better)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight model #9
    model_9_index = model_numbers.index(9) if 9 in model_numbers else None
    if model_9_index is not None:
        ax1.bar([model_numbers[model_9_index]], [predictions_per_second[model_9_index]], 
                color='red', alpha=0.7, label='Model #9')
        ax1.legend()
    
    # Plot average time per prediction
    ax2.bar(model_numbers, avg_time_ms, color='green', alpha=0.7)
    ax2.set_xlabel('Model Number')
    ax2.set_ylabel('Average Time per Prediction (ms)')
    ax2.set_title('Average Prediction Time (Lower is Better)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight model #9
    if model_9_index is not None:
        ax2.bar([model_numbers[model_9_index]], [avg_time_ms[model_9_index]], 
                color='red', alpha=0.7, label='Model #9')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('model_prediction_speed_comparison.png')
    logger.info("Comparison plot saved as 'model_prediction_speed_comparison.png'")
    plt.close()

def main():
    """Main function to run the prediction speed test."""
    # Load configuration
    yaml_config = load_config()
    
    # Get data
    logger.info("Loading data...")
    data = get_data(
        symbol=yaml_config["data"]["symbol"],
        period=yaml_config["data"]["period"],
        interval=yaml_config["data"]["interval"],
        train_ratio=yaml_config["data"]["train_ratio"],
        validation_ratio=yaml_config["data"]["validation_ratio"]
    )
    
    # Use test data for predictions
    test_data = data[2]  # get_data returns (train_df, validation_df, test_df)
    
    # Define models to test
    models_to_test = [
        "walk_forward_model_9.zip",  # Model #9 (our focus)
        "walk_forward_model_1.zip",  # For comparison
        "walk_forward_model_5.zip",  # For comparison
        "walk_forward_model_10.zip", # For comparison
        "walk_forward_model_14.zip", # For comparison
        "best_model.zip"             # Best model for comparison
    ]
    
    # Test each model
    all_results = []
    
    for model_path in models_to_test:
        logger.info(f"\n=== Testing model: {model_path} ===")
        results = test_prediction_speed(
            model_path=model_path,
            test_data=test_data,
            num_predictions=1000
        )
        all_results.append(results)
    
    # Print comparison table
    logger.info("\nModel Comparison:")
    logger.info(f"{'Model':<20} {'Predictions/sec':<20} {'Avg Time (ms)':<20} {'Load Time (s)':<15}")
    logger.info("-" * 75)
    
    for result in all_results:
        model_name = result["model_path"]
        preds_per_sec = result["predictions_per_second"]
        avg_time_ms = result["avg_time_per_prediction"] * 1000
        load_time = result["load_time"]
        
        logger.info(f"{model_name:<20} {preds_per_sec:<20.2f} {avg_time_ms:<20.4f} {load_time:<15.4f}")
    
    # Plot comparison
    plot_comparison(all_results)
    
    # Detailed analysis of model #9
    logger.info("\n=== Detailed Analysis of Model #9 ===")
    
    # Test with different batch sizes
    batch_sizes = [1, 10, 100, 1000]
    batch_results = []
    
    for batch_size in batch_sizes:
        logger.info(f"\nTesting model #9 with batch size: {batch_size}")
        batch_result = test_prediction_speed(
            model_path="walk_forward_model_9.zip",
            test_data=test_data,
            num_predictions=batch_size
        )
        batch_results.append(batch_result)
    
    # Print batch size comparison
    logger.info("\nBatch Size Comparison for Model #9:")
    logger.info(f"{'Batch Size':<15} {'Predictions/sec':<20} {'Avg Time (ms)':<20}")
    logger.info("-" * 55)
    
    for result in batch_results:
        batch_size = result["num_predictions"]
        preds_per_sec = result["predictions_per_second"]
        avg_time_ms = result["avg_time_per_prediction"] * 1000
        
        logger.info(f"{batch_size:<15} {preds_per_sec:<20.2f} {avg_time_ms:<20.4f}")

if __name__ == "__main__":
    main() 