import time
import logging
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from environment import TradingEnv
from get_data import get_data
from config import config
import yaml

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
        indicators=yaml_config["indicators"],
        train_ratio=yaml_config["data"]["train_ratio"],
        validation_ratio=yaml_config["data"]["validation_ratio"]
    )
    
    # Use test data for predictions
    test_data = data["test"]
    
    # Model path for model #9
    model_path = "walk_forward_model_9.zip"
    
    # Test prediction speed
    results = test_prediction_speed(
        model_path=model_path,
        test_data=test_data,
        num_predictions=1000  # Adjust as needed
    )
    
    # You can also test with different batch sizes
    batch_sizes = [1, 10, 100, 1000]
    batch_results = []
    
    for batch_size in batch_sizes:
        logger.info(f"\nTesting with batch size: {batch_size}")
        batch_result = test_prediction_speed(
            model_path=model_path,
            test_data=test_data,
            num_predictions=batch_size
        )
        batch_results.append(batch_result)
    
    # Print comparison
    logger.info("\nBatch size comparison:")
    for result in batch_results:
        logger.info(f"  Batch size: {result['num_predictions']}, "
                   f"Predictions per second: {result['predictions_per_second']:.2f}, "
                   f"Avg time per prediction: {result['avg_time_per_prediction']*1000:.4f} ms")

if __name__ == "__main__":
    main() 