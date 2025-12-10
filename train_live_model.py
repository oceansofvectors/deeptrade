#!/usr/bin/env python3
# train_live_model.py
# This script trains a model for live trading using data from data/live.csv

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from utils.seeding import set_global_seed
from typing import List, Dict, Tuple, Optional, Union

# Import custom modules
from config import config
from get_data import process_technical_indicators, ensure_numeric
import money  # Import for formatting functions
from normalization import scale_window, get_standardized_column_names  # Add normalization module

# Import hyperparameter tuning and evaluation functions
try:
    from walk_forward import hyperparameter_tuning
    print(f"Successfully imported hyperparameter_tuning: {hyperparameter_tuning}")
    from walk_forward import filter_market_hours
    from train import train_agent_iteratively, evaluate_agent
except ImportError as e:
    print(f"Warning: Could not import all modules. Error: {e}")
    print("Make sure walk_forward.py and train.py exist.")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_live_data(csv_filepath="data/live.csv") -> pd.DataFrame:
    """
    Load and process data from a CSV file for live trading.
    
    Args:
        csv_filepath: Path to the CSV file with market data
        
    Returns:
        DataFrame: Processed data with technical indicators
    """
    logger.info(f"Loading live trading data from {csv_filepath}")
    
    try:
        # Check if file exists
        if not os.path.exists(csv_filepath):
            logger.error(f"File not found: {csv_filepath}")
            return None
            
        # Read the CSV file
        df = pd.read_csv(csv_filepath)
        
        # Debug information
        logger.info(f"Raw data columns: {df.columns.tolist()}")
        logger.info(f"Data shape: {df.shape}")
        
        # Check for required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return None
        
        # Convert timestamp to datetime and set as index
        try:
            # Check the format of the timestamp
            sample_timestamp = df['timestamp'].iloc[0]
            logger.info(f"Sample timestamp: {sample_timestamp}")
            
            # Try to determine if it's a numeric timestamp or string format
            if isinstance(sample_timestamp, (int, float)) or str(sample_timestamp).isdigit():
                # If it's numeric, convert with unit='s'
                logger.info("Converting numeric timestamp to datetime")
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                # If it's a string in ISO format, convert directly
                logger.info("Converting ISO format timestamp to datetime")
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df = df.set_index('timestamp')
            logger.info("Successfully converted timestamp column to datetime index")
        except Exception as e:
            logger.error(f"Error converting timestamp column: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
        # Ensure all data columns are numeric
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Check for NaN values
        if df[numeric_cols].isna().any().any():
            logger.warning(f"Dataset contains NaN values after conversion to numeric")
            df = df.dropna(subset=numeric_cols)
            logger.info(f"Dropped rows with NaN values. New shape: {df.shape}")
        
        # Process technical indicators
        logger.info("Processing technical indicators")
        processed_df = process_technical_indicators(df)
        
        if processed_df is None:
            logger.error("Technical indicator processing failed")
            return None
            
        # Calculate close_norm manually if it wasn't created
        if 'close_norm' not in processed_df.columns:
            logger.info("Manually calculating close_norm")
            # Calculate close_norm as the percentage change from the first close price
            first_close = processed_df['close'].iloc[0]
            processed_df['close_norm'] = processed_df['close'] / first_close - 1.0
        
        # Apply market hours filter if configured
        if config["data"].get("market_hours_only", False):
            logger.info("Filtering data to include only market hours")
            try:
                processed_df = filter_market_hours(processed_df)
            except Exception as e:
                logger.warning(f"Could not filter for market hours: {e}. Using all data.")
        
        logger.info(f"Data loaded and processed. Final shape: {processed_df.shape}")
        logger.info(f"Final columns: {processed_df.columns.tolist()}")
        
        return processed_df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def normalize_window_data(data: pd.DataFrame, window_folder: str) -> pd.DataFrame:
    """
    Normalize the window data using the normalization module.
    
    Args:
        data: DataFrame with technical indicators
        window_folder: Folder to save normalizers
        
    Returns:
        Normalized DataFrame
    """
    logger.info("Normalizing data for training")
    
    # Split data for normalization (we'll use the same data for training/validation/test)
    train_ratio = config["data"].get("train_ratio", 0.7)
    validation_ratio = config["data"].get("validation_ratio", 0.15)
    
    train_split_idx = int(len(data) * train_ratio)
    validation_split_idx = train_split_idx + int(len(data) * validation_ratio)
    
    train_data = data.iloc[:train_split_idx].copy()
    validation_data = data.iloc[train_split_idx:validation_split_idx].copy()
    test_data = data.iloc[validation_split_idx:].copy()
    
    # Get columns to normalize
    skip_columns = [
        'close_norm', 'position', 'trend_direction', 'supertrend', 
        'time', 'timestamp', 'date', 'DOW', 
        'Up Trend', 'Down Trend', 'open', 'high', 'low', 'close', 
        'Open', 'High', 'Low', 'Close', 'Volume', 'volume'
    ]
    cols_to_scale = get_standardized_column_names(data, skip_columns)
    
    logger.info(f"Normalizing {len(cols_to_scale)} columns: {cols_to_scale}")
    
    # Use feature range from config
    feature_range = config.get("normalization", {}).get("feature_range", (-1, 1))
    
    # Normalize the data
    scaler, normalized_train, normalized_val, normalized_test = scale_window(
        train_data=train_data,
        val_data=validation_data,
        test_data=test_data,
        cols_to_scale=cols_to_scale,
        feature_range=feature_range,
        window_folder=window_folder,
        use_sigmoid=config.get("normalization", {}).get("use_sigmoid", True),
        sigmoid_k=config.get("normalization", {}).get("sigmoid_k", 2.0)
    )
    
    # Combine the normalized datasets back into one
    normalized_data = pd.concat([normalized_train, normalized_val, normalized_test])
    
    # Sort by index to ensure chronological order
    normalized_data = normalized_data.sort_index()
    
    logger.info(f"Data normalized. Shape: {normalized_data.shape}")
    
    return normalized_data

def main():
    set_global_seed(config['seed'])
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Current timestamp for model folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_folder = f'models/live_{timestamp}'
    os.makedirs(model_folder, exist_ok=True)
    
    # Get training parameters from config
    train_ratio = config["data"].get("train_ratio", 0.7)
    validation_ratio = config["data"].get("validation_ratio", 0.15)
    
    # Get enabled indicators from config
    enabled_indicators = []
    indicators_config = config.get("indicators", {})
    
    # Check each indicator
    if indicators_config.get("rsi", {}).get("enabled", False):
        enabled_indicators.append("RSI")
    if indicators_config.get("cci", {}).get("enabled", False):
        enabled_indicators.append("CCI")
    if indicators_config.get("macd", {}).get("enabled", False):
        enabled_indicators.extend(["MACD", "MACD_SIGNAL", "MACD_HIST"])
    if indicators_config.get("atr", {}).get("enabled", False):
        enabled_indicators.append("ATR")
    if indicators_config.get("supertrend", {}).get("enabled", False):
        enabled_indicators.append("SUPERTREND")
    if indicators_config.get("adx", {}).get("enabled", False):
        enabled_indicators.append("ADX")
    if indicators_config.get("adx_pos", {}).get("enabled", False):
        enabled_indicators.append("ADX_POS")
    if indicators_config.get("adx_neg", {}).get("enabled", False):
        enabled_indicators.append("ADX_NEG")
    if indicators_config.get("stoch_k", {}).get("enabled", False):
        enabled_indicators.append("STOCH_K")
    if indicators_config.get("stoch_d", {}).get("enabled", False):
        enabled_indicators.append("STOCH_D")
    if indicators_config.get("roc", {}).get("enabled", False):
        enabled_indicators.append("ROC")
    if indicators_config.get("williams_r", {}).get("enabled", False):
        enabled_indicators.append("WILLIAMS_R")
    if indicators_config.get("sma", {}).get("enabled", False):
        enabled_indicators.append("SMA_NORM")
    if indicators_config.get("ema", {}).get("enabled", False):
        enabled_indicators.append("EMA_NORM")
    if indicators_config.get("disparity", {}).get("enabled", False):
        enabled_indicators.append("DISPARITY")
    if indicators_config.get("obv", {}).get("enabled", False):
        enabled_indicators.append("OBV_NORM")
    if indicators_config.get("cmf", {}).get("enabled", False):
        enabled_indicators.append("CMF")
    if indicators_config.get("psar", {}).get("enabled", False):
        enabled_indicators.extend(["PSAR_NORM", "PSAR_DIR"])
    if indicators_config.get("volume", {}).get("enabled", False):
        enabled_indicators.append("VOLUME_MA")
    if indicators_config.get("vwap", {}).get("enabled", False):
        enabled_indicators.append("VWAP_NORM")
    
    # Always include day of week indicators
    enabled_indicators.extend(["DOW_SIN", "DOW_COS"])
    
    # Include minutes since open indicators if enabled
    if indicators_config.get("minutes_since_open", {}).get("enabled", False):
        logger.info("Minutes since open indicator is enabled, adding MSO_SIN and MSO_COS")
        enabled_indicators.extend(["MSO_SIN", "MSO_COS"])
    else:
        logger.info("Minutes since open indicator is disabled in config")
    
    # Log the final list of enabled indicators
    logger.info(f"Enabled indicators for training: {enabled_indicators}")

    # DEBUGGING: Print the features that will be used in the observation vector
    # This will help debug why the model expects 9 features
    observation_features = enabled_indicators.copy()  # Use only enabled indicators for observation
    observation_features.append("position")  # Last feature is always position
    logger.info(f"OBSERVATION VECTOR will contain {len(observation_features)} features: {observation_features}")
    
    # Save enabled indicators to model folder
    with open(os.path.join(model_folder, "enabled_indicators.json"), "w") as f:
        json.dump(enabled_indicators, f, indent=4)

    # Log training parameters
    logger.info(f"Training parameters:")
    logger.info(f"- train_ratio: {train_ratio}")
    logger.info(f"- validation_ratio: {validation_ratio}")
    logger.info(f"- risk_management: {config['risk_management']['enabled']}")
    logger.info(f"- enabled_indicators: {enabled_indicators}")
    
    # Load data from the live CSV file
    data = load_live_data("data/live.csv")
    
    if data is None or len(data) == 0:
        logger.error("Failed to load data or dataset is empty")
        sys.exit(1)
    
    logger.info(f"Loaded dataset with {len(data)} records")
    
    # Normalize the data using the normalization module
    normalized_data = normalize_window_data(data, model_folder)
    
    # Save the normalized data to CSV
    normalized_csv_path = f"data/normalized_data_{timestamp}.csv"
    normalized_data.to_csv(normalized_csv_path)
    logger.info(f"Saved normalized data to {normalized_csv_path}")
    
    # Also save a copy with a fixed name for easy access
    normalized_data.to_csv("data/normalized_data_latest.csv")
    logger.info("Saved normalized data to data/normalized_data_latest.csv")
    
    # Check if all required columns for training are available
    required_columns = ['close_norm'] + enabled_indicators
    missing_columns = [col for col in required_columns if col not in normalized_data.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns for training: {missing_columns}")
        logger.error("Available columns: {}".format(normalized_data.columns.tolist()))
        
        # For missing indicators, create dummy columns with zeros
        logger.warning("Creating dummy columns with zeros for missing indicators")
        for col in missing_columns:
            if col != 'close_norm':  # close_norm should be handled separately
                normalized_data[col] = 0.0
    
    # Split data into training, validation, and test sets
    train_split_idx = int(len(normalized_data) * train_ratio)
    validation_split_idx = train_split_idx + int(len(normalized_data) * validation_ratio)
    
    train_data = normalized_data.iloc[:train_split_idx].copy()
    validation_data = normalized_data.iloc[train_split_idx:validation_split_idx].copy()
    test_data = normalized_data.iloc[validation_split_idx:].copy()
    
    logger.info(f"Data split into:")
    logger.info(f"- Training: {len(train_data)} records")
    logger.info(f"- Validation: {len(validation_data)} records") 
    logger.info(f"- Test: {len(test_data)} records")
    
    # Get training hyperparameters
    initial_timesteps = config["training"].get("total_timesteps", 50000)
    additional_timesteps = config["training"].get("additional_timesteps", 10000)
    max_iterations = config["training"].get("max_iterations", 20)
    n_stagnant_loops = config["training"].get("n_stagnant_loops", 3)
    improvement_threshold = config["training"].get("improvement_threshold", 0.1)
    
    # Get the evaluation metric from config
    evaluation_metric = config.get("training", {}).get("evaluation", {}).get("metric", "return")
    logger.info(f"Using {evaluation_metric} as evaluation metric for model selection")
    
    # Initialize model parameters
    model_params = None
    
    # Perform hyperparameter tuning if enabled
    if config.get("hyperparameter_tuning", {}).get("enabled", True):
        logger.info(f"Starting hyperparameter tuning with {config['hyperparameter_tuning']['n_trials']} trials using {evaluation_metric} metric")
        
        try:
            # Run hyperparameter tuning
            tuning_kwargs = {
                'train_data': train_data,
                'validation_data': validation_data,
                'n_trials': config['hyperparameter_tuning']['n_trials'],
                'window_folder': model_folder,
                'eval_metric': evaluation_metric
            }
            
            best_params = hyperparameter_tuning(**tuning_kwargs)
            logger.info(f"Hyperparameter tuning completed. Best parameters: {best_params}")
            
            # Use the best parameters for model training
            model_params = best_params.get('best_params', best_params)
            
            # Save tuning results
            with open(os.path.join(model_folder, "best_params.json"), "w") as f:
                json_safe_params = best_params.get('best_params', best_params)
                json.dump(json_safe_params, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.warning("Proceeding with default parameters")
    else:
        logger.warning("Hyperparameter tuning is disabled. Using default parameters.")
    
    # Log the model parameters being used
    logger.info(f"Training model with parameters: {model_params if model_params else 'default parameters'}")
    
    # Train the model iteratively
    try:
        model, validation_results, all_results = train_agent_iteratively(
            train_data, 
            validation_data,
            initial_timesteps=initial_timesteps,
            additional_timesteps=additional_timesteps,
            max_iterations=max_iterations,
            n_stagnant_loops=n_stagnant_loops,
            improvement_threshold=improvement_threshold,
            evaluation_metric=evaluation_metric,
            model_params=model_params,
            window_folder=model_folder
        )
        
        # Evaluate on test data
        test_results = evaluate_agent(model, test_data, verbose=1, deterministic=True)
        
        # Log test results
        logger.info(f"Test Results:")
        logger.info(f"- Final Portfolio Value: ${test_results['final_portfolio_value']:.2f}")
        logger.info(f"- Total Return: {test_results['total_return_pct']:.2f}%")
        logger.info(f"- Total Trades: {test_results['trade_count']}")
        
        # Save validation and test results
        with open(os.path.join(model_folder, "validation_results.json"), "w") as f:
            json.dump(validation_results, f, indent=4, default=str)
            
        with open(os.path.join(model_folder, "test_results.json"), "w") as f:
            json.dump(test_results, f, indent=4, default=str)
        
        # Save model in the model folder
        model_path = os.path.join(model_folder, "model")
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Also save model as best_model in the root directory for live_trading.py
        model.save("best_model")
        logger.info("Model saved as best_model in root directory for live trading")
        
        # Save training iterations data
        training_stats = []
        for i, result in enumerate(all_results):
            entry = {
                "iteration": i,
                "return_pct": result.get("total_return_pct", 0),
                "portfolio_value": result.get("final_portfolio_value", 0),
                "hit_rate": result.get("hit_rate", 0),
                "is_best": result.get("is_best", False),
                "metric_used": result.get("metric_used", evaluation_metric)
            }
            training_stats.append(entry)
        
        with open(os.path.join(model_folder, "training_stats.json"), "w") as f:
            json.dump(training_stats, f, indent=4)

        logger.info(f"Training completed successfully. Model is ready for live trading.")
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 