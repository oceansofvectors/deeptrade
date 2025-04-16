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

# Import custom modules
from config import config
from get_data import process_technical_indicators, ensure_numeric
import money  # Import for formatting functions

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
        
        # Check if we have the required columns
        required_cols = ['open', 'high', 'low', 'close']
        time_cols = ['time', 'timestamp', 'datetime']  # Accept multiple possible time column names
        
        # Convert column names to lowercase for case-insensitive matching
        available_cols = [col.lower() for col in df.columns]
        
        # Check if all required columns are present (case insensitive)
        missing_price_cols = [col for col in required_cols if col not in available_cols]
        if missing_price_cols:
            logger.error(f"Missing required price columns: {missing_price_cols}")
            return None
            
        # Check if at least one of the time columns is present
        time_col_found = next((col for col in df.columns if col.lower() in time_cols), None)
        if not time_col_found:
            logger.error(f"Missing time column in data. Available columns: {df.columns.tolist()}")
            return None
        
        # Create a standardized DataFrame with consistent column names
        standard_df = pd.DataFrame()
        
        # Map columns (case-insensitive) to standard format
        column_map = {}
        for std_col in required_cols + time_cols:
            for avail_col in df.columns:
                if avail_col.lower() == std_col:
                    column_map[avail_col] = std_col.lower()
        
        # Create the mapping for the standardized DataFrame
        for old_name, new_name in column_map.items():
            standard_df[new_name] = df[old_name]
        
        # Ensure we have our time column with a consistent name
        time_col_name = next(col for col in standard_df.columns if col in time_cols)
        if time_col_name != 'time':
            standard_df['time'] = standard_df[time_col_name]
        
        # Convert time column to datetime
        try:
            # Try different approaches to convert time to datetime depending on its format
            if pd.api.types.is_numeric_dtype(standard_df['time']):
                # If time is already numeric (timestamp), convert to datetime
                logger.info("Converting numeric timestamp to datetime")
                standard_df['time'] = pd.to_datetime(standard_df['time'], unit='s')
            else:
                # If time is string or already datetime
                logger.info("Converting string or datetime to datetime")
                standard_df['time'] = pd.to_datetime(standard_df['time'])
            
            # Set time as index
            standard_df = standard_df.set_index('time')
            logger.info("Successfully converted time column to datetime index")
        except Exception as e:
            logger.error(f"Error converting time column: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
        # Ensure price columns are numeric
        price_cols = ['open', 'high', 'low', 'close']
        if 'volume' in standard_df.columns:
            price_cols.append('volume')
        
        # Ensure data types are correct
        standard_df = ensure_numeric(standard_df, price_cols)
        
        # Rename columns to match what technical indicator calculation expects
        rename_map = {
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        standard_df = standard_df.rename(columns=rename_map)
        
        # Process technical indicators
        processed_df = process_technical_indicators(standard_df)
        
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

def get_risk_params():
    """
    Extract risk management parameters from config, respecting enabled/disabled settings.
    
    Returns:
        dict: Risk management parameters with only enabled features
    """
    risk_params = {
        "enabled": False,
        "stop_loss": None,
        "take_profit": None,
        "trailing_stop": None,
        "position_size": config["environment"].get("position_size", 1)
    }
    
    # Check if risk management is enabled globally
    risk_config = config.get("risk_management", {})
    risk_enabled = risk_config.get("enabled", False)
    
    if not risk_enabled:
        logger.info("Risk management is disabled globally")
        return risk_params
    
    risk_params["enabled"] = True
    
    # Process stop loss if enabled
    stop_loss_config = risk_config.get("stop_loss", {})
    if stop_loss_config.get("enabled", False):
        risk_params["stop_loss"] = stop_loss_config.get("percentage", 0)
        logger.info(f"Stop loss enabled at {risk_params['stop_loss']}%")
    else:
        logger.info("Stop loss is disabled")
    
    # Process take profit if enabled
    take_profit_config = risk_config.get("take_profit", {})
    if take_profit_config.get("enabled", False):
        risk_params["take_profit"] = take_profit_config.get("percentage", 0)
        logger.info(f"Take profit enabled at {risk_params['take_profit']}%")
    else:
        logger.info("Take profit is disabled")
    
    # Process trailing stop if enabled
    trailing_stop_config = risk_config.get("trailing_stop", {})
    if trailing_stop_config.get("enabled", False):
        risk_params["trailing_stop"] = trailing_stop_config.get("percentage", 0)
        logger.info(f"Trailing stop enabled at {risk_params['trailing_stop']}%")
    else:
        logger.info("Trailing stop is disabled")
    
    # Process position sizing if enabled
    position_sizing_config = risk_config.get("position_sizing", {})
    if position_sizing_config.get("enabled", False):
        risk_params["position_size"] = position_sizing_config.get("size_multiplier", 1.0)
        logger.info(f"Position sizing enabled with multiplier: {risk_params['position_size']}")
    else:
        logger.info(f"Position sizing is disabled, using default size: {risk_params['position_size']}")
    
    return risk_params

def main():
    # Create output directories
    os.makedirs('models', exist_ok=True)
    
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
        enabled_indicators.append("TREND_DIRECTION")
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
    
    # Log the final list of enabled indicators
    logger.info(f"Enabled indicators for training: {enabled_indicators}")

    # DEBUGGING: Print the features that will be used in the observation vector
    # This will help debug why the model expects 9 features
    observation_features = ["close_norm"]  # First feature is always close_norm
    observation_features.extend(enabled_indicators)  # Add all enabled indicators
    observation_features.append("position")  # Last feature is always position
    logger.info(f"OBSERVATION VECTOR will contain {len(observation_features)} features: {observation_features}")
    
    # Save enabled indicators to model folder
    with open(os.path.join(model_folder, "enabled_indicators.json"), "w") as f:
        json.dump(enabled_indicators, f, indent=4)
    
    # Get risk management parameters
    risk_params = get_risk_params()
    
    # Save risk parameters to model folder
    with open(os.path.join(model_folder, "risk_params.json"), "w") as f:
        json.dump(risk_params, f, indent=4)
    
    # Log training parameters
    logger.info(f"Training parameters:")
    logger.info(f"- train_ratio: {train_ratio}")
    logger.info(f"- validation_ratio: {validation_ratio}")
    logger.info(f"- risk_management: {risk_params['enabled']}")
    logger.info(f"- enabled_indicators: {enabled_indicators}")
    
    # Load data from the live CSV file
    data = load_live_data("data/live.csv")
    
    if data is None or len(data) == 0:
        logger.error("Failed to load data or dataset is empty")
        sys.exit(1)
    
    logger.info(f"Loaded dataset with {len(data)} records")
    
    # Split data into training, validation, and test sets
    train_split_idx = int(len(data) * train_ratio)
    validation_split_idx = train_split_idx + int(len(data) * validation_ratio)
    
    train_data = data.iloc[:train_split_idx].copy()
    validation_data = data.iloc[train_split_idx:validation_split_idx].copy()
    test_data = data.iloc[validation_split_idx:].copy()
    
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
            model_params=model_params
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
        
        # Save a copy of risk parameters in the root directory for live_trading.py
        with open("risk_params.json", "w") as f:
            json.dump(risk_params, f, indent=4)
        logger.info("Risk parameters saved to risk_params.json for live trading")
            
        logger.info(f"Training completed successfully. Model is ready for live trading.")
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 