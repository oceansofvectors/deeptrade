import logging
import re
import pandas as pd
import numpy as np
import os
import json
import yaml
from datetime import datetime, time, timedelta
import pytz
from typing import Any, List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import optuna
import math
from decimal import Decimal
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import pickle

# RecurrentPPO for LSTM support
try:
    from sb3_contrib import RecurrentPPO
    RECURRENT_PPO_AVAILABLE = True
except ImportError:
    RECURRENT_PPO_AVAILABLE = False

from environment import TradingEnv
from get_data import get_data
from train import evaluate_agent, plot_results, train_walk_forward_model
from trade import trade_with_risk_management, save_trade_history  # Import trade_with_risk_management and save_trade_history
from config import config
import money
from utils.seeding import seed_worker  # Import the seed_worker function
from utils.device import get_device  # Import device utility
from normalization import scale_window, get_standardized_column_names  # Import both functions from normalization
from indicators.lstm_features import LSTMFeatureGenerator, tune_lstm_hyperparameters
from utils.synthetic_bears import augment_with_synthetic_bears

from utils.log_format import (
    ACTION_NAMES,
    ANSI_BOLD,
    ANSI_GREEN,
    ANSI_RED,
    ANSI_RESET,
    AnsiStrippingFormatter,
    bold,
    color_pct,
    format_action_distribution,
)

# Setup logging to save to file and console
os.makedirs('models/logs', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'models/logs/walk_forward_{timestamp}.log'

_log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
_file_handler = logging.FileHandler(log_filename)
_file_handler.setFormatter(AnsiStrippingFormatter(_log_fmt))
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter(_log_fmt))

logging.basicConfig(
    level=logging.INFO,
    handlers=[_file_handler, _stream_handler]
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


def _apply_best_hyperparameters(config_data: Dict[str, Any], best_params: Dict[str, Any]) -> List[str]:
    """Apply tuned parameters to a config dictionary and return updated keys."""
    updated_params = []

    if 'model' not in config_data:
        config_data['model'] = {}
    if 'sequence_model' not in config_data:
        config_data['sequence_model'] = {}
    if 'reward' not in config_data:
        config_data['reward'] = {}
    if 'augmentation' not in config_data:
        config_data['augmentation'] = {}
    if 'synthetic_bears' not in config_data['augmentation']:
        config_data['augmentation']['synthetic_bears'] = {}

    model_params = ['learning_rate', 'n_steps', 'batch_size', 'gamma', 'gae_lambda', 'ent_coef']
    for param in model_params:
        if param in best_params:
            value = float(best_params[param])
            if config_data['model'].get(param) != value:
                updated_params.append(param)
            config_data['model'][param] = value

    sequence_params = ['lstm_hidden_size', 'n_lstm_layers']
    for param in sequence_params:
        if param in best_params:
            value = int(best_params[param])
            if config_data['sequence_model'].get(param) != value:
                updated_params.append(param)
            config_data['sequence_model'][param] = value

    reward_param_map = {
        'reward_loss_multiplier': 'loss_multiplier',
        'reward_turnover_penalty': 'turnover_penalty',
        'reward_calm_holding_bonus': 'calm_holding_bonus',
    }
    for best_param, config_param in reward_param_map.items():
        if best_param in best_params:
            value = float(best_params[best_param])
            if config_data['reward'].get(config_param) != value:
                updated_params.append(best_param)
            config_data['reward'][config_param] = value

    if 'synthetic_oversample_ratio' in best_params:
        value = float(best_params['synthetic_oversample_ratio'])
        if config_data['augmentation']['synthetic_bears'].get('oversample_ratio') != value:
            updated_params.append('synthetic_oversample_ratio')
        config_data['augmentation']['synthetic_bears']['oversample_ratio'] = value

    return updated_params


def apply_best_hyperparameters_to_runtime_config(best_params: Dict[str, Any]) -> List[str]:
    """Update the in-memory config for the current run."""
    return _apply_best_hyperparameters(config, best_params)

def save_best_hyperparameters_to_config(best_params: Dict, config_path: str = "config.yaml"):
    """
    Save the best hyperparameters to config.yaml for reuse in future runs.

    Updates the model section with tuned values for: learning_rate, n_steps,
    batch_size, gamma, gae_lambda, ent_coef, and LSTM params if present.

    Args:
        best_params: Dictionary of best hyperparameters from tuning
        config_path: Path to the config.yaml file
    """
    try:
        # Read the current config
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        updated_params = _apply_best_hyperparameters(config_data, best_params)

        # Write back to config
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved best hyperparameters to {config_path}")
        logger.info(f"Updated params: {updated_params}")

    except Exception as e:
        logger.error(f"Failed to save hyperparameters to config: {e}")

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

# Import filter_market_hours from shared utilities
from utils.data_utils import filter_market_hours

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
        
    # Create evaluation environment with realistic transaction costs
    env = TradingEnv(
        test_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 2.50),
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
    entry_price = Decimal('0')  # Track entry price for P&L calculation

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

        # Handle MultiDiscrete action space (dynamic SL/TP)
        # Use np.ndim to safely check if action is an array (avoids len() on scalar)
        if isinstance(action, np.ndarray) and action.ndim > 0 and action.size > 1:
            # MultiDiscrete action: [position_action, sl_idx, tp_idx]
            position_action = int(action[0])
        else:
            position_action = int(action)
        action_history.append(position_action)

        # Convert action to position (0 = long/buy, 1 = short/sell, 2 = hold, 3 = flat)
        if position_action == 0:  # Long
            new_position = 1
        elif position_action == 1:  # Short
            new_position = -1
        elif position_action == 3:  # Flat - close position
            new_position = 0
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
            elif action == 3:  # Flat - prediction is correct if avoiding losses (price moves against previous position)
                # Flat is correct if we avoided a losing move
                if current_position == 1:  # Was long, flat is correct if price went down
                    prediction_correct = price_change < 0
                elif current_position == -1:  # Was short, flat is correct if price went up
                    prediction_correct = price_change > 0
                else:  # Already flat, correct if price doesn't move much
                    threshold = current_price * money.to_decimal(0.001)
                    prediction_correct = abs(price_change) <= threshold
            else:  # Hold (action == 2) - prediction is correct if price doesn't change much
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
        
        # Only record actual position changes (not every step)
        position_changed = new_position != current_position
        if position_changed:
            trade_count += 1

            # Calculate P&L for exiting trades
            is_profitable = False
            trade_pnl = Decimal('0')
            if current_position != 0 and entry_price > 0:
                if current_position == 1:  # Was long
                    trade_pnl = current_price - entry_price
                    is_profitable = trade_pnl > 0
                elif current_position == -1:  # Was short
                    trade_pnl = entry_price - current_price
                    is_profitable = trade_pnl > 0

            action_names = {0: "LONG", 1: "SHORT", 2: "HOLD", 3: "FLAT"}
            trade_info = {
                "step": current_step,
                "action": action_names.get(int(action), "UNKNOWN"),
                "price": float(current_price),
                "timestamp": test_data.index[current_step].strftime('%Y-%m-%d %H:%M:%S') if hasattr(test_data.index[current_step], 'strftime') else str(test_data.index[current_step]),
                "old_position": current_position,
                "new_position": new_position,
                "entry_price": float(entry_price) if entry_price > 0 else None,
                "trade_pnl_points": float(trade_pnl),
                "profitable": is_profitable
            }
            trade_history.append(trade_info)

            # Update entry price for new positions
            if new_position != 0:
                entry_price = current_price
            else:
                entry_price = Decimal('0')

            current_position = new_position
        
        # Take step in environment
        new_obs, reward, done, truncated, info = env.step(action)
        obs = new_obs
        done = done or truncated

        # Update portfolio from environment
        current_portfolio = env.net_worth
        # Sample portfolio history (every 10 steps or on position change) to reduce memory
        if position_changed or len(portfolio_history) == 0 or current_step % 10 == 0:
            portfolio_history.append(float(current_portfolio))
    
    # Calculate final metrics
    prediction_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    total_return_pct = money.calculate_return_pct(current_portfolio, initial_balance)
    
    # Calculate hit rate (percentage of profitable trades) from actual trade P&L
    profitable_trades = sum(1 for t in trade_history if t.get("profitable", False))
    # Only count trades that exited a position (have entry_price)
    completed_trades = sum(1 for t in trade_history if t.get("entry_price") is not None)
    hit_rate = (profitable_trades / completed_trades * 100) if completed_trades > 0 else 0
    
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
        logger.info(f"Return: {color_pct(float(total_return_pct))}, Final portfolio: ${float(current_portfolio):.2f}")
        logger.info(f"Total trades: {trade_count}, Actions: {format_action_distribution(action_history)}")
    
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


def get_lstm_feature_params(
    train_data: pd.DataFrame,
    artifact_folder: Optional[str] = None,
    provided_params: Optional[Dict[str, Any]] = None,
    allow_tuning: bool = True,
) -> Optional[Dict[str, Any]]:
    """Resolve LSTM feature hyperparameters, optionally tuning once on the provided train split."""
    lstm_config = config.get("indicators", {}).get("lstm_features", {})
    if not lstm_config.get("enabled", False):
        return None

    if provided_params is not None:
        return provided_params

    lstm_tuning_config = lstm_config.get("tuning", {})
    if allow_tuning and lstm_tuning_config.get("enabled", False):
        return tune_lstm_hyperparameters(
            train_data=train_data,
            tuning_config=lstm_tuning_config,
            base_config=lstm_config,
            window_folder=artifact_folder,
        )

    return {
        "lookback": lstm_config.get("lookback", 20),
        "hidden_size": lstm_config.get("hidden_size", 32),
        "num_layers": lstm_config.get("num_layers", 1),
        "output_size": lstm_config.get("output_size", 8),
        "pretrain_epochs": lstm_config.get("pretrain_epochs", 50),
        "pretrain_lr": lstm_config.get("pretrain_lr", 0.001),
        "pretrain_batch_size": lstm_config.get("pretrain_batch_size", 64),
        "pretrain_patience": lstm_config.get("pretrain_patience", 10),
    }


def apply_lstm_features_to_datasets(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: Optional[pd.DataFrame] = None,
    artifact_folder: Optional[str] = None,
    lstm_params: Optional[Dict[str, Any]] = None,
    prev_lstm_checkpoint_path: Optional[str] = None,
    save_artifacts: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """Train one LSTM autoencoder on train_data and transform all provided datasets."""
    resolved_params = get_lstm_feature_params(
        train_data=train_data,
        artifact_folder=artifact_folder,
        provided_params=lstm_params,
        allow_tuning=lstm_params is None,
    )
    if resolved_params is None:
        return train_data.copy(), validation_data.copy(), test_data.copy() if test_data is not None else None, None

    lstm_generator = LSTMFeatureGenerator(
        lookback=resolved_params["lookback"],
        hidden_size=resolved_params["hidden_size"],
        num_layers=resolved_params["num_layers"],
        output_size=resolved_params["output_size"],
        pretrain_epochs=resolved_params.get("pretrain_epochs", 50),
        pretrain_lr=resolved_params.get("pretrain_lr", 0.001),
        pretrain_batch_size=resolved_params.get("pretrain_batch_size", 64),
        pretrain_patience=resolved_params.get("pretrain_patience", 10),
    )

    checkpoint_path = None
    if artifact_folder and save_artifacts:
        checkpoint_path = f"{artifact_folder}/lstm_autoencoder_checkpoint.pt"

    lstm_generator.fit(train_data.copy(), checkpoint_path=checkpoint_path, warm_start_path=prev_lstm_checkpoint_path)

    transformed_train = lstm_generator.transform(train_data.copy())
    transformed_validation = lstm_generator.transform(validation_data.copy())
    transformed_test = lstm_generator.transform(test_data.copy()) if test_data is not None else None

    if artifact_folder and save_artifacts:
        lstm_generator.save(f"{artifact_folder}/lstm_generator.pkl")

    return transformed_train, transformed_validation, transformed_test, resolved_params


def drop_unused_model_columns(*datasets: pd.DataFrame) -> None:
    """Drop columns that should not reach the policy network."""
    risk_config = config.get("risk_management", {})
    risk_enabled = risk_config.get("enabled", False)
    dynamic_sl_tp_enabled = risk_config.get("dynamic_sl_tp", {}).get("enabled", False)
    sl_mode_atr = risk_config.get("stop_loss", {}).get("mode") == "atr"
    tp_mode_atr = risk_config.get("take_profit", {}).get("mode") == "atr"
    needs_ohlc = risk_enabled or dynamic_sl_tp_enabled or sl_mode_atr or tp_mode_atr

    cols_to_drop = ['volume', 'Volume', 'SMA', 'EMA', 'VWAP', 'PSAR', 'OBV', 'VOLUME_NORM', 'DOW', 'position']
    if not needs_ohlc:
        cols_to_drop.extend(['open', 'Open', 'OPEN', 'high', 'low', 'High', 'Low', 'HIGH', 'LOW'])

    for dataset in datasets:
        if dataset is None:
            continue
        cols_present = [col for col in cols_to_drop if col in dataset.columns]
        if cols_present:
            dataset.drop(columns=cols_present, inplace=True)


def prepare_model_datasets(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: Optional[pd.DataFrame] = None,
    artifact_folder: Optional[str] = None,
    augment_oversample_ratio: Optional[float] = None,
    augmentation_seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Apply augmentation, scaling, and feature dropping using shared walk-forward rules."""
    prepared_train = train_data.copy()
    prepared_validation = validation_data.copy()
    prepared_test = test_data.copy() if test_data is not None else validation_data.copy()

    aug_config = config.get("augmentation", {}).get("synthetic_bears", {})
    if aug_config.get("enabled", False):
        oversample_ratio = augment_oversample_ratio
        if oversample_ratio is None:
            oversample_ratio = aug_config.get("oversample_ratio", 0.3)
        prepared_train = augment_with_synthetic_bears(
            prepared_train,
            oversample_ratio=oversample_ratio,
            segment_length_pct=aug_config.get("segment_length_pct", 0.15),
            seed=config.get("seed", 42) if augmentation_seed is None else augmentation_seed,
        )

    cols_to_scale = get_standardized_column_names(prepared_train)
    scaler_type = config.get("normalization", {}).get("scaler_type", "robust")

    _, prepared_train, prepared_validation, prepared_test = scale_window(
        train_data=prepared_train,
        val_data=prepared_validation,
        test_data=prepared_test,
        cols_to_scale=cols_to_scale,
        feature_range=(-1, 1),
        window_folder=artifact_folder,
        scaler_type=scaler_type,
    )

    drop_unused_model_columns(prepared_train, prepared_validation, prepared_test)
    return prepared_train, prepared_validation, prepared_test if test_data is not None else None


def get_objective_metric(results: Dict[str, Any], eval_metric: str) -> float:
    """Extract the metric Optuna should maximize from an evaluation result dict."""
    metric_map = {
        "return": "total_return_pct",
        "sortino": "sortino_ratio",
        "calmar": "calmar_ratio",
        "hit_rate": "hit_rate",
        "prediction_accuracy": "prediction_accuracy",
    }
    metric_key = metric_map.get(eval_metric, "total_return_pct")
    metric_value = results.get(metric_key)
    if metric_value is None or not np.isfinite(metric_value):
        raise ValueError(f"Metric '{metric_key}' is undefined for trial results")
    return float(metric_value)

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
    tuning_trials: int,
    best_hyperparameters: Dict = None,
    lstm_feature_params: Dict = None,
    prev_lstm_checkpoint_path: str = None,
    prev_model_path: str = None
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

    train_data, validation_data, test_data, resolved_lstm_params = apply_lstm_features_to_datasets(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        artifact_folder=window_folder,
        lstm_params=lstm_feature_params,
        prev_lstm_checkpoint_path=prev_lstm_checkpoint_path,
    )
    if resolved_lstm_params is not None:
        window_logger.info(
            "Using LSTM feature params: hidden=%s, layers=%s, output=%s, lookback=%s, lr=%.6f",
            resolved_lstm_params["hidden_size"],
            resolved_lstm_params["num_layers"],
            resolved_lstm_params["output_size"],
            resolved_lstm_params["lookback"],
            resolved_lstm_params["pretrain_lr"],
        )

    train_data, validation_data, test_data = prepare_model_datasets(
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        artifact_folder=window_folder,
        augment_oversample_ratio=(best_hyperparameters or {}).get("synthetic_oversample_ratio"),
    )

    logger.info(f"Model input columns ({len(train_data.columns)}): {train_data.columns.tolist()}")

    # Validate warm-start model path
    if prev_model_path and not os.path.exists(f"{prev_model_path}.zip"):
        logger.info(f"Previous model not found at {prev_model_path}.zip, training from scratch")
        prev_model_path = None

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
        tuning_trials=tuning_trials,
        model_params=best_hyperparameters,  # Pass the best hyperparameters to use
        window_label=f"[W{window_idx+1}/{num_windows}] ",
        prev_model_path=prev_model_path
    )

    # Save loss history if available
    if "loss_history" in training_stats and training_stats["loss_history"]:
        loss_history_path = f"{window_folder}/loss_history.json"
        save_json(training_stats["loss_history"], loss_history_path)

    # Plot training progress (pass iterations list)
    plot_training_progress(training_stats.get("iterations", training_stats), window_folder)

    # Load the best model for testing (try RecurrentPPO first if enabled)
    seq_config = config.get("sequence_model", {})
    use_recurrent = seq_config.get("enabled", False) and RECURRENT_PPO_AVAILABLE

    if use_recurrent:
        try:
            model = RecurrentPPO.load(f"{window_folder}/model")
        except Exception:
            model = PPO.load(f"{window_folder}/model")
    else:
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
    stop_loss_mode = "percentage"
    take_profit_mode = "percentage"
    stop_loss_atr_multiplier = None
    take_profit_atr_multiplier = None

    # Only set risk parameters if risk management is enabled
    if risk_enabled:
        # Daily risk limit configuration
        daily_risk_config = risk_config.get("daily_risk_limit", {})
        if daily_risk_config.get("enabled", False):
            daily_risk_limit = daily_risk_config.get("max_daily_loss", 1000.0)

        # Stop loss configuration
        stop_loss_config = risk_config.get("stop_loss", {})
        if stop_loss_config.get("enabled", False):
            stop_loss_mode = stop_loss_config.get("mode", "percentage")
            if stop_loss_mode == "percentage":
                stop_loss_pct = stop_loss_config.get("percentage", 0.0)
            elif stop_loss_mode == "atr":
                stop_loss_atr_multiplier = stop_loss_config.get("atr_multiplier", 2.0)

        # Take profit configuration
        take_profit_config = risk_config.get("take_profit", {})
        if take_profit_config.get("enabled", False):
            take_profit_mode = take_profit_config.get("mode", "percentage")
            if take_profit_mode == "percentage":
                take_profit_pct = take_profit_config.get("percentage", 0.0)
            elif take_profit_mode == "atr":
                take_profit_atr_multiplier = take_profit_config.get("atr_multiplier", 3.0)

        # Trailing stop configuration
        trailing_stop_config = risk_config.get("trailing_stop", {})
        if trailing_stop_config.get("enabled", False):
            trailing_stop_pct = trailing_stop_config.get("percentage", 0.0)

        # Position sizing configuration
        position_sizing_config = risk_config.get("position_sizing", {})
        if position_sizing_config.get("enabled", False):
            position_size = position_sizing_config.get("size_multiplier", 1.0)
            max_risk_per_trade_pct = position_sizing_config.get("max_risk_per_trade_percentage", 0.0)

    # Evaluate with risk management if enabled
    if risk_enabled:

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
            daily_risk_limit=daily_risk_limit,
            stop_loss_mode=stop_loss_mode,
            take_profit_mode=take_profit_mode,
            stop_loss_atr_multiplier=stop_loss_atr_multiplier,
            take_profit_atr_multiplier=take_profit_atr_multiplier
        )
    else:
        # Evaluate without risk management
        test_env = TradingEnv(
            test_data,
            initial_balance=config["environment"]["initial_balance"],
            transaction_cost=config["environment"].get("transaction_cost", 2.50)
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
    
    max_dd = test_results.get('max_drawdown', 0.0)
    calmar = test_results.get('calmar_ratio', 0.0)
    sortino = test_results.get('sortino_ratio', 0.0)
    window_result["max_drawdown"] = max_dd
    window_result["calmar_ratio"] = calmar
    window_result["sortino_ratio"] = sortino
    ret_pct = test_results['total_return_pct']
    action_dist = format_action_distribution(test_results.get('action_history'))
    logger.info(
        f"Window {window_idx}: Return={color_pct(ret_pct)}, "
        f"Sortino={bold(f'{sortino:.2f}')}, MaxDD={max_dd:.2f}%, Calmar={calmar:.2f}, "
        f"Portfolio=${test_results['final_portfolio_value']:.2f}"
    )
    logger.info(f"Window {window_idx} actions: {action_dist}")

    if "trade_history" in test_results:
        window_result["has_trade_history"] = True

        # Only save trade history if there are actual trades in the history
        if test_results["trade_history"] and len(test_results["trade_history"]) > 0:
            trade_history_path = f'{window_folder}/trade_history.csv'
            save_trade_history(test_results["trade_history"], trade_history_path)
    else:
        window_result["has_trade_history"] = False
    
    return window_result

def walk_forward_testing(
    data: pd.DataFrame,
    window_size: int,
    step_size: int,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    embargo_days: int = 0,
    initial_timesteps: int = 10000,
    additional_timesteps: int = 5000,
    max_iterations: int = 10,
    n_stagnant_loops: int = 3,
    improvement_threshold: float = 0.1,
    run_hyperparameter_tuning: bool = False,
    tuning_trials: int = 30,
    max_windows: int = 0
) -> Dict:
    """
    Perform walk-forward testing with anchored walk-forward analysis.
    """
    # Create session folder within models directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_folder = f'models/session_{timestamp}'
    os.makedirs(f'{session_folder}/models', exist_ok=True)
    os.makedirs(f'{session_folder}/plots', exist_ok=True)
    os.makedirs(f'{session_folder}/reports', exist_ok=True)
    
    logger.info(f"Created session folder: {session_folder}")
    
    # Initialize all_window_results list
    all_window_results = []
    
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
    
    # Filter data to include only market hours if configured
    if config.get("data", {}).get("market_hours_only", True):
        logger.info("Filtering data to NYSE market hours only")
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
    if max_windows > 0:
        num_windows = min(num_windows, max_windows)
        logger.info(f"Limiting to {num_windows} windows (max_windows={max_windows})")
    logger.info(f"Number of walk-forward windows: {num_windows}")
    
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
    best_hyperparameters = None
    lstm_feature_params = None
    tuning_metadata = None
    
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
        # Handle both tz-aware and tz-naive timestamps
        if data_eastern.index.tz is None:
            data_eastern.index = data_eastern.index.tz_localize('UTC').tz_convert(eastern)
        else:
            data_eastern.index = data_eastern.index.tz_convert(eastern)
        
        # Extract data for this window by trading days
        window_mask = (data_eastern.index.date.astype(str) >= start_day) & (data_eastern.index.date.astype(str) <= end_day)
        window_data = data_eastern[window_mask].copy()

        # Convert back to original timezone (UTC if it was tz-aware, remove tz if it was naive)
        if data.index.tz is None:
            window_data.index = window_data.index.tz_localize(None)
        else:
            window_data.index = window_data.index.tz_convert('UTC')
        
        # Split window into train, validation, and test sets with embargo gap
        train_idx = int(len(window_data) * train_ratio)
        validation_idx = train_idx + int(len(window_data) * validation_ratio)

        # Calculate embargo in bars (approximately 78 bars per day for market hours, 288 for 24h)
        # Use a rough estimate of bars per day based on data frequency
        if len(window_data) > 1:
            time_diff = (window_data.index[1] - window_data.index[0]).total_seconds()
            bars_per_day = int(24 * 60 * 60 / time_diff) if time_diff > 0 else 288
        else:
            bars_per_day = 288
        embargo_bars = embargo_days * bars_per_day

        train_data = window_data.iloc[:train_idx].copy()
        validation_data = window_data.iloc[train_idx:validation_idx].copy()

        # Apply embargo: skip embargo_bars between validation and test
        test_start_idx = validation_idx + embargo_bars
        if test_start_idx >= len(window_data):
            # Reduce embargo to maximum feasible size (ensure at least 5% of window for test)
            min_test_bars = max(int(len(window_data) * 0.05), 10)
            test_start_idx = max(validation_idx + 1, len(window_data) - min_test_bars)
            actual_embargo = test_start_idx - validation_idx
            logger.warning(f"Window {i+1} - Embargo reduced from {embargo_bars} to {actual_embargo} bars")
        test_data = window_data.iloc[test_start_idx:].copy()

        logger.info(f"Window {i+1} - Train: {train_data.index[0]} to {train_data.index[-1]} ({len(train_data)} bars)")
        logger.info(f"Window {i+1} - Val: {validation_data.index[0]} to {validation_data.index[-1]} ({len(validation_data)} bars)")
        if embargo_bars > 0 and test_start_idx > validation_idx:
            logger.info(f"Window {i+1} - Embargo: {embargo_bars} bars ({embargo_days} days)")
        logger.info(f"Window {i+1} - Test: {test_data.index[0]} to {test_data.index[-1]} ({len(test_data)} bars)")
        
        # Store window data
        window_data_list.append({
            "window_idx": i+1,
            "window_data": window_data,
            "train_data": train_data,
            "validation_data": validation_data,
            "test_data": test_data,
            "window_folder": window_folder
        })
    
    if run_hyperparameter_tuning:
        logger.info("Starting staged hyperparameter tuning before walk-forward execution")
        tuning_result = run_staged_hyperparameter_tuning(
            data=data,
            window_data_list=window_data_list,
            session_folder=session_folder,
            eval_metric=eval_metric,
            fallback_trials=tuning_trials,
        )
        best_hyperparameters = tuning_result["best_hyperparameters"]
        lstm_feature_params = tuning_result.get("lstm_feature_params")
        tuning_metadata = tuning_result

        session_params["tuning_scope"] = tuning_result.get("scope")
        session_params["tuning_period"] = tuning_result.get("tuning_period")
        session_params["tuning_stage_winners"] = tuning_result.get("stage_winners")
        session_params["tuned_hyperparameters"] = best_hyperparameters
        save_json(session_params, f'{session_folder}/reports/session_parameters.json')
    
    # Process windows - either in parallel or sequentially
    # Note: With LSTM features, each window trains its own autoencoder
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
                        False,  # Don't run hyperparameter tuning for each window
                        tuning_trials,
                        best_hyperparameters,
                        lstm_feature_params,
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

        prev_lstm_checkpoint_path = None  # For LSTM warm-starting between windows
        prev_model_path = None  # For policy warm-starting between windows
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
                    False,  # Don't run hyperparameter tuning for each window
                    tuning_trials,
                    best_hyperparameters,
                    lstm_feature_params,
                    prev_lstm_checkpoint_path=prev_lstm_checkpoint_path,
                    prev_model_path=prev_model_path
                )
                all_window_results.append(window_result)
                # Update warm-start paths for next window
                prev_lstm_checkpoint_path = f"{window_data_dict['window_folder']}/lstm_autoencoder_checkpoint.pt"
                prev_model_path = f"{window_data_dict['window_folder']}/model"
            except Exception as e:
                logger.error(f"Error processing window {window_data_dict['window_idx']}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # Aggregate results across all windows
    returns = [res["return"] for res in all_window_results]
    portfolio_values = [res["portfolio_value"] for res in all_window_results]
    trade_counts = [res["trade_count"] for res in all_window_results]
    sortinos = [res.get("sortino_ratio", 0.0) for res in all_window_results]
    
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
    avg_sortino = float(np.mean(sortinos)) if sortinos else 0.0
    avg_hit_rate = np.mean(hit_rates) if hit_rates else 0
    avg_profitable_trades = np.mean(profitable_trades) if profitable_trades else 0
    avg_prediction_accuracy = np.mean(prediction_accuracies) if prediction_accuracies else 0
    avg_correct_predictions = np.mean(correct_predictions) if correct_predictions else 0
    avg_total_predictions = np.mean(total_predictions) if total_predictions else 0
    
    logger.info(f"\n{'='*80}\nWalk-Forward Testing Summary\n{'='*80}")
    logger.info(f"Number of windows: {num_windows}")
    logger.info(f"Average return: {color_pct(avg_return)}")
    logger.info(f"Average Sortino: {bold(f'{avg_sortino:.2f}')}")
    
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
        "avg_sortino": avg_sortino,
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
    if tuning_metadata is not None:
        summary_results["tuning_scope"] = tuning_metadata.get("scope")
        summary_results["tuning_period"] = tuning_metadata.get("tuning_period")
        summary_results["tuning_stage_winners"] = tuning_metadata.get("stage_winners")
        summary_results["tuned_hyperparameters"] = tuning_metadata.get("best_hyperparameters")
    
    # Don't include all window results in the summary JSON (they may contain non-serializable objects)
    save_json(summary_results, f'{session_folder}/reports/summary_results.json')
    
    # Add window results back for the return value (not for JSON serialization)
    summary_results["all_window_results"] = all_window_results
    
    # Plot results
    plot_walk_forward_results(all_window_results, session_folder, eval_metric)
    
    # Export consolidated trade history
    export_consolidated_trade_history(all_window_results, session_folder)
    
    return summary_results

def get_default_stage_params(parameter_names: List[str]) -> Dict[str, Any]:
    """Return current config defaults for the requested tuning parameters."""
    defaults = {}
    reward_config = config.get("reward", {})
    model_config = config.get("model", {})
    seq_config = config.get("sequence_model", {})
    aug_config = config.get("augmentation", {}).get("synthetic_bears", {})

    for name in parameter_names:
        if name == "reward_loss_multiplier":
            defaults[name] = reward_config.get("loss_multiplier", 0.7)
        elif name == "reward_turnover_penalty":
            defaults[name] = reward_config.get("turnover_penalty", 0.05)
        elif name == "reward_calm_holding_bonus":
            defaults[name] = reward_config.get("calm_holding_bonus", 0.0005)
        elif name == "synthetic_oversample_ratio":
            defaults[name] = aug_config.get("oversample_ratio", 0.3)
        elif name in {"learning_rate", "ent_coef", "gamma", "gae_lambda"}:
            defaults[name] = model_config.get(name)
        elif name in {"n_steps", "batch_size"}:
            defaults[name] = int(model_config.get(name))
        elif name == "lstm_hidden_size":
            defaults[name] = int(seq_config.get("lstm_hidden_size", 256))
        elif name == "n_lstm_layers":
            defaults[name] = int(seq_config.get("n_lstm_layers", 1))

    return defaults


def select_tuning_datasets(data: pd.DataFrame, window_data_list: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Select the tuning train/validation datasets according to the configured scope."""
    tuning_config = config.get("hyperparameter_tuning", {})
    scope_config = tuning_config.get("scope", {})
    scope_mode = scope_config.get("mode", "dedicated_global")

    filtered_data = data.copy()
    if config.get("data", {}).get("market_hours_only", True):
        filtered_data = filter_market_hours(filtered_data)

    if scope_mode == "first_window":
        first_window = window_data_list[0]
        train_data = first_window["train_data"].copy()
        validation_data = first_window["validation_data"].copy()
        metadata = {
            "mode": "first_window",
            "requested_start": str(train_data.index[0]),
            "requested_end": str(validation_data.index[-1]),
            "actual_start": str(train_data.index[0]),
            "actual_end": str(validation_data.index[-1]),
            "train_rows": len(train_data),
            "validation_rows": len(validation_data),
        }
        return train_data, validation_data, metadata

    def _slice_by_date_range(frame: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        sliced = frame
        index = sliced.index
        if start_date:
            start_ts = pd.Timestamp(start_date)
            if index.tz is not None and start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize(index.tz)
            sliced = sliced.loc[sliced.index >= start_ts]
        if end_date:
            end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
            if index.tz is not None and end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize(index.tz)
            sliced = sliced.loc[sliced.index < end_ts]
        return sliced.copy()

    global_period = scope_config.get("global_period", {})
    train_period = scope_config.get("train_period", {})
    validation_period = scope_config.get("validation_period", {})
    if train_period or validation_period:
        train_start = train_period.get("start_date")
        train_end = train_period.get("end_date")
        validation_start = validation_period.get("start_date")
        validation_end = validation_period.get("end_date")

        train_data = _slice_by_date_range(filtered_data, train_start, train_end)
        validation_data = _slice_by_date_range(filtered_data, validation_start, validation_end)
        if train_data.empty or validation_data.empty:
            raise ValueError("Configured train_period/validation_period produced an empty tuning split")
        if len(train_data) < 20 or len(validation_data) < 10:
            raise ValueError("Configured train_period/validation_period is too short for stable tuning")

        metadata = {
            "mode": "dedicated_global",
            "requested_start": train_start,
            "requested_end": validation_end,
            "actual_start": str(train_data.index[0]),
            "actual_end": str(validation_data.index[-1]),
            "train_rows": len(train_data),
            "validation_rows": len(validation_data),
            "train_period": {"start_date": train_start, "end_date": train_end},
            "validation_period": {"start_date": validation_start, "end_date": validation_end},
        }
        return train_data, validation_data, metadata

    start_date = global_period.get("start_date")
    end_date = global_period.get("end_date")
    selected_data = _slice_by_date_range(filtered_data, start_date, end_date)
    if selected_data.empty:
        raise ValueError("Configured global tuning period produced an empty dataset")

    train_ratio = float(scope_config.get("train_ratio", 0.8))
    validation_ratio = float(scope_config.get("validation_ratio", 0.2))
    if train_ratio <= 0 or validation_ratio <= 0:
        raise ValueError("Global tuning train_ratio and validation_ratio must both be positive")

    ratio_total = train_ratio + validation_ratio
    train_cutoff = int(len(selected_data) * (train_ratio / ratio_total))
    if train_cutoff <= 0 or train_cutoff >= len(selected_data):
        raise ValueError("Global tuning split produced an empty train or validation partition")

    train_data = selected_data.iloc[:train_cutoff].copy()
    validation_data = selected_data.iloc[train_cutoff:].copy()
    if len(train_data) < 20 or len(validation_data) < 10:
        raise ValueError("Global tuning split is too short for stable tuning")

    metadata = {
        "mode": "dedicated_global",
        "requested_start": start_date,
        "requested_end": end_date,
        "actual_start": str(selected_data.index[0]),
        "actual_end": str(selected_data.index[-1]),
        "train_rows": len(train_data),
        "validation_rows": len(validation_data),
    }
    return train_data, validation_data, metadata


def save_trial_records(trial_records: List[Dict[str, Any]], json_path: Optional[str] = None, csv_path: Optional[str] = None) -> None:
    """Persist raw trial history as JSON and a flattened CSV."""
    ordered_records = sorted(trial_records, key=lambda record: (record.get("stage", ""), record.get("trial_number", -1)))
    if json_path:
        save_json(ordered_records, json_path)
    if csv_path:
        flattened_records = []
        for record in ordered_records:
            flat_record = {key: value for key, value in record.items() if key != "params"}
            for param_name, param_value in record.get("params", {}).items():
                flat_record[f"param_{param_name}"] = param_value
            flattened_records.append(flat_record)
        pd.DataFrame(flattened_records).to_csv(csv_path, index=False)


def hyperparameter_tuning(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    n_trials: int = 30,
    window_folder: str = None,
    eval_metric: str = "return",
    hit_rate_min_trades: int = 5,
    stage_name: str = "default",
    parameter_names: Optional[List[str]] = None,
    tuning_timesteps: Optional[int] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    storage_path: Optional[str] = None,
    base_seed: Optional[int] = None,
) -> Dict:
    """
    Run hyperparameter tuning using Optuna with stage-specific parameter subsets.
    """
    from threading import Lock

    logger.info("Starting %s hyperparameter tuning with %s trials", stage_name, n_trials)
    logger.info("Evaluation metric: %s", eval_metric)

    tuning_config = config.get("hyperparameter_tuning", {})
    hp_config = tuning_config.get("parameters", {})
    parallel_config = tuning_config.get("parallel_processing", {})
    pruner_config = tuning_config.get("pruner", {})
    use_parallel = parallel_config.get("enabled", True)
    n_jobs = parallel_config.get("n_jobs", 0)
    if n_jobs <= 0:
        n_jobs = multiprocessing.cpu_count()
    if not use_parallel:
        n_jobs = 1

    if parameter_names is None:
        parameter_names = list(hp_config.keys())
    fixed_params = dict(fixed_params or {})
    tuning_timesteps = int(tuning_timesteps or tuning_config.get("tuning_timesteps", 20000))
    base_seed = int(config.get("seed", 42) if base_seed is None else base_seed)
    chunk_size = 5000
    sentinel_score = -1e12

    seq_config = config.get("sequence_model", {})
    use_recurrent = seq_config.get("enabled", False) and RECURRENT_PPO_AVAILABLE
    study_name = f"{stage_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    sampler = optuna.samplers.TPESampler(
        seed=base_seed,
        multivariate=True,
        n_startup_trials=max(5, int(pruner_config.get("n_startup_trials", 5))),
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(pruner_config.get("n_startup_trials", 5)),
        n_warmup_steps=int(pruner_config.get("n_warmup_reports", 2)),
    )

    if storage_path is None and window_folder:
        storage_path = os.path.join(window_folder, "optuna.db")
    storage = f"sqlite:///{storage_path}" if storage_path else None

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    int_params = {"n_steps", "batch_size", "n_lstm_layers"}
    trial_records: List[Dict[str, Any]] = []
    trial_lock = Lock()

    def sample_param(trial: optuna.Trial, param_name: str) -> Any:
        param_config = hp_config.get(param_name, {})
        if param_name == "lstm_hidden_size" and not use_recurrent:
            return seq_config.get("lstm_hidden_size", 256)
        if param_name == "n_lstm_layers" and not use_recurrent:
            return seq_config.get("n_lstm_layers", 1)
        if "choices" in param_config:
            return trial.suggest_categorical(param_name, param_config["choices"])
        if "min" in param_config and "max" in param_config:
            if param_name in int_params:
                return trial.suggest_int(
                    param_name,
                    int(param_config["min"]),
                    int(param_config["max"]),
                    log=param_config.get("log", False),
                )
            return trial.suggest_float(
                param_name,
                float(param_config["min"]),
                float(param_config["max"]),
                log=param_config.get("log", False),
            )
        return get_default_stage_params([param_name]).get(param_name)

    def record_trial(status: str, trial_number: int, params: Dict[str, Any], objective_value: float, seed: int,
                     results: Optional[Dict[str, Any]] = None, reason: Optional[str] = None) -> None:
        results = results or {}
        record = {
            "stage": stage_name,
            "trial_number": trial_number,
            "params": params,
            "objective_metric": objective_value,
            "return": results.get("total_return_pct"),
            "sortino": results.get("sortino_ratio"),
            "calmar": results.get("calmar_ratio"),
            "max_drawdown": results.get("max_drawdown"),
            "trade_count": results.get("trade_count"),
            "status": status,
            "reason": reason,
            "seed": seed,
        }
        with trial_lock:
            trial_records.append(record)

    def evaluate_model(model: Any, prepared_validation_data: pd.DataFrame) -> Dict[str, Any]:
        if eval_metric == "prediction_accuracy":
            return evaluate_agent_prediction_accuracy(model, prepared_validation_data, verbose=0, deterministic=True)
        return evaluate_agent(model, prepared_validation_data, verbose=0, deterministic=True)

    def build_model(train_env: TradingEnv, params: Dict[str, Any], trial_seed: int) -> Any:
        learning_rate = params.get("learning_rate", config["model"].get("learning_rate", 0.0003))
        n_steps = int(params.get("n_steps", config["model"].get("n_steps", 2048)))
        batch_size = int(params.get("batch_size", config["model"].get("batch_size", 64)))
        ent_coef = params.get("ent_coef", config["model"].get("ent_coef", 0.01))
        gamma = params.get("gamma", config["model"].get("gamma", 0.99))
        gae_lambda = params.get("gae_lambda", config["model"].get("gae_lambda", 0.95))
        device = get_device(seq_config.get("device", "auto"), for_recurrent=use_recurrent)

        if use_recurrent:
            shared_lstm = seq_config.get("shared_lstm", False)
            return RecurrentPPO(
                "MlpLstmPolicy",
                train_env,
                verbose=0,
                learning_rate=learning_rate,
                n_steps=n_steps,
                ent_coef=ent_coef,
                batch_size=batch_size,
                gamma=gamma,
                gae_lambda=gae_lambda,
                seed=trial_seed,
                device=device,
                policy_kwargs={
                    "lstm_hidden_size": int(params.get("lstm_hidden_size", seq_config.get("lstm_hidden_size", 256))),
                    "n_lstm_layers": int(params.get("n_lstm_layers", seq_config.get("n_lstm_layers", 1))),
                    "shared_lstm": shared_lstm,
                    "enable_critic_lstm": not shared_lstm,
                    "net_arch": {"pi": [128, 64], "vf": [128, 64]},
                },
            )

        return PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            learning_rate=learning_rate,
            n_steps=n_steps,
            ent_coef=ent_coef,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            seed=trial_seed,
            device=device,
            policy_kwargs={"net_arch": [128, 64]},
        )

    def objective(trial: optuna.Trial) -> float:
        trial_seed = base_seed + trial.number
        params = dict(fixed_params)
        for param_name in parameter_names:
            params[param_name] = sample_param(trial, param_name)

        reward_overrides = {}
        if "reward_loss_multiplier" in params:
            reward_overrides["loss_multiplier"] = float(params["reward_loss_multiplier"])
        if "reward_turnover_penalty" in params:
            reward_overrides["turnover_penalty"] = float(params["reward_turnover_penalty"])
        if "reward_calm_holding_bonus" in params:
            reward_overrides["calm_holding_bonus"] = float(params["reward_calm_holding_bonus"])

        prepared_train = None
        train_env = None
        model = None
        latest_results = None
        latest_metric = sentinel_score
        try:
            prepared_train, prepared_validation, _ = prepare_model_datasets(
                train_data=train_data,
                validation_data=validation_data,
                test_data=None,
                artifact_folder=None,
                augment_oversample_ratio=params.get("synthetic_oversample_ratio"),
                augmentation_seed=trial_seed,
            )
            train_env = TradingEnv(
                prepared_train,
                initial_balance=config["environment"]["initial_balance"],
                transaction_cost=config["environment"].get("transaction_cost", 2.50),
                position_size=config["environment"].get("position_size", 1),
                reward_overrides=reward_overrides,
            )
            model = build_model(train_env, params, trial_seed)

            total_steps = 0
            report_idx = 0
            while total_steps < tuning_timesteps:
                learn_steps = min(chunk_size, tuning_timesteps - total_steps)
                model.learn(total_timesteps=learn_steps, reset_num_timesteps=False, progress_bar=False)
                total_steps += learn_steps
                latest_results = evaluate_model(model, prepared_validation)
                latest_metric = get_objective_metric(latest_results, eval_metric)
                trial.report(latest_metric, report_idx)
                report_idx += 1
                if trial.should_prune():
                    record_trial("pruned", trial.number, params, latest_metric, trial_seed, latest_results, "trial pruned by Optuna")
                    raise optuna.TrialPruned()

            trade_count = int(latest_results.get("trade_count", 0))
            if eval_metric == "hit_rate":
                min_trades = int(hit_rate_min_trades)
            elif eval_metric == "prediction_accuracy":
                min_trades = 0
            else:
                min_trades = max(20, len(prepared_validation) // 200)

            if trade_count < min_trades:
                record_trial(
                    "rejected",
                    trial.number,
                    params,
                    sentinel_score,
                    trial_seed,
                    latest_results,
                    f"too_few_trades ({trade_count}<{min_trades})",
                )
                logger.info(
                    "[%s] Trial %s rejected: too few trades (%s<%s), return=%s, sortino=%s, calmar=%.4f, params=%s",
                    stage_name,
                    trial.number,
                    trade_count,
                    min_trades,
                    color_pct(latest_results.get("total_return_pct", 0.0)),
                    bold(f"{latest_results.get('sortino_ratio', 0.0):.2f}"),
                    latest_results.get("calmar_ratio", 0.0),
                    params,
                )
                return sentinel_score

            total_return_pct = float(latest_results.get("total_return_pct", 0.0))
            if eval_metric in {"sortino", "calmar"} and total_return_pct <= 0:
                record_trial(
                    "rejected",
                    trial.number,
                    params,
                    sentinel_score,
                    trial_seed,
                    latest_results,
                    f"non_positive_return ({total_return_pct:.2f}%)",
                )
                logger.info(
                    "[%s] Trial %s rejected: non-positive return (%s), sortino=%s, calmar=%.4f, trades=%s, params=%s",
                    stage_name,
                    trial.number,
                    color_pct(total_return_pct),
                    bold(f"{latest_results.get('sortino_ratio', 0.0):.2f}"),
                    latest_results.get("calmar_ratio", 0.0),
                    trade_count,
                    params,
                )
                return sentinel_score

            action_dist = format_action_distribution(latest_results.get("action_counts", latest_results.get("action_history")))
            logger.info(
                "[%s] Trial %s: %s=%.4f, return=%s, sortino=%s, calmar=%.4f, trades=%s, actions=[%s], params=%s",
                stage_name,
                trial.number,
                eval_metric,
                latest_metric,
                color_pct(latest_results.get("total_return_pct", 0.0)),
                bold(f"{latest_results.get('sortino_ratio', 0.0):.2f}"),
                latest_results.get("calmar_ratio", 0.0),
                latest_results.get("trade_count", 0),
                action_dist,
                params,
            )
            record_trial("completed", trial.number, params, latest_metric, trial_seed, latest_results)
            return latest_metric
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            logger.warning("[%s] Trial %s failed: %s", stage_name, trial.number, exc)
            record_trial("failed", trial.number, params, sentinel_score, trial_seed, latest_results, str(exc))
            return sentinel_score
        finally:
            if train_env is not None:
                train_env.close()

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=False)

    completed_records = [record for record in trial_records if record["status"] == "completed"]
    if not completed_records:
        best_params = get_default_stage_params(parameter_names)
        best_value = float(sentinel_score)
        logger.warning(
            "[%s] No completed Optuna trials survived guardrails; falling back to current config defaults for this stage",
            stage_name,
        )
    else:
        best_params = dict(study.best_params)
        best_value = float(study.best_value)
    logger.info("[%s] Best %s: %.4f", stage_name, eval_metric, best_value)
    for param, value in best_params.items():
        logger.info("  %s: %s", param, value)

    return {
        "best_params": best_params,
        "best_value": best_value,
        "study": study,
        "trial_records": trial_records,
        "stage_name": stage_name,
    }


def run_staged_hyperparameter_tuning(
    data: pd.DataFrame,
    window_data_list: List[Dict[str, Any]],
    session_folder: str,
    eval_metric: str,
    fallback_trials: int,
) -> Dict[str, Any]:
    """Run the global two-stage RL tuner once and persist its artifacts."""
    reports_dir = f"{session_folder}/reports"
    tuning_config = config.get("hyperparameter_tuning", {})
    stage_config = tuning_config.get("stages", {})
    reporting_config = tuning_config.get("reporting", {})
    scope_config = tuning_config.get("scope", {})
    storage_path = os.path.join(reports_dir, "optuna.db")

    raw_tuning_train, raw_tuning_validation, scope_metadata = select_tuning_datasets(data, window_data_list)

    lstm_feature_params = get_lstm_feature_params(
        train_data=raw_tuning_train,
        artifact_folder=reports_dir,
        provided_params=None,
        allow_tuning=True,
    )
    tuning_train, tuning_validation, _, lstm_feature_params = apply_lstm_features_to_datasets(
        train_data=raw_tuning_train,
        validation_data=raw_tuning_validation,
        test_data=None,
        artifact_folder=reports_dir,
        lstm_params=lstm_feature_params,
        save_artifacts=True,
    )

    stage1_names = [
        "reward_loss_multiplier",
        "reward_turnover_penalty",
        "reward_calm_holding_bonus",
        "synthetic_oversample_ratio",
    ]
    stage2_names = [
        "learning_rate",
        "n_steps",
        "batch_size",
        "ent_coef",
        "gamma",
        "gae_lambda",
        "lstm_hidden_size",
        "n_lstm_layers",
    ]

    all_trial_records: List[Dict[str, Any]] = []
    stage_winners: Dict[str, Dict[str, Any]] = {}

    reward_stage_cfg = stage_config.get("reward_and_augmentation", {})
    if reward_stage_cfg.get("enabled", True):
        reward_stage = hyperparameter_tuning(
            train_data=tuning_train,
            validation_data=tuning_validation,
            n_trials=int(reward_stage_cfg.get("n_trials", fallback_trials)),
            window_folder=reports_dir,
            eval_metric=eval_metric,
            stage_name="reward_and_augmentation",
            parameter_names=stage1_names,
            tuning_timesteps=int(reward_stage_cfg.get("tuning_timesteps", 30000)),
            fixed_params={},
            storage_path=storage_path,
            base_seed=int(config.get("seed", 42)) + 1000,
        )
        stage_winners["reward_and_augmentation"] = reward_stage["best_params"]
        all_trial_records.extend(reward_stage["trial_records"])
        save_json(reward_stage["best_params"], f"{reports_dir}/global_tuning_stage1_best_params.json")
    else:
        stage_winners["reward_and_augmentation"] = get_default_stage_params(stage1_names)

    ppo_stage_cfg = stage_config.get("ppo_and_sequence", {})
    fixed_stage2_params = dict(stage_winners["reward_and_augmentation"])
    if ppo_stage_cfg.get("enabled", True):
        ppo_stage = hyperparameter_tuning(
            train_data=tuning_train,
            validation_data=tuning_validation,
            n_trials=int(ppo_stage_cfg.get("n_trials", fallback_trials)),
            window_folder=reports_dir,
            eval_metric=eval_metric,
            stage_name="ppo_and_sequence",
            parameter_names=stage2_names,
            tuning_timesteps=int(ppo_stage_cfg.get("tuning_timesteps", 30000)),
            fixed_params=fixed_stage2_params,
            storage_path=storage_path,
            base_seed=int(config.get("seed", 42)) + 2000,
        )
        stage_winners["ppo_and_sequence"] = ppo_stage["best_params"]
        all_trial_records.extend(ppo_stage["trial_records"])
        save_json(ppo_stage["best_params"], f"{reports_dir}/global_tuning_stage2_best_params.json")
    else:
        stage_winners["ppo_and_sequence"] = get_default_stage_params(stage2_names)

    best_hyperparameters = {}
    best_hyperparameters.update(stage_winners["reward_and_augmentation"])
    best_hyperparameters.update(stage_winners["ppo_and_sequence"])
    save_json(best_hyperparameters, f"{reports_dir}/global_tuning_best_params.json")

    if reporting_config.get("save_trials_json", True) or reporting_config.get("save_trials_csv", True):
        save_trial_records(
            all_trial_records,
            json_path=f"{reports_dir}/global_tuning_trials.json" if reporting_config.get("save_trials_json", True) else None,
            csv_path=f"{reports_dir}/global_tuning_trials.csv" if reporting_config.get("save_trials_csv", True) else None,
        )

    summary = {
        "scope": scope_config.get("mode", "dedicated_global"),
        "tuning_period": scope_metadata,
        "evaluation_metric": eval_metric,
        "stage_winners": stage_winners,
        "best_hyperparameters": best_hyperparameters,
        "lstm_feature_params": lstm_feature_params,
    }
    save_json(summary, f"{reports_dir}/global_tuning_summary.json")

    apply_best_hyperparameters_to_runtime_config(best_hyperparameters)
    save_best_hyperparameters_to_config(best_hyperparameters)

    return {
        "scope": scope_config.get("mode", "dedicated_global"),
        "tuning_period": scope_metadata,
        "stage_winners": stage_winners,
        "best_hyperparameters": best_hyperparameters,
        "lstm_feature_params": lstm_feature_params,
        "trial_records": all_trial_records,
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
        
        # Add marker for best models
        if best:
            plt.text(iteration, value, '*',
                    ha='center', va='top', fontsize=14, fontweight='bold', color='gold')
    
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
    import argparse
    parser = argparse.ArgumentParser(description="Walk-forward testing")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Scenario name from scenarios.yaml (e.g. steady_bull_2021)")
    args = parser.parse_args()

    # Create model directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/logs', exist_ok=True)

    # Load scenario config if specified
    data_file = "data/NQ_2024_unix.csv"
    start_date = None
    end_date = None

    if args.scenario:
        with open("scenarios.yaml", "r") as f:
            scenarios = yaml.safe_load(f).get("scenarios", {})
        if args.scenario not in scenarios:
            print(f"ERROR: Unknown scenario '{args.scenario}'. Available: {list(scenarios.keys())}")
            return
        scenario = scenarios[args.scenario]
        data_file = scenario["data_file"]
        start_date = scenario.get("start_date")
        end_date = scenario.get("end_date")
        logger.info(f"Running scenario '{args.scenario}': {scenario['description']}")
        logger.info(f"  Data file: {data_file}, Date range: {start_date} to {end_date}")

    # Load data from TradingView CSV instead of Yahoo Finance
    full_data = load_tradingview_data(data_file)

    # Check if data loading was successful
    if full_data is None or len(full_data) == 0:
        logger.error("Failed to load TradingView data or dataset is empty. Check your data file.")
        print("\nERROR: Data loading failed. Please check the TradingView data file and ensure it contains valid data.")
        return

    # Filter by date range if scenario specified dates
    if start_date:
        full_data = full_data[full_data.index >= pd.Timestamp(start_date)]
    if end_date:
        full_data = full_data[full_data.index <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]

    logger.info(f"Loaded {len(full_data)} rows from {full_data.index[0].strftime('%Y-%m-%d')} to {full_data.index[-1].strftime('%Y-%m-%d')}")
    
    # Get walk-forward parameters from config or use defaults
    wf_config = config.get("walk_forward", {})
    window_size = wf_config.get("window_size", 14)  # 14 trading days default
    step_size = wf_config.get("step_size", 7)       # 7 trading days default
    
    # Get evaluation metric from config
    eval_metric = config.get("training", {}).get("evaluation", {}).get("metric", "return")
    
    # Check if risk management is enabled
    risk_config = config.get("risk_management", {})
    risk_enabled = risk_config.get("enabled", False)
    
    if risk_enabled:
        # Log risk management settings
        logger.info("Risk management is ENABLED for walk-forward testing")

        # Dynamic SL/TP configuration
        dynamic_sl_tp_config = risk_config.get("dynamic_sl_tp", {})
        if dynamic_sl_tp_config.get("enabled", False):
            sl_range = dynamic_sl_tp_config.get("sl_multiplier_range", [1.5, 5.0])
            tp_range = dynamic_sl_tp_config.get("tp_multiplier_range", [1.5, 5.0])
            logger.info(f"  - Dynamic SL/TP: ENABLED (model chooses multipliers)")
            logger.info(f"    - SL range: {sl_range[0]}x - {sl_range[1]}x ATR")
            logger.info(f"    - TP range: {tp_range[0]}x - {tp_range[1]}x ATR")

        # Stop loss configuration
        stop_loss_config = risk_config.get("stop_loss", {})
        if stop_loss_config.get("enabled", False):
            stop_loss_mode = stop_loss_config.get("mode", "percentage")
            if stop_loss_mode == "atr":
                stop_loss_atr = stop_loss_config.get("atr_multiplier", 2.0)
                logger.info(f"  - Stop loss: {stop_loss_atr}x ATR")
            else:
                stop_loss_pct = stop_loss_config.get("percentage", 1.0)
                logger.info(f"  - Stop loss: {stop_loss_pct}%")
        else:
            logger.info("  - Stop loss: Disabled")

        # Take profit configuration
        take_profit_config = risk_config.get("take_profit", {})
        if take_profit_config.get("enabled", False):
            take_profit_mode = take_profit_config.get("mode", "percentage")
            if take_profit_mode == "atr":
                take_profit_atr = take_profit_config.get("atr_multiplier", 3.0)
                logger.info(f"  - Take profit: {take_profit_atr}x ATR")
            else:
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
        train_ratio=config["data"].get("train_ratio", 0.75),
        validation_ratio=config["data"].get("validation_ratio", 0.05),
        embargo_days=config["data"].get("embargo_days", 0),
        initial_timesteps=config["training"].get("total_timesteps", 10000),
        additional_timesteps=config["training"].get("additional_timesteps", 5000),
        max_iterations=config["training"].get("max_iterations", 10),
        n_stagnant_loops=config["training"].get("n_stagnant_loops", 3),
        improvement_threshold=config["training"].get("improvement_threshold", 0.1),
        run_hyperparameter_tuning=hyperparameter_tuning_enabled,
        tuning_trials=tuning_trials,
        max_windows=config.get("walk_forward", {}).get("max_windows", 0)
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
    print(f"Average return: {color_pct(results['avg_return'])}")
    _avg_sortino_str = f"{results.get('avg_sortino', 0.0):.2f}"
    print(f"Average Sortino: {bold(_avg_sortino_str)}")
    
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
