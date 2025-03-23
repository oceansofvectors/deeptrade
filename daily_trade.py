import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import os
from typing import Tuple, Dict, List
from dateutil.relativedelta import relativedelta
import optuna  # Add import for optuna

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from environment import TradingEnv
from config import config
import money
from trade import trade_with_risk_management, plot_results, save_trade_history
from get_data import filter_market_hours, get_data
# Import evaluate_agent_prediction_accuracy from walk_forward
from walk_forward import evaluate_agent_prediction_accuracy, calculate_hit_rate_from_trade_results

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

def split_today_and_training_window(data: pd.DataFrame, window_days: int = 14, 
                                   train_ratio: float = 0.7, 
                                   validation_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training, validation, and test data where test data is one day after validation data.
    
    Args:
        data: Processed DataFrame with DateTime index
        window_days: Number of trading days to use for training and validation window
        train_ratio: Proportion of window to use for training
        validation_ratio: Proportion of window to use for validation
        
    Returns:
        tuple: (training_data, validation_data, test_data)
    """
    # Get the most recent date in the data
    last_date = data.index.max()
    
    # Convert data index to Eastern Time for consistent day counting
    if data.index.tz is None:
        eastern_data_index = data.index.tz_localize('UTC').tz_convert('US/Eastern')
    else:
        eastern_data_index = data.index.tz_convert('US/Eastern')
    
    # Extract unique trading days
    unique_days = sorted(set(eastern_data_index.date.astype(str)))
    logger.info(f"Found {len(unique_days)} unique trading days in the dataset")
    
    # Make sure we have enough trading days
    if len(unique_days) <= window_days:
        logger.warning(f"Not enough trading days ({len(unique_days)}) for window size ({window_days}). Using all available data.")
        window_days = len(unique_days) - 1  # Leave at least one day for test
    
    # Select the trading days for our window
    window_days_list = unique_days[-window_days:]
    window_start_day = window_days_list[0]
    
    logger.info(f"Using window of {window_days} trading days from {window_start_day} to {window_days_list[-1]}")
    
    # Filter data to include only the window days
    window_data = data[data.index.tz_convert('US/Eastern').date.astype(str) >= window_start_day].copy()
    
    # Split historical data into training, validation and test by date
    # First determine size of training and validation periods in terms of data points
    historical_size = len(window_data)
    train_size = int(historical_size * train_ratio)
    validation_size = int(historical_size * validation_ratio)
    
    # Get the data for each period
    train_data = window_data.iloc[:train_size].copy()
    validation_data = window_data.iloc[train_size:train_size + validation_size].copy()
    
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

def hyperparameter_tuning(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    n_trials: int = 30,
    eval_metric: str = "return",
    hit_rate_min_trades: int = 5,
    min_predictions: int = 10
) -> Dict:
    """
    Run hyperparameter tuning using Optuna.
    
    Args:
        train_data: Training data DataFrame
        validation_data: Validation data DataFrame
        n_trials: Number of trials for Optuna optimization
        eval_metric: Evaluation metric to optimize for
        hit_rate_min_trades: Minimum number of trades required for hit rate to be meaningful
        min_predictions: Minimum number of predictions required for prediction accuracy to be meaningful
        
    Returns:
        Dict: Best hyperparameters and optimization results
    """
    logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
    logger.info(f"Evaluation metric: {eval_metric}")
    
    # Get risk management parameters from config
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
        
        logger.info(f"Risk management is enabled: SL={stop_loss_pct}%, TP={take_profit_pct}%, TS={trailing_stop_pct}%")
    else:
        logger.info("Risk management is disabled")
    
    # Get hyperparameter ranges from config
    hp_config = config.get("hyperparameter_tuning", {}).get("parameters", {})
    
    # Create Optuna study
    study_name = f"daily_hyperparam_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(direction="maximize", study_name=study_name)
    
    def objective(trial):
        # Define hyperparameters to tune using ranges from config
        learning_rate_cfg = hp_config.get("learning_rate", {})
        learning_rate = trial.suggest_float(
            "learning_rate", 
            learning_rate_cfg.get("min", 1e-5), 
            learning_rate_cfg.get("max", 1e-2), 
            log=learning_rate_cfg.get("log", True)
        )
        
        n_steps_cfg = hp_config.get("n_steps", {})
        n_steps = trial.suggest_int(
            "n_steps", 
            n_steps_cfg.get("min", 128), 
            n_steps_cfg.get("max", 2048), 
            log=n_steps_cfg.get("log", True)
        )
        
        ent_coef_cfg = hp_config.get("ent_coef", {})
        ent_coef = trial.suggest_float(
            "ent_coef", 
            ent_coef_cfg.get("min", 0.00001), 
            ent_coef_cfg.get("max", 0.5), 
            log=ent_coef_cfg.get("log", True)
        )
        
        batch_size_cfg = hp_config.get("batch_size", {})
        batch_size = trial.suggest_int(
            "batch_size", 
            batch_size_cfg.get("min", 8), 
            batch_size_cfg.get("max", 128), 
            log=batch_size_cfg.get("log", True)
        )
        
        gamma_cfg = hp_config.get("gamma", {})
        gamma = trial.suggest_float(
            "gamma", 
            gamma_cfg.get("min", 0.9), 
            gamma_cfg.get("max", 0.9999)
        )
        
        gae_lambda_cfg = hp_config.get("gae_lambda", {})
        gae_lambda = trial.suggest_float(
            "gae_lambda", 
            gae_lambda_cfg.get("min", 0.9), 
            gae_lambda_cfg.get("max", 0.999)
        )
        
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
            gae_lambda=gae_lambda
        )
        
        # Train the model
        total_timesteps = config["training"].get("total_timesteps", 10000)
        model.learn(total_timesteps=total_timesteps)
        
        # Create temporary directory for this trial
        os.makedirs('models/daily/trials', exist_ok=True)
        
        # Evaluate on validation data based on the chosen metric
        if eval_metric == "prediction_accuracy":
            # Use prediction accuracy evaluation
            results = evaluate_agent_prediction_accuracy(model, validation_data, verbose=0, deterministic=True)
        elif eval_metric == "hit_rate":
            # Save the model temporarily for evaluation with risk management
            temp_model_path = f"models/daily/trials/temp_model_trial_{trial.number}"
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
        else:  # Default to return
            # Save the model temporarily for evaluation
            temp_model_path = f"models/daily/trials/temp_model_trial_{trial.number}"
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
    
    # Save visualization
    try:
        os.makedirs('models/plots/tuning', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create optimization visualization plots if plotly is available
        try:
            import plotly
            
            # Create optimization visualization plots
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.write_image(f'models/plots/tuning/optimization_history_{timestamp}.png')
            
            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.write_image(f'models/plots/tuning/param_importances_{timestamp}.png')
            
            logger.info(f"Saved optimization visualizations to models/plots/tuning/")
        except ImportError:
            logger.warning("Plotly is not installed. Skipping optimization visualization plots.")
    except Exception as e:
        logger.warning(f"Could not save optimization visualizations: {e}")
    
    return {
        "best_params": best_params,
        "best_value": best_value,
        "study": study
    }

def train_model(train_data: pd.DataFrame, validation_data: pd.DataFrame,
               initial_timesteps: int = None,
               additional_timesteps: int = None,
               max_iterations: int = None,
               n_stagnant_loops: int = None,
               improvement_threshold: float = None,
               eval_metric: str = None,
               hit_rate_min_trades: int = None,
               min_predictions: int = None,
               seed: int = 42) -> Tuple[PPO, List[Dict], str]:
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
        eval_metric: Metric to use for validation (return, hit_rate, or prediction_accuracy)
        hit_rate_min_trades: Minimum number of trades for hit rate to be meaningful
        min_predictions: Minimum number of predictions for prediction accuracy to be meaningful
        seed: Random seed for reproducibility
        
    Returns:
        Tuple[PPO, List[Dict], str]: Best trained model based on validation performance and training statistics and model path
    """
    # Set all random seeds for reproducibility
    set_all_seeds(seed)
    
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
    if eval_metric is None:
        eval_metric = config["training"].get("evaluation", {}).get("metric", "return")
    if hit_rate_min_trades is None:
        hit_rate_min_trades = config["training"].get("evaluation", {}).get("hit_rate_min_trades", 3)
    if min_predictions is None:
        min_predictions = config["training"].get("evaluation", {}).get("min_predictions", 10)
    
    # Log evaluation metric
    logger.info(f"Using {eval_metric} as evaluation metric")
    if eval_metric == "hit_rate":
        logger.info(f"Minimum trades for hit rate: {hit_rate_min_trades}")
    elif eval_metric == "prediction_accuracy":
        logger.info(f"Minimum predictions for accuracy: {min_predictions}")
    
    # Create model folder if it doesn't exist
    os.makedirs('models/daily', exist_ok=True)
    
    # Get risk management configuration
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
    
    # Initialize training environment
    train_env = TradingEnv(
        train_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=0.0,
        position_size=config["environment"].get("position_size", 1)
    )
    
    # Set the environment seed for reproducibility
    train_env.reset(seed=seed)
    
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
        seed=seed  # Add seed for reproducibility
    )
    
    # Initialize training statistics list
    training_stats = []
    
    # Initial training
    logger.info(f"Starting initial training for {initial_timesteps} timesteps")
    model.learn(total_timesteps=initial_timesteps)
    
    # Save initial model for validation
    initial_model_path = "models/daily/initial_model"
    model.save(initial_model_path)
    
    # Evaluate initial model on validation data based on the chosen metric
    if eval_metric == "prediction_accuracy":
        # Load the model for evaluation
        loaded_model = PPO.load(initial_model_path)
        results = evaluate_agent_prediction_accuracy(loaded_model, validation_data, verbose=0, deterministic=True)
        best_metric_value = results["prediction_accuracy"]
        best_metric_name = "prediction_accuracy"
    elif eval_metric == "hit_rate":
        # Evaluate with risk management for hit rate
        results = trade_with_risk_management(
            model_path=initial_model_path,
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
        # Calculate hit rate from trade results
        results = calculate_hit_rate_from_trade_results(results)
        best_metric_value = results["hit_rate"]
        best_metric_name = "hit_rate"
    else:  # Default to return
        # Evaluate using return
        results = trade_with_risk_management(
            model_path=initial_model_path,
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
        best_metric_value = results["total_return_pct"]
        best_metric_name = "return_pct"
    
    best_model = model
    best_model_path = initial_model_path
    
    # Record initial training statistics
    initial_stats = {
        "iteration": 0,
        "timesteps": initial_timesteps,
        "return_pct": results.get("total_return_pct", 0),
        "portfolio_value": results["final_portfolio_value"],
        "trade_count": results["trade_count"],
        best_metric_name: best_metric_value,
        "is_best": True
    }
    
    # Add prediction accuracy stats if available
    if "prediction_accuracy" in results:
        initial_stats["prediction_accuracy"] = results["prediction_accuracy"]
        initial_stats["correct_predictions"] = results["correct_predictions"]
        initial_stats["total_predictions"] = results["total_predictions"]
    
    # Add hit rate stats if available
    if "hit_rate" in results:
        initial_stats["hit_rate"] = results["hit_rate"]
        initial_stats["profitable_trades"] = results.get("profitable_trades", 0)
    
    training_stats.append(initial_stats)
    
    logger.info(f"Initial validation {best_metric_name}: {best_metric_value:.2f}%")
    
    # Counter for consecutive iterations without significant improvement
    stagnant_counter = 0
    
    # Continue training until max_iterations or n_stagnant_loops consecutive iterations without improvement
    for iteration in range(1, max_iterations + 1):
        # Train for additional timesteps
        model.learn(total_timesteps=additional_timesteps)
        
        # Save current iteration model
        current_model_path = f"models/daily/model_iteration_{iteration}"
        model.save(current_model_path)
        
        # Evaluate on validation data based on the chosen metric
        if eval_metric == "prediction_accuracy":
            # Load the model for evaluation
            loaded_model = PPO.load(current_model_path)
            results = evaluate_agent_prediction_accuracy(loaded_model, validation_data, verbose=0, deterministic=True)
            current_metric_value = results["prediction_accuracy"]
        elif eval_metric == "hit_rate":
            # Evaluate with risk management for hit rate
            results = trade_with_risk_management(
                model_path=current_model_path,
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
            # Calculate hit rate from trade results
            results = calculate_hit_rate_from_trade_results(results)
            current_metric_value = results["hit_rate"]
        else:  # Default to return
            # Evaluate using return
            results = trade_with_risk_management(
                model_path=current_model_path,
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
            current_metric_value = results["total_return_pct"]
        
        # Calculate improvement
        improvement = current_metric_value - best_metric_value
        logger.info(f"Iteration {iteration} - Validation {best_metric_name}: {current_metric_value:.2f}%, Improvement: {improvement:.2f}%")
        
        # Record training statistics
        is_best = current_metric_value > best_metric_value + improvement_threshold
        iteration_stats = {
            "iteration": iteration,
            "timesteps": additional_timesteps,
            "return_pct": results.get("total_return_pct", 0),
            "portfolio_value": results["final_portfolio_value"],
            "trade_count": results["trade_count"],
            best_metric_name: current_metric_value,
            "improvement": improvement,
            "is_best": is_best
        }
        
        # Add prediction accuracy stats if available
        if "prediction_accuracy" in results:
            iteration_stats["prediction_accuracy"] = results["prediction_accuracy"]
            iteration_stats["correct_predictions"] = results["correct_predictions"]
            iteration_stats["total_predictions"] = results["total_predictions"]
        
        # Add hit rate stats if available
        if "hit_rate" in results:
            iteration_stats["hit_rate"] = results["hit_rate"]
            iteration_stats["profitable_trades"] = results.get("profitable_trades", 0)
        
        training_stats.append(iteration_stats)
        
        # Check if this is the best model so far
        if is_best:
            best_metric_value = current_metric_value
            best_model = model
            best_model_path = current_model_path
            logger.info(f"New best model found! Validation {best_metric_name}: {best_metric_value:.2f}%")
            
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
    
    logger.info(f"Training completed. Best validation {best_metric_name}: {best_metric_value:.2f}%")
    
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

def execute_test_trade(model: PPO, model_path: str, test_data: pd.DataFrame, seed: int = 42) -> Dict:
    """
    Execute a test trade on the test data using the trained model with risk management.
    
    Args:
        model: Trained PPO model (kept for compatibility)
        model_path: Path to the saved model file
        test_data: Test market data
        seed: Random seed for reproducibility
        
    Returns:
        Dict: Trade results
    """
    # Set all random seeds for reproducibility
    set_all_seeds(seed)
    
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
    is_best = [stat["is_best"] for stat in training_stats]
    
    # Determine which metric is being used for validation
    available_metrics = ["return_pct", "hit_rate", "prediction_accuracy"]
    metric_name = "return_pct"  # Default
    
    for metric in available_metrics:
        if metric in training_stats[0]:
            if "is_best" in training_stats[0] and metric != "return_pct":
                # If we have both is_best flag and a non-default metric, this is likely the evaluation metric
                metric_name = metric
                break
    
    # Create label based on metric name
    if metric_name == "return_pct":
        metric_label = "Return (%)"
    elif metric_name == "hit_rate":
        metric_label = "Hit Rate (%)"
    elif metric_name == "prediction_accuracy":
        metric_label = "Prediction Accuracy (%)"
    else:
        metric_label = metric_name.replace("_", " ").title()
    
    # Extract metric values
    metric_values = [stat.get(metric_name, 0) for stat in training_stats]
    
    # Create main plot figure
    plt.figure(figsize=(10, 6))
    
    # Plot metric values
    plt.plot(iterations, metric_values, marker='o', linestyle='-', color='blue', label=metric_label)
    
    # Highlight best models
    best_iterations = [iterations[i] for i in range(len(iterations)) if is_best[i]]
    best_values = [metric_values[i] for i in range(len(metric_values)) if is_best[i]]
    plt.scatter(best_iterations, best_values, color='green', s=100, marker='*', label='Best Model')
    
    plt.xlabel('Training Iteration')
    plt.ylabel(metric_label)
    plt.title(f'Daily Training Progress - {metric_label}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    os.makedirs('models/plots', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.tight_layout()
    plt.savefig(f'models/plots/daily_training_progress_{timestamp}.png')
    plt.close()
    
    logger.info(f"Training progress plot saved to models/plots/daily_training_progress_{timestamp}.png")
    
    # Create additional plots based on the metric
    if metric_name == "hit_rate":
        plt.figure(figsize=(10, 6))
        trade_counts = [stat.get("trade_count", 0) for stat in training_stats]
        profitable_trades = [stat.get("profitable_trades", 0) for stat in training_stats]
        
        plt.bar(iterations, trade_counts, color='blue', alpha=0.7, label='Total Trades')
        plt.bar(iterations, profitable_trades, color='green', alpha=0.7, label='Profitable Trades')
        
        plt.xlabel('Training Iteration')
        plt.ylabel('Number of Trades')
        plt.title('Trade Performance by Iteration')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'models/plots/daily_trade_performance_{timestamp}.png')
        plt.close()
        
        logger.info(f"Trade performance plot saved to models/plots/daily_trade_performance_{timestamp}.png")
    
    elif metric_name == "prediction_accuracy":
        # Check if we have prediction stats in the training stats
        if "total_predictions" in training_stats[0]:
            plt.figure(figsize=(10, 6))
            total_predictions = [stat.get("total_predictions", 0) for stat in training_stats]
            correct_predictions = [stat.get("correct_predictions", 0) for stat in training_stats]
            
            plt.bar(iterations, total_predictions, color='blue', alpha=0.7, label='Total Predictions')
            plt.bar(iterations, correct_predictions, color='green', alpha=0.7, label='Correct Predictions')
            
            plt.xlabel('Training Iteration')
            plt.ylabel('Number of Predictions')
            plt.title('Prediction Performance by Iteration')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'models/plots/daily_prediction_performance_{timestamp}.png')
            plt.close()
            
            logger.info(f"Prediction performance plot saved to models/plots/daily_prediction_performance_{timestamp}.png")

def set_all_seeds(seed=42):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: The seed value to use
    """
    import numpy as np
    import torch
    import random
    import os
    
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(f"All random seeds set to {seed} for reproducibility")

def main():
    """
    Main function to run the daily trading process.
    """
    # Set random seeds for reproducibility (using our function)
    set_all_seeds(42)
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/daily', exist_ok=True)
    os.makedirs('models/logs', exist_ok=True)
    os.makedirs('models/plots', exist_ok=True)
    os.makedirs('models/plots/tuning', exist_ok=True)
    
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
    
    # Get training parameters
    training_config = config.get("training", {})
    initial_timesteps = training_config.get("total_timesteps", 10000)
    additional_timesteps = training_config.get("additional_timesteps", 5000)
    max_iterations = training_config.get("max_iterations", 10)
    n_stagnant_loops = training_config.get("n_stagnant_loops", 3)
    improvement_threshold = training_config.get("improvement_threshold", 0.1)
    
    # Get evaluation metric configuration
    eval_metric = training_config.get("evaluation", {}).get("metric", "return")
    hit_rate_min_trades = training_config.get("evaluation", {}).get("hit_rate_min_trades", 3)
    min_predictions = training_config.get("evaluation", {}).get("min_predictions", 10)
    
    # Get hyperparameter tuning configuration
    hyperparameter_config = config.get("hyperparameter_tuning", {})
    run_hyperparameter_tuning = hyperparameter_config.get("enabled", False)
    tuning_trials = hyperparameter_config.get("n_trials", 30)
    
    # Run hyperparameter tuning if enabled
    model_hyperparameters = {}
    if run_hyperparameter_tuning:
        logger.info(f"Hyperparameter tuning enabled with {tuning_trials} trials")
        logger.info(f"Evaluation metric for tuning: {eval_metric}")
        
        # Run hyperparameter tuning
        tuning_results = hyperparameter_tuning(
            train_data=train_data,
            validation_data=validation_data,
            n_trials=tuning_trials,
            eval_metric=eval_metric,
            hit_rate_min_trades=hit_rate_min_trades,
            min_predictions=min_predictions
        )
        
        # Get best parameters from tuning
        model_hyperparameters = tuning_results["best_params"]
        logger.info(f"Using tuned hyperparameters: {model_hyperparameters}")
    else:
        logger.info("Hyperparameter tuning disabled. Using default parameters from config.")
    
    # Train model on the window with iterative validation
    logger.info(f"Starting iterative training with initial {initial_timesteps} timesteps and max {max_iterations} iterations")
    
    # Create training environment
    train_env = TradingEnv(
        train_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=0.0,
        position_size=config["environment"].get("position_size", 1)
    )
    
    # Initialize model with tuned hyperparameters if available
    if model_hyperparameters:
        model = PPO(
            "MlpPolicy", 
            train_env, 
            verbose=0,
            ent_coef=model_hyperparameters.get("ent_coef", config["model"].get("ent_coef", 0.01)),
            learning_rate=model_hyperparameters.get("learning_rate", config["model"].get("learning_rate", 0.0003)),
            n_steps=model_hyperparameters.get("n_steps", config["model"].get("n_steps", 2048)),
            batch_size=model_hyperparameters.get("batch_size", config["model"].get("batch_size", 64)),
            gamma=model_hyperparameters.get("gamma", 0.99),
            gae_lambda=model_hyperparameters.get("gae_lambda", 0.95),
            seed=42
        )
        
        # Train and evaluate model with tuned hyperparameters
        model, training_stats, best_model_path = train_model(
            train_data,
            validation_data,
            initial_timesteps=initial_timesteps,
            additional_timesteps=additional_timesteps,
            max_iterations=max_iterations,
            n_stagnant_loops=n_stagnant_loops,
            improvement_threshold=improvement_threshold,
            eval_metric=eval_metric,
            hit_rate_min_trades=hit_rate_min_trades,
            min_predictions=min_predictions,
            seed=42
        )
    else:
        # Train and evaluate model with default parameters
        model, training_stats, best_model_path = train_model(
            train_data,
            validation_data,
            initial_timesteps=initial_timesteps,
            additional_timesteps=additional_timesteps,
            max_iterations=max_iterations,
            n_stagnant_loops=n_stagnant_loops,
            improvement_threshold=improvement_threshold,
            eval_metric=eval_metric,
            hit_rate_min_trades=hit_rate_min_trades,
            min_predictions=min_predictions,
            seed=42
        )
    
    # Plot training progress
    plot_training_progress(training_stats)
    
    # Execute test trade using the best model
    logger.info("Running trade on test data using the best model")
    results = execute_test_trade(model, best_model_path, test_data, seed=42)
    
    # Plot results
    plots_dir = "models/plots"
    os.makedirs(plots_dir, exist_ok=True)
    plot_results(results, plots_dir=plots_dir)
    
    # Save trade history
    trade_history_file = f"models/logs/daily_trade_history_{timestamp}.csv"
    save_trade_history(results["trade_history"], trade_history_file)
    logger.info(f"Trade history saved to {trade_history_file}")
    
    # Print summary of results
    logger.info(f"\n{'='*80}\nDaily Trade Results\n{'='*80}")
    logger.info(f"Final Portfolio Value: ${results['final_portfolio_value']:.2f}")
    logger.info(f"Total Return: {results['total_return_pct']:.2f}%")
    logger.info(f"Total Trades: {results['trade_count']}")
    logger.info(f"Final Position: {results['final_position']}")
    
    # Add additional metrics based on evaluation metric
    if eval_metric == "prediction_accuracy" and "prediction_accuracy" in results:
        logger.info(f"Prediction Accuracy: {results['prediction_accuracy']:.2f}% ({results.get('correct_predictions', 0)}/{results.get('total_predictions', 0)})")
    elif eval_metric == "hit_rate" and "hit_rate" in results:
        logger.info(f"Hit Rate: {results['hit_rate']:.2f}% ({results.get('profitable_trades', 0)}/{results['trade_count']})")
    
    # Print a warning if no trades were made
    if results['trade_count'] == 0:
        logger.warning("No trades were executed in test period. The model may need more training data or different parameters.")
    
    logger.info(f"{'='*80}\nDaily trading session completed at {datetime.now().strftime('%Y%m%d_%H%M%S')}\n{'='*80}")

if __name__ == "__main__":
    main() 