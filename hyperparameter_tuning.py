import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import concurrent.futures
import multiprocessing
import optuna
from stable_baselines3 import PPO

from environment import TradingEnv
from config import config
from trade import trade_with_risk_management
from walk_forward import calculate_hit_rate_from_trade_results, evaluate_agent_prediction_accuracy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def objective_func(
    trial: optuna.Trial,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    eval_metric: str = "return",
    hit_rate_min_trades: int = 5,
    min_predictions: int = 10
) -> float:
    """
    Objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial
        train_data: Training data
        validation_data: Validation data
        eval_metric: Evaluation metric (return, hit_rate, prediction_accuracy)
        hit_rate_min_trades: Minimum number of trades required for hit rate
        min_predictions: Minimum number of predictions for prediction accuracy
        
    Returns:
        float: Value to maximize
    """
    # Get hyperparameter ranges from config
    hp_config = config.get("hyperparameter_tuning", {}).get("parameters", {})
    
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
    
    # Use fixed values for gamma and gae_lambda from config
    gamma = hp_config.get("gamma", 0.995)  # Use fixed value instead of suggesting
    gae_lambda = hp_config.get("gae_lambda", 0.95)  # Use fixed value instead of suggesting
    
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
    os.makedirs('models/trials', exist_ok=True)
    
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
    
    # Evaluate on validation data based on the chosen metric
    if eval_metric == "prediction_accuracy":
        # Use prediction accuracy evaluation
        results = evaluate_agent_prediction_accuracy(model, validation_data, verbose=0, deterministic=True)
    elif eval_metric == "hit_rate":
        # Save the model temporarily for evaluation with risk management
        temp_model_path = f"models/trials/temp_model_trial_{trial.number}"
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
        temp_model_path = f"models/trials/temp_model_trial_{trial.number}"
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

def parallel_hyperparameter_tuning(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    n_trials: int = 30,
    eval_metric: str = "return",
    hit_rate_min_trades: int = 5,
    min_predictions: int = 10,
    n_jobs: int = -1
) -> Dict:
    """
    Run hyperparameter tuning in parallel using Optuna.
    
    Args:
        train_data: Training data DataFrame
        validation_data: Validation data DataFrame
        n_trials: Number of trials for Optuna optimization
        eval_metric: Evaluation metric to optimize for
        hit_rate_min_trades: Minimum number of trades required for hit rate to be meaningful
        min_predictions: Minimum number of predictions required for prediction accuracy to be meaningful
        n_jobs: Number of parallel jobs (-1 means use all available cores)
        
    Returns:
        Dict: Best hyperparameters and optimization results
    """
    logger.info(f"Starting parallel hyperparameter tuning with {n_trials} trials")
    logger.info(f"Evaluation metric: {eval_metric}")
    
    # If n_jobs is -1, use all available cores
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    logger.info(f"Using {n_jobs} parallel workers for hyperparameter tuning")
    
    # Create Optuna study
    study_name = f"parallel_hyperparam_tuning_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    sampler = optuna.samplers.TPESampler(seed=config["seed"])
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name)
    
    # Create a partial function with fixed arguments
    from functools import partial
    objective = partial(
        objective_func,
        train_data=train_data,
        validation_data=validation_data,
        eval_metric=eval_metric,
        hit_rate_min_trades=hit_rate_min_trades,
        min_predictions=min_predictions
    )
    
    # Run optimization with parallelism
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    
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
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
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