import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import concurrent.futures
import multiprocessing
import optuna
from stable_baselines3 import PPO
import mlflow

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
    seed_value = config.get('seed', 42)
    
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
        seed=seed_value,
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
    
    # Log trial metrics to MLflow (if MLflow run is active)
    try:
        if mlflow.active_run():
            step = trial.number
            mlflow.log_metric(f"parallel_trial_{eval_metric}", metric_value, step=step)
            mlflow.log_metric("parallel_trial_learning_rate", learning_rate, step=step)
            mlflow.log_metric("parallel_trial_n_steps", n_steps, step=step)
            mlflow.log_metric("parallel_trial_ent_coef", ent_coef, step=step)
            mlflow.log_metric("parallel_trial_batch_size", batch_size, step=step)
            mlflow.log_metric("parallel_trial_total_return_pct", results.get("total_return_pct", 0), step=step)
            mlflow.log_metric("parallel_trial_final_portfolio_value", results.get("final_portfolio_value", 0), step=step)
            mlflow.log_metric("parallel_trial_trade_count", results.get("trade_count", 0), step=step)
            
            if eval_metric == "hit_rate":
                mlflow.log_metric("parallel_trial_hit_rate", results.get("hit_rate", 0), step=step)
                mlflow.log_metric("parallel_trial_profitable_trades", results.get("profitable_trades", 0), step=step)
            elif eval_metric == "prediction_accuracy":
                mlflow.log_metric("parallel_trial_prediction_accuracy", results.get("prediction_accuracy", 0), step=step)
                mlflow.log_metric("parallel_trial_correct_predictions", results.get("correct_predictions", 0), step=step)
                mlflow.log_metric("parallel_trial_total_predictions", results.get("total_predictions", 0), step=step)
    except Exception as e:
        # Don't fail the trial if MLflow logging fails
        logger.warning(f"Failed to log trial {trial.number} to MLflow: {e}")
    
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
    
    # Start MLflow run for parallel hyperparameter tuning (nested if parent run exists)
    is_nested = mlflow.active_run() is not None
    mlflow.start_run(run_name=f"Parallel_Hyperparameter_Tuning_{eval_metric}", nested=is_nested)
    
    # Log parallel hyperparameter tuning configuration
    mlflow.log_param("n_trials", n_trials)
    mlflow.log_param("eval_metric", eval_metric)
    mlflow.log_param("hit_rate_min_trades", hit_rate_min_trades)
    mlflow.log_param("min_predictions", min_predictions)
    mlflow.log_param("train_data_size", len(train_data))
    mlflow.log_param("validation_data_size", len(validation_data))
    mlflow.log_param("is_nested_run", is_nested)
    
    # If n_jobs is -1, use all available cores
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
    
    logger.info(f"Using {n_jobs} parallel workers for hyperparameter tuning")
    
    # Log parallel processing configuration
    mlflow.log_param("n_jobs", n_jobs)
    mlflow.log_param("parallel_processing", True)
    
    # Get risk management parameters from config and log them
    risk_config = config.get("risk_management", {})
    risk_enabled = risk_config.get("enabled", False)
    mlflow.log_param("risk_management_enabled", risk_enabled)
    
    if risk_enabled:
        # Log risk management parameters
        stop_loss_config = risk_config.get("stop_loss", {})
        take_profit_config = risk_config.get("take_profit", {})
        trailing_stop_config = risk_config.get("trailing_stop", {})
        position_sizing_config = risk_config.get("position_sizing", {})
        
        mlflow.log_param("stop_loss_enabled", stop_loss_config.get("enabled", False))
        mlflow.log_param("take_profit_enabled", take_profit_config.get("enabled", False))
        mlflow.log_param("trailing_stop_enabled", trailing_stop_config.get("enabled", False))
        mlflow.log_param("position_sizing_enabled", position_sizing_config.get("enabled", False))
        
        if stop_loss_config.get("enabled", False):
            mlflow.log_param("stop_loss_percentage", stop_loss_config.get("percentage", 1.0))
        if take_profit_config.get("enabled", False):
            mlflow.log_param("take_profit_percentage", take_profit_config.get("percentage", 2.0))
        if trailing_stop_config.get("enabled", False):
            mlflow.log_param("trailing_stop_percentage", trailing_stop_config.get("percentage", 0.5))
        if position_sizing_config.get("enabled", False):
            mlflow.log_param("position_size_multiplier", position_sizing_config.get("size_multiplier", 1.0))
            mlflow.log_param("max_risk_per_trade_percentage", position_sizing_config.get("max_risk_per_trade_percentage", 2.0))
    
    # Create Optuna study
    study_name = f"parallel_hyperparam_tuning_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    sampler = optuna.samplers.TPESampler(seed=config["seed"])
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=study_name)
    
    # Log study configuration
    mlflow.log_param("study_name", study_name)
    mlflow.log_param("sampler_type", "TPESampler")
    mlflow.log_param("optimization_direction", "maximize")
    mlflow.log_param("seed", config["seed"])
    
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
    
    # Log best results to MLflow
    mlflow.log_param("best_learning_rate", best_params.get("learning_rate", 0))
    mlflow.log_param("best_n_steps", best_params.get("n_steps", 0))
    mlflow.log_param("best_ent_coef", best_params.get("ent_coef", 0))
    mlflow.log_param("best_batch_size", best_params.get("batch_size", 0))
    mlflow.log_metric(f"best_parallel_{eval_metric}", best_value)
    
    # Log study statistics
    mlflow.log_metric("parallel_total_trials", len(study.trials))
    mlflow.log_metric("parallel_completed_trials", len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]))
    mlflow.log_metric("parallel_failed_trials", len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]))
    
    # Calculate and log optimization progress metrics
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        trial_values = [t.value for t in completed_trials]
        mlflow.log_metric("parallel_optimization_mean", np.mean(trial_values))
        mlflow.log_metric("parallel_optimization_std", np.std(trial_values))
        mlflow.log_metric("parallel_optimization_min", np.min(trial_values))
        mlflow.log_metric("parallel_optimization_max", np.max(trial_values))
        
        # Log improvement over trials
        best_so_far = []
        current_best = float('-inf')
        for value in trial_values:
            if value > current_best:
                current_best = value
            best_so_far.append(current_best)
        
        # Log final improvement
        if len(best_so_far) > 1:
            total_improvement = best_so_far[-1] - best_so_far[0]
            mlflow.log_metric("parallel_total_improvement", total_improvement)
    
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
            fig1.write_image(f'models/plots/tuning/parallel_optimization_history_{timestamp}.png')
            
            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.write_image(f'models/plots/tuning/parallel_param_importances_{timestamp}.png')
            
            # Log plots as artifacts to MLflow
            try:
                mlflow.log_artifact(f'models/plots/tuning/parallel_optimization_history_{timestamp}.png')
            except Exception as e:
                logger.warning(f"Failed to log parallel optimization history plot to MLflow: {e}")
            
            try:
                mlflow.log_artifact(f'models/plots/tuning/parallel_param_importances_{timestamp}.png')
            except Exception as e:
                logger.warning(f"Failed to log parallel parameter importances plot to MLflow: {e}")
            
            logger.info(f"Saved parallel optimization visualizations to models/plots/tuning/")
        except ImportError:
            logger.warning("Plotly is not installed. Skipping optimization visualization plots.")
    except Exception as e:
        logger.warning(f"Could not save optimization visualizations: {e}")
    
    # End MLflow run
    mlflow.end_run()
    
    return {
        "best_params": best_params,
        "best_value": best_value,
        "study": study
    } 