import logging
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from decimal import Decimal
import os
import json

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from environment import TradingEnv
from get_data import get_data
from config import config
import money  # Import the new money module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEE_RATE = 0.0  # No trading fees

def train_agent(train_data, total_timesteps: int):
    """
    Train a PPO model based on training data.

    Args:
        train_data (pd.DataFrame): Training dataset.
        total_timesteps (int): Number of training timesteps.

    Returns:
        model: Trained PPO model.
    """
    env = TradingEnv(
        train_data,
        initial_balance=config["environment"]["initial_balance"],
        position_size=config["environment"].get("position_size", 1)
    )
    check_env(env, skip_render_check=True)
    model = PPO("MlpPolicy", env, verbose=1)
    logger.info("Starting training for %d timesteps", total_timesteps)
    model.learn(total_timesteps=total_timesteps)
    logger.info("Training completed")
    return model

def train_agent_iteratively(train_data, validation_data, initial_timesteps: int, max_iterations: int = 20, 
                           n_stagnant_loops: int = 3, improvement_threshold: float = 0.1, additional_timesteps: int = 10000,
                           evaluation_metric: str = "return", model_params: dict = None):
    """
    Train a PPO model iteratively based on validation performance.
    
    Args:
        train_data (pd.DataFrame): Training dataset.
        validation_data (pd.DataFrame): Validation dataset for model selection.
        initial_timesteps (int): Initial number of training timesteps.
        max_iterations (int): Maximum number of training iterations.
        n_stagnant_loops (int): Number of consecutive iterations without improvement before stopping.
        improvement_threshold (float): Minimum percentage improvement considered significant.
        additional_timesteps (int): Number of additional timesteps for each iteration.
        evaluation_metric (str): Metric to use for evaluation ("return", "hit_rate", or "prediction_accuracy").
        model_params (dict, optional): Model hyperparameters to use for training.
        
    Returns:
        tuple: (best_model, best_results, all_results)
    """
    # Initialize training environment
    train_env = TradingEnv(
        train_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=0.0,  # No transaction costs
        position_size=config["environment"].get("position_size", 1)
    )
    
    check_env(train_env, skip_render_check=True)
    
    # DEBUGGING: Print the observation space and actual features
    logger.info(f"Observation space shape: {train_env.observation_space.shape}")
    
    # Check the observation vector from the reset
    obs, _ = train_env.reset()
    logger.info(f"Actual observation vector shape: {obs.shape}")
    logger.info(f"First few values of observation: {obs[:5]}")
    
    # Check technical indicators in the environment
    logger.info(f"Technical indicators used in environment: {train_env.technical_indicators}")
    logger.info(f"Total observation components: close_norm + {len(train_env.technical_indicators)} indicators + position = {1 + len(train_env.technical_indicators) + 1}")
    
    # Get verbosity level from config
    verbose_level = config["training"].get("verbose", 1)
    
    # Use provided model parameters or get defaults from config
    if model_params is None:
        model_params = {
            "ent_coef": config["model"].get("ent_coef", 0.01),
            "learning_rate": config["model"].get("learning_rate", 0.0003),
            "n_steps": config["model"].get("n_steps", 2048),
            "batch_size": config["model"].get("batch_size", 64),
            "gamma": config["model"].get("gamma", 0.99),
            "gae_lambda": config["model"].get("gae_lambda", 0.95),
        }
    
    # Initialize the PPO model with the specified parameters
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=verbose_level,
        ent_coef=model_params.get("ent_coef", 0.01),
        learning_rate=model_params.get("learning_rate", 0.0003),
        n_steps=model_params.get("n_steps", 2048),
        batch_size=model_params.get("batch_size", 64),
        gamma=model_params.get("gamma", 0.99),
        gae_lambda=model_params.get("gae_lambda", 0.95),
    )
    
    # Initial training
    logger.info(f"Starting initial training for {initial_timesteps} timesteps")
    model.learn(total_timesteps=initial_timesteps)
    
    # Evaluate initial model on validation data using the specified metric
    if verbose_level > 0:
        logger.info(f"Evaluating model on validation data with {evaluation_metric} metric")
    
    # Use the appropriate evaluation function based on the metric
    if evaluation_metric == "prediction_accuracy":
        # Need to import from walk_forward module
        from walk_forward import evaluate_agent_prediction_accuracy
        results = evaluate_agent_prediction_accuracy(model, validation_data, verbose=verbose_level, deterministic=True)
        best_metric_value = results["prediction_accuracy"]
        metric_name = "prediction_accuracy"
    elif evaluation_metric == "hit_rate":
        results = evaluate_agent(model, validation_data, verbose=verbose_level, deterministic=True)
        best_metric_value = results["hit_rate"]
        metric_name = "hit_rate"
    else:  # Default to "return"
        results = evaluate_agent(model, validation_data, verbose=verbose_level, deterministic=True)
        best_metric_value = results["total_return_pct"]
        metric_name = "return"
    
    # Store best model and results
    best_model = model
    best_results = results
    
    # Save the initial model as the best model so far
    best_model.save("best_model")
    
    # Log evaluation results based on metric
    logger.info(f"Initial training completed. Validation {metric_name.replace('_', ' ').title()}: {best_metric_value:.2f}%, " 
                f"Validation Portfolio: ${results['final_portfolio_value']:.2f}")
    
    # Store all results for comparison
    all_results = [results]
    
    # Add metric information to results
    results["metric_used"] = metric_name
    
    # Counter for consecutive iterations without significant improvement
    stagnant_counter = 0
    
    # Continue training until max_iterations or n_stagnant_loops consecutive iterations without improvement
    for iteration in range(1, max_iterations + 1):
        # Train for additional timesteps
        if verbose_level > 0:
            logger.info(f"Starting iteration {iteration} training for {additional_timesteps} timesteps")
        model.learn(total_timesteps=additional_timesteps)
        
        # Evaluate the model on validation data using the specified metric
        if evaluation_metric == "prediction_accuracy":
            from walk_forward import evaluate_agent_prediction_accuracy
            results = evaluate_agent_prediction_accuracy(model, validation_data, verbose=verbose_level, deterministic=True)
            current_metric_value = results["prediction_accuracy"]
        elif evaluation_metric == "hit_rate":
            results = evaluate_agent(model, validation_data, verbose=verbose_level, deterministic=True)
            current_metric_value = results["hit_rate"]
        else:  # Default to "return"
            results = evaluate_agent(model, validation_data, verbose=verbose_level, deterministic=True)
            current_metric_value = results["total_return_pct"]
        
        # Add metric information to results
        results["metric_used"] = metric_name
        results["is_best"] = False  # Will update this if it becomes the best model
        
        all_results.append(results)
        
        # Calculate improvement
        improvement = current_metric_value - best_metric_value
        logger.info(f"Iteration {iteration} - Validation {metric_name.replace('_', ' ').title()}: {current_metric_value:.2f}%, " 
                   f"Validation Portfolio: ${results['final_portfolio_value']:.2f}, "
                   f"Improvement: {improvement:.2f}%")
        
        # Check if this is the best model so far based on validation performance
        if current_metric_value > best_metric_value + improvement_threshold:
            best_metric_value = current_metric_value
            best_model = model
            best_results = results
            # Mark this result as the new best
            results["is_best"] = True
            
            logger.info(f"New best model found! Validation {metric_name.replace('_', ' ').title()}: {best_metric_value:.2f}%, " 
                       f"Validation Portfolio: ${best_results['final_portfolio_value']:.2f}")
            
            # Save the best model
            best_model.save("best_model")
            
            # Reset stagnant counter since we found improvement
            stagnant_counter = 0
        else:
            # Increment stagnant counter if no significant improvement
            stagnant_counter += 1
            if verbose_level > 0:
                logger.info(f"No significant improvement. Stagnant iterations: {stagnant_counter}/{n_stagnant_loops}")
        
        # Stop if we've had n_stagnant_loops consecutive iterations without improvement
        if stagnant_counter >= n_stagnant_loops:
            logger.info(f"Stopping training after {n_stagnant_loops} consecutive iterations without significant improvement")
            break
    
    logger.info(f"Iterative training completed. Best validation {metric_name.replace('_', ' ').title()}: {best_metric_value:.2f}%, " 
               f"Validation Portfolio: ${best_results['final_portfolio_value']:.2f}")
    return best_model, best_results, all_results

def evaluate_agent(model, test_data, verbose=0, deterministic=True, render=False):
    """
    Evaluate a trained agent on test data.
    
    Args:
        model: Trained model to evaluate
        test_data: Test data DataFrame
        verbose: Verbosity level (0=silent, 1=info)
        deterministic: Whether to make deterministic predictions
        render: Whether to render the environment during evaluation
        
    Returns:
        Dict: Results including portfolio value, return, etc.
    """
    # Check for the presence of close_norm column
    if 'close_norm' not in test_data.columns:
        # Determine which price column names are used in this dataframe
        if 'Close' in test_data.columns:
            close_col = 'Close'
        else:
            close_col = 'close'
        
        # Calculate close_norm if missing
        test_data['close_norm'] = test_data[close_col].pct_change().fillna(0)
    
    # Create evaluation environment
    env = TradingEnv(
        test_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 0.0),
        position_size=config["environment"].get("position_size", 1)
    )
    
    # Reset environment and store initial net worth
    obs, _ = env.reset()
    initial_net_worth = env.net_worth  # STORE INITIAL NET WORTH HERE
    
    # Track portfolio values and actions over time
    portfolio_history = [float(env.net_worth)]
    action_history = []
    
    # Start evaluation
    done = False
    total_reward = 0
    step_count = 0
    trade_count = 0
    
    # Track trading positions
    current_position = 0  # 0 = no position, 1 = long, -1 = short
    
    # Track entry points for trades
    entry_price = 0
    entry_step = -1
    trade_history = []
    last_action = None  # Track the last action to record action changes
    
    # Determine which case is used for price columns
    if 'Close' in test_data.columns:
        close_col = 'Close'
    elif 'CLOSE' in test_data.columns:
        close_col = 'CLOSE'
    else:
        close_col = 'close'
    
    # Get initial price for reference
    current_price_initial = round(float(test_data.loc[test_data.index[env.current_step], close_col]), 2)
    
    if verbose > 0:
        print(f"Starting evaluation with initial price: ${current_price_initial}")
    
    # Main evaluation loop
    while not done:
        # Get current step information
        current_step = env.current_step
        
        # Get current price
        if current_step < len(test_data):
            current_price = round(float(test_data.loc[test_data.index[current_step], close_col]), 2)
        else:
            current_price = current_price_initial  # Fallback if index out of bounds
        
        # Get current action
        action, _ = model.predict(obs, deterministic=deterministic)
        action_history.append(action)
        
        # Check if action has changed, which means a potential trade
        if last_action is not None and action != last_action:
            # Record the trade when action changes
            trade_history.append({
                "date": test_data.index[current_step],
                "trade_type": "Buy" if action == 0 else "Sell" if action == 1 else "Hold",
                "price": current_price,
                "portfolio_value": float(money.format_money(env.net_worth, 2)),
                "profitable": True if current_position == 0 else float(env.net_worth) > float(entry_price),
                "position_from": entry_price if entry_price > 0 else 0,
                "position_to": current_price
            })
            
            if action != 2:  # If not a hold action, update entry info
                entry_price = current_price
                entry_step = current_step
        
        # Take action in environment
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        
        # Update last action
        last_action = action
        
        # Update portfolio value
        portfolio_history.append(float(money.format_money(env.net_worth, 2)))
        
        # Update trade history when position changes (original logic)
        if env.position != current_position:
            trade_count += 1
            if current_position == 0:
                # Entry point
                entry_price = current_price
                entry_step = current_step
                current_position = env.position
            else:
                # Exit point
                trade_history.append({
                    "date": test_data.index[current_step],
                    "trade_type": "Sell" if current_position == -1 else "Buy",
                    "price": current_price,
                    "portfolio_value": float(money.format_money(env.net_worth, 2)),
                    "profitable": float(env.net_worth) > float(entry_price),
                    "position_from": entry_price,
                    "position_to": current_price
                })
                current_position = env.position
        
        # Update total reward
        total_reward += reward
        
        # Update step count
        step_count += 1
        
        # Render environment if specified
        if render:
            env.render()
    
    # Calculate return
    final_net_worth = env.net_worth
    return_pct = money.calculate_return_pct(final_net_worth, initial_net_worth)
    
    # Calculate hit rate
    hit_rate = 0
    if trade_count > 0:
        hit_rate = (trade_count / step_count) * 100
    
    # Calculate portfolio value
    final_portfolio_value = float(money.format_money(env.net_worth, 2))
    
    # Calculate total return
    total_return_pct = float(money.format_money(return_pct, 2))
    
    # Calculate trade history
    trade_history_df = pd.DataFrame(trade_history)
    
    # Calculate action distribution
    action_counts = {0: 0, 1: 0, 2: 0}
    for action in action_history:
        action_counts[int(action)] += 1
    
    # Prepare results
    results = {
        "final_portfolio_value": final_portfolio_value,
        "total_return_pct": total_return_pct,
        "trade_count": trade_count,
        "hit_rate": hit_rate,
        "final_position": env.position,
        "dates": [test_data.index[env.current_step]],
        "price_history": [current_price],
        "portfolio_history": portfolio_history,
        "trade_history": trade_history_df.to_dict(orient="records"),
        "buy_dates": [],
        "buy_prices": [],
        "sell_dates": [],
        "sell_prices": [],
        "action_counts": action_counts
    }
    
    # Add additional information
    results["metric_used"] = "return"
    results["is_best"] = False
    results["profitable_trades"] = trade_count
    results["profitable"] = float(env.net_worth) > float(entry_price)
    results["profitable_pct"] = (float(env.net_worth) - float(entry_price)) / float(entry_price) * 100 if trade_count > 0 else 0
    results["profitable_trade_pct"] = results["profitable_pct"] / trade_count if trade_count > 0 else 0
    results["profitable_trade_return"] = results["profitable_trade_pct"] * 100
    results["profitable_trade_return_pct"] = results["profitable_trade_return"] / 100
    results["profitable_trade_return_str"] = f"{results['profitable_trade_return_pct']:.2f}%"
    
    return results

def plot_results(results):
    """
    Plot BTC price, trade signals, and portfolio value over time using Plotly.
    """
    dates = results["dates"]
    price_history = results["price_history"]
    portfolio_history = results["portfolio_history"]

    # Create subplots with 2 rows in one column and shared X-axis
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.15,  # Increased spacing between subplots
        subplot_titles=(
            "Asset Price and Trade Signals",
            "Portfolio Value"  # Simplified title - we'll add the details in annotations
        )
    )

    # Plot BTC Price line in the first row
    fig.add_trace(
        go.Scatter(x=dates, y=price_history, name="Price", line=dict(color="blue")),
        row=1, col=1
    )
    
    # Plot Buy signals with triangle-up markers
    fig.add_trace(
        go.Scatter(
            x=results["buy_dates"],
            y=results["buy_prices"],
            name="Buy",
            mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=12)
        ),
        row=1, col=1
    )
    
    # Plot Sell signals with triangle-down markers
    fig.add_trace(
        go.Scatter(
            x=results["sell_dates"],
            y=results["sell_prices"],
            name="Sell",
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=12)
        ),
        row=1, col=1
    )
    
    # Plot Portfolio Value in the second row
    fig.add_trace(
        go.Scatter(
            x=dates, 
            y=portfolio_history, 
            name="Portfolio Value", 
            line=dict(color="purple"),
            showlegend=False  # Remove duplicate legend entry
        ),
        row=2, col=1
    )
    
    # Add portfolio performance annotation
    fig.add_annotation(
        text=f"Initial: ${config['environment']['initial_balance']:,.2f}<br>Final: ${results['final_portfolio_value']:,.2f}<br>Return: {results['total_return_pct']}%",
        xref="paper", yref="paper",
        x=1.0, y=0.4,  # Position annotation on right side of bottom plot
        showarrow=False,
        font=dict(size=12),
        align="left"
    )
    
    # Update layout for titles and axes
    fig.update_layout(
        height=800,
        width=1200,  # Increased width
        showlegend=True,
        xaxis=dict(title="Date", rangeslider=dict(visible=True)),
        yaxis=dict(title="Asset Price ($)", tickprefix="$"),
        xaxis2=dict(title="Date"),
        yaxis2=dict(title="Portfolio Value ($)", tickprefix="$"),
        legend=dict(
            x=0.5,  # Center the legend
            y=1.15,  # Move legend above the plot
            xanchor="center",
            orientation="h"
        ),
        margin=dict(l=60, r=60, t=100, b=50)  # Increased margins
    )
    
    fig.show()

def plot_training_progress(all_results):
    """
    Plot the training progress across iterations.
    
    Args:
        all_results (list): List of evaluation results from each training iteration.
    """
    iterations = list(range(len(all_results)))
    returns = [result["total_return_pct"] for result in all_results]
    trade_counts = [result["trade_count"] for result in all_results]
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=iterations, y=returns, name="Return %", line=dict(color="blue", width=3)),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=iterations, y=trade_counts, name="Trade Count", line=dict(color="red", width=2, dash="dot")),
        secondary_y=True,
    )
    
    # Add figure title
    fig.update_layout(
        title_text="Training Progress by Iteration",
        xaxis=dict(title="Iteration"),
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        height=600,
        width=1000,
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Return %", secondary_y=False)
    fig.update_yaxes(title_text="Trade Count", secondary_y=True)
    
    fig.show()

def save_trade_history(trade_history, filename="trade_history.csv"):
    """
    Export the trade history to a CSV file.

    Args:
        trade_history (list): List of trade event dictionaries.
        filename (str): Output filename (default "trade_history.csv").
    """
    trade_history_df = pd.DataFrame(trade_history)
    trade_history_df.to_csv(filename, index=False)
    logger.info("Trade history saved to %s", filename)

def train_walk_forward_model(train_data, validation_data, initial_timesteps=20000, additional_timesteps=10000, 
                         max_iterations=200, n_stagnant_loops=10, improvement_threshold=0.05, window_folder=None,
                         run_hyperparameter_tuning=False, tuning_trials=30, tuning_folder=None):
    """
    Train a model using walk-forward optimization.
    
    Args:
        train_data (pd.DataFrame): Training dataset.
        validation_data (pd.DataFrame): Validation dataset for model selection.
        initial_timesteps (int): Initial number of training timesteps.
        additional_timesteps (int): Number of additional timesteps for each iteration.
        max_iterations (int): Maximum number of training iterations.
        n_stagnant_loops (int): Number of consecutive iterations without improvement before stopping.
        improvement_threshold (float): Minimum percentage improvement considered significant.
        window_folder (str): Path to save model and results for this window.
        run_hyperparameter_tuning (bool): Whether to run hyperparameter tuning.
        tuning_trials (int): Number of trials for hyperparameter tuning.
        tuning_folder (str): Path to save hyperparameter tuning results.
        
    Returns:
        tuple: (trained_model, training_stats)
    """
    # Log the start of training
    logger.info(f"Starting model training with parameters:")
    logger.info(f"- initial_timesteps: {initial_timesteps}")
    logger.info(f"- additional_timesteps: {additional_timesteps}")
    logger.info(f"- max_iterations: {max_iterations}")
    logger.info(f"- n_stagnant_loops: {n_stagnant_loops}")
    logger.info(f"- improvement_threshold: {improvement_threshold}")
    
    # Get the evaluation metric from config
    evaluation_metric = config.get("training", {}).get("evaluation", {}).get("metric", "return")
    logger.info(f"Using {evaluation_metric} as the evaluation metric for model selection")
    
    model_params = None
    
    # Perform hyperparameter tuning if enabled
    if run_hyperparameter_tuning:
        logger.info(f"Starting hyperparameter tuning with {tuning_trials} trials using {evaluation_metric} metric")
        
        # Import the hyperparameter tuning function
        from walk_forward import hyperparameter_tuning
        
        # Run hyperparameter tuning with specified metric
        best_params = hyperparameter_tuning(
            train_data=train_data,
            validation_data=validation_data,
            n_trials=tuning_trials,
            eval_metric=evaluation_metric
        )
        
        logger.info(f"Hyperparameter tuning completed. Best parameters: {best_params}")
        
        # Use the best parameters for model training
        model_params = best_params
        
        # Save tuning results if a folder is provided
        if tuning_folder:
            os.makedirs(tuning_folder, exist_ok=True)
            with open(os.path.join(tuning_folder, "best_params.json"), "w") as f:
                json.dump(best_params, f, indent=4)
    
    # Log the model parameters being used
    logger.info(f"Training model with parameters: {model_params if model_params else 'default parameters'}")
    
    # Train the model iteratively
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
    
    # Save validation results
    if window_folder:
        os.makedirs(window_folder, exist_ok=True)
        
        # Save validation results in json format
        validation_results_json = {
            "final_portfolio_value": validation_results.get("final_portfolio_value", 0),
            "total_return_pct": validation_results.get("total_return_pct", 0),
            "trade_count": validation_results.get("trade_count", 0),
            "profitable_trades": validation_results.get("profitable_trades", 0),
            "hit_rate": validation_results.get("hit_rate", 0),
            "prediction_accuracy": validation_results.get("prediction_accuracy", 0),
            "correct_predictions": validation_results.get("correct_predictions", 0),
            "total_predictions": validation_results.get("total_predictions", 0),
            "final_position": validation_results.get("final_position", 0),
            "evaluation_metric_used": evaluation_metric
        }
        
        with open(os.path.join(window_folder, "validation_results.json"), "w") as f:
            json.dump(validation_results_json, f, indent=4)
        
        # Save model
        model.save(os.path.join(window_folder, "model"))
        
        # Prepare training stats for all iterations
        training_stats = []
        for i, result in enumerate(all_results):
            entry = {
                "iteration": i,
                "return_pct": result.get("total_return_pct", 0),
                "portfolio_value": result.get("final_portfolio_value", 0),
                "hit_rate": result.get("hit_rate", 0),
                "prediction_accuracy": result.get("prediction_accuracy", 0),
                "is_best": result.get("is_best", False),
                "metric_used": result.get("metric_used", evaluation_metric)
            }
            training_stats.append(entry)
        
        # Save training stats
        with open(os.path.join(window_folder, "training_stats.json"), "w") as f:
            json.dump(training_stats, f, indent=4)
    
    return model, training_stats

def main():
    # Load data using settings from YAML with three-way split and direct Yahoo Finance download
    train_data, validation_data, test_data = get_data(
        symbol=config["data"]["symbol"],
        period=config["data"]["period"],
        interval=config["data"]["interval"],
        train_ratio=config["data"].get("train_ratio", 0.7),
        validation_ratio=config["data"].get("validation_ratio", 0.15),
        use_yfinance=True  # Use Yahoo Finance directly
    )

    # Use the iterative training approach with validation data for model selection
    initial_timesteps = config["training"].get("total_timesteps", 50000)
    max_iterations = config["training"].get("max_iterations", 20)
    improvement_threshold = config["training"].get("improvement_threshold", 0.1)
    additional_timesteps = config["training"].get("additional_timesteps", 10000)
    n_stagnant_loops = config["training"].get("n_stagnant_loops", 3)
    
    logger.info(f"Starting iterative training with initial_timesteps={initial_timesteps}, "
               f"max_iterations={max_iterations}, n_stagnant_loops={n_stagnant_loops}, "
               f"improvement_threshold={improvement_threshold}%, "
               f"additional_timesteps={additional_timesteps}")
    
    # Train using train data and validate using validation data
    best_model, best_validation_results, all_validation_results = train_agent_iteratively(
        train_data, 
        validation_data,  # Use validation data for model selection
        initial_timesteps=initial_timesteps,
        max_iterations=max_iterations,
        n_stagnant_loops=n_stagnant_loops,
        improvement_threshold=improvement_threshold,
        additional_timesteps=additional_timesteps
    )
    
    # Final evaluation on the test data (previously unseen)
    logger.info("Performing final evaluation on test data (previously unseen)")
    test_results = evaluate_agent(best_model, test_data, verbose=config["training"].get("verbose", 1), deterministic=True)
    
    # Log detailed evaluation results for the best model on test data
    logger.info("Final Test Results:")
    logger.info("Final Portfolio Value: $%.2f", test_results["final_portfolio_value"])
    logger.info("Total Return: %.2f%%", test_results["total_return_pct"])
    logger.info("Total Trades Executed: %d", test_results["trade_count"])
    logger.info("Final Position: %d", test_results["final_position"])
    
    # Save trade history to CSV
    save_trade_history(test_results["trade_history"], "best_model_test_trade_history.csv")
    
    # Plot results for the best model on test data
    logger.info("Plotting test evaluation results...")
    plot_results(test_results)
    
    # Plot training progress using validation results
    logger.info("Plotting training progress across iterations...")
    plot_training_progress(all_validation_results)
    
    # Also save validation trade history for comparison
    save_trade_history(best_validation_results["trade_history"], "best_model_validation_trade_history.csv")

if __name__ == "__main__":
    main()

