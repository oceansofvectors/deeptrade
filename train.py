# Standard library imports
import json
import logging
import os
from decimal import Decimal

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback

# RecurrentPPO for LSTM support
try:
    from sb3_contrib import RecurrentPPO
    RECURRENT_PPO_AVAILABLE = True
except ImportError:
    RECURRENT_PPO_AVAILABLE = False

# Local application imports
from config import config
from environment import TradingEnv
from get_data import get_data
from utils.seeding import set_global_seed
from utils.device import get_device
import money

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEE_RATE = 0.0  # No trading fees


class LossTrackingCallback(BaseCallback):
    """
    Callback to track policy loss, value loss, and entropy loss during training.
    Used for loss-based early stopping instead of profit-based model selection.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.n_updates = 0

    def _on_rollout_end(self):
        """Called after each rollout collection, before training update."""
        return True

    def _on_step(self):
        """Called after each training step."""
        return True

    def _on_training_end(self):
        """Called at the end of training - extract final losses from logger."""
        if hasattr(self.model, "logger") and self.model.logger is not None:
            logs = self.model.logger.name_to_value
            # SB3 stores losses with these keys
            if "train/policy_gradient_loss" in logs:
                self.policy_losses.append(logs["train/policy_gradient_loss"])
            if "train/value_loss" in logs:
                self.value_losses.append(logs["train/value_loss"])
            if "train/entropy_loss" in logs:
                self.entropy_losses.append(logs["train/entropy_loss"])
            self.n_updates += 1

    def get_latest_losses(self):
        """Get the most recent loss values."""
        return {
            "policy_loss": self.policy_losses[-1] if self.policy_losses else None,
            "value_loss": self.value_losses[-1] if self.value_losses else None,
            "entropy_loss": self.entropy_losses[-1] if self.entropy_losses else None,
        }

    def get_avg_loss(self):
        """Get average value loss (primary metric for early stopping)."""
        if not self.value_losses:
            return None
        return sum(self.value_losses) / len(self.value_losses)

    def reset(self):
        """Reset for a new training iteration."""
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.n_updates = 0

def train_agent(train_data, total_timesteps: int):
    """
    Train a PPO model based on training data.

    Args:
        train_data (pd.DataFrame): Training dataset.
        total_timesteps (int): Number of training timesteps.

    Returns:
        model: Trained PPO or RecurrentPPO model.
    """
    env = TradingEnv(
        train_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 2.50),
        position_size=config["environment"].get("position_size", 1)
    )
    check_env(env, skip_render_check=True)

    # Get sequence model config
    seq_config = config.get("sequence_model", {})
    use_recurrent = seq_config.get("enabled", False) and RECURRENT_PPO_AVAILABLE

    # Get device configuration (use CPU for recurrent models due to MPS LSTM bugs)
    device_config = seq_config.get("device", "auto")
    device = get_device(device_config, for_recurrent=use_recurrent)
    logger.info(f"Using device: {device}")

    # Check for learning rate decay in config
    use_lr_decay = config["model"].get("use_lr_decay", False)
    if use_lr_decay:
        # Set up learning rate parameters
        initial_lr = config["model"].get("learning_rate", 0.0003)
        final_lr = config["model"].get("final_learning_rate", 1e-5)

        # Create linear learning rate schedule
        # LinearSchedule(start, end, end_fraction) - end_fraction=1.0 means decay over full training
        from stable_baselines3.common.utils import LinearSchedule
        learning_rate = LinearSchedule(initial_lr, final_lr, 1.0)
        logger.debug(f"LR decay: {initial_lr} -> {final_lr}")
    else:
        # Use constant learning rate
        learning_rate = config["model"].get("learning_rate", 0.0003)

    # Check for entropy coefficient decay in config
    # Note: RecurrentPPO doesn't support function-based ent_coef, so use constant for LSTM
    use_ent_decay = config["model"].get("ent_coef_decay", False)
    if use_ent_decay and not use_recurrent:
        initial_ent = config["model"].get("ent_coef", 0.01)
        final_ent = config["model"].get("final_ent_coef", 0.001)
        # LinearSchedule(start, end, end_fraction) - end_fraction=1.0 means decay over full training
        from stable_baselines3.common.utils import LinearSchedule
        ent_coef = LinearSchedule(initial_ent, final_ent, 1.0)
        logger.debug(f"Entropy decay: {initial_ent} -> {final_ent}")
    else:
        ent_coef = config["model"].get("ent_coef", 0.01)

    # Initialize model with configured parameters
    if use_recurrent:
        logger.info("Using RecurrentPPO with LSTM policy for temporal sequence learning")
        shared_lstm = seq_config.get("shared_lstm", False)
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            seed=config.get('seed'),
            device=device,
            policy_kwargs={
                "lstm_hidden_size": seq_config.get("lstm_hidden_size", 256),
                "n_lstm_layers": seq_config.get("n_lstm_layers", 1),
                "shared_lstm": shared_lstm,
                "enable_critic_lstm": not shared_lstm,  # Can't have both shared and separate critic LSTM
            }
        )
    else:
        if seq_config.get("enabled", False) and not RECURRENT_PPO_AVAILABLE:
            logger.warning("RecurrentPPO requested but sb3-contrib not installed. Falling back to standard PPO.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            seed=config.get('seed'),
            device=device,
        )

    logger.info("Starting training for %d timesteps", total_timesteps)
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    logger.info("Training completed")
    return model

def train_agent_iteratively(train_data, validation_data, initial_timesteps: int, max_iterations: int = 20, 
                           n_stagnant_loops: int = 3, improvement_threshold: float = 0.1, additional_timesteps: int = 10000,
                           evaluation_metric: str = "return", model_params: dict = None, window_folder: str = None):
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
        window_folder (str, optional): Folder to save models and results.
        
    Returns:
        tuple: (best_model, best_results, all_results)
    """
    # Get dsr_eta from model_params (if tuned), environment config, or use default
    if model_params and "dsr_eta" in model_params:
        dsr_eta = model_params["dsr_eta"]
    else:
        dsr_eta = config.get("environment", {}).get("dsr_eta", 0.01)

    # Initialize training environment with realistic transaction costs
    train_env = TradingEnv(
        train_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 2.50),
        position_size=config["environment"].get("position_size", 1),
        dsr_eta=dsr_eta
    )
    
    check_env(train_env, skip_render_check=True)

    # Debug observation space info
    obs, _ = train_env.reset()
    logger.debug(f"Observation space: {train_env.observation_space.shape}, indicators: {len(train_env.technical_indicators)}")
    
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
    
    # Get learning rate decay parameters from config or use defaults
    use_lr_decay = config["model"].get("use_lr_decay", False)
    # Total timesteps for decay is initial + additional * max iterations
    total_decay_timesteps = initial_timesteps + (additional_timesteps * max_iterations)

    if use_lr_decay:
        # Set up learning rate parameters
        initial_lr = model_params.get("learning_rate", 0.0003)
        final_lr = config["model"].get("final_learning_rate", 1e-5)

        # Create linear learning rate schedule
        # LinearSchedule(start, end, end_fraction) - end_fraction=1.0 means decay over full training
        from stable_baselines3.common.utils import LinearSchedule
        learning_rate = LinearSchedule(initial_lr, final_lr, 1.0)
    else:
        # Use constant learning rate
        learning_rate = model_params.get("learning_rate", 0.0003)

    # Get sequence model config (need this before entropy coef setup)
    seq_config = config.get("sequence_model", {})
    use_recurrent = seq_config.get("enabled", False) and RECURRENT_PPO_AVAILABLE

    # Get entropy coefficient decay parameters from config or use defaults
    # Note: RecurrentPPO doesn't support function-based ent_coef, so use constant for LSTM
    use_ent_decay = config["model"].get("ent_coef_decay", False)
    if use_ent_decay and not use_recurrent:
        initial_ent = model_params.get("ent_coef", 0.01)
        final_ent = config["model"].get("final_ent_coef", 0.001)
        # LinearSchedule(start, end, end_fraction) - end_fraction=1.0 means decay over full training
        from stable_baselines3.common.utils import LinearSchedule
        ent_coef = LinearSchedule(initial_ent, final_ent, 1.0)
    else:
        ent_coef = model_params.get("ent_coef", 0.01)

    # Get device configuration (use CPU for recurrent models due to MPS LSTM bugs)
    device_config = seq_config.get("device", "auto")
    device = get_device(device_config, for_recurrent=use_recurrent)

    # Initialize the model with the specified parameters
    if use_recurrent:
        logger.debug("Using RecurrentPPO with LSTM")
        # Get LSTM params from model_params (if tuned) or from config
        lstm_hidden_size = model_params.get("lstm_hidden_size", seq_config.get("lstm_hidden_size", 256))
        n_lstm_layers = model_params.get("n_lstm_layers", seq_config.get("n_lstm_layers", 1))
        shared_lstm = seq_config.get("shared_lstm", False)

        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            verbose=verbose_level,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            n_steps=model_params.get("n_steps", 2048),
            batch_size=model_params.get("batch_size", 64),
            gamma=model_params.get("gamma", 0.99),
            seed=config.get('seed'),
            gae_lambda=model_params.get("gae_lambda", 0.95),
            device=device,
            policy_kwargs={
                "lstm_hidden_size": lstm_hidden_size,
                "n_lstm_layers": n_lstm_layers,
                "shared_lstm": shared_lstm,
                "enable_critic_lstm": not shared_lstm,  # Can't have both shared and separate critic LSTM
            }
        )
    else:
        if seq_config.get("enabled", False) and not RECURRENT_PPO_AVAILABLE:
            logger.warning("RecurrentPPO requested but sb3-contrib not installed. Falling back to standard PPO.")
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=verbose_level,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            n_steps=model_params.get("n_steps", 2048),
            batch_size=model_params.get("batch_size", 64),
            gamma=model_params.get("gamma", 0.99),
            seed=config.get('seed'),
            gae_lambda=model_params.get("gae_lambda", 0.95),
            device=device,
        )
    
    # Create loss tracking callback for early stopping
    loss_callback = LossTrackingCallback(verbose=verbose_level)
    loss_history = []
    best_loss = float('inf')
    loss_stagnant_counter = 0

    # Initial training
    logger.info(f"Starting initial training for {initial_timesteps} timesteps")
    model.learn(total_timesteps=initial_timesteps, progress_bar=True, callback=loss_callback)

    # Get initial loss values
    initial_losses = loss_callback.get_latest_losses()
    if initial_losses["value_loss"] is not None:
        loss_history.append(initial_losses)
        best_loss = initial_losses["value_loss"]
        logger.info(f"Iter 0 - Loss: {best_loss:.4f}")

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
    elif evaluation_metric == "calmar":
        results = evaluate_agent(model, validation_data, verbose=verbose_level, deterministic=True)
        best_metric_value = results["calmar_ratio"]
        metric_name = "calmar"
    else:  # Default to "return"
        results = evaluate_agent(model, validation_data, verbose=verbose_level, deterministic=True)
        best_metric_value = results["total_return_pct"]
        metric_name = "return"
    
    # Store best model and results
    best_model = model
    best_results = results
    
    # Save the initial model as the best model so far
    best_model.save(os.path.join(window_folder, "best_model"))
    
    # Log evaluation results based on metric
    logger.info(f"Initial training completed. Validation {metric_name.replace('_', ' ').title()}: {best_metric_value:.2f}%, " 
                f"Validation Portfolio: ${results['final_portfolio_value']:.2f}")
    
    # Store all results for comparison
    all_results = [results]
    
    # Add metric information to results
    results["metric_used"] = metric_name

    # Continue training until max_iterations or loss plateau
    for iteration in range(1, max_iterations + 1):
        # Reset callback for this iteration
        loss_callback.reset()

        # Train for additional timesteps
        model.learn(total_timesteps=additional_timesteps, progress_bar=True, callback=loss_callback)

        # Get loss values for this iteration
        iter_losses = loss_callback.get_latest_losses()
        current_loss = iter_losses["value_loss"] if iter_losses["value_loss"] is not None else float('inf')
        if iter_losses["value_loss"] is not None:
            loss_history.append(iter_losses)

        # Evaluate the model on validation data using the specified metric
        if evaluation_metric == "prediction_accuracy":
            from walk_forward import evaluate_agent_prediction_accuracy
            results = evaluate_agent_prediction_accuracy(model, validation_data, verbose=verbose_level, deterministic=True)
            current_metric_value = results["prediction_accuracy"]
        elif evaluation_metric == "hit_rate":
            results = evaluate_agent(model, validation_data, verbose=verbose_level, deterministic=True)
            current_metric_value = results["hit_rate"]
        elif evaluation_metric == "calmar":
            results = evaluate_agent(model, validation_data, verbose=verbose_level, deterministic=True)
            current_metric_value = results["calmar_ratio"]
        else:  # Default to "return"
            results = evaluate_agent(model, validation_data, verbose=verbose_level, deterministic=True)
            current_metric_value = results["total_return_pct"]

        # Add metric and loss information to results
        results["metric_used"] = metric_name
        results["is_best"] = False  # Will update this if it becomes the best model
        results["loss_info"] = iter_losses

        all_results.append(results)

        # Early stopping based on training loss plateau
        if current_loss < best_loss:
            best_loss = current_loss
            loss_stagnant_counter = 0
        else:
            loss_stagnant_counter += 1

        # Model selection based on VALIDATION metric (higher is better for calmar/return)
        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_model = model
            best_results = results
            results["is_best"] = True

            logger.info(f"Iter {iteration} - New best (val {metric_name}={current_metric_value:.2f}): Train loss={current_loss:.4f}, Portfolio: ${results['final_portfolio_value']:.2f}")

            # Save the best model
            best_model.save(os.path.join(window_folder, "best_model"))
        else:
            logger.info(f"Iter {iteration} - Val {metric_name}={current_metric_value:.2f}, Train loss={current_loss:.4f} [loss stagnant {loss_stagnant_counter}/{n_stagnant_loops}]")

        # Early stopping based on training loss plateau
        if loss_stagnant_counter >= n_stagnant_loops:
            logger.info(f"Early stop: training loss plateau for {n_stagnant_loops} iterations")
            break

    # Add loss history to best_results for saving
    best_results["loss_history"] = loss_history

    logger.info(f"Training complete. Best {metric_name}: {best_metric_value:.2f}, Best train loss: {best_loss:.4f}")
    return best_model, best_results, all_results

def evaluate_agent(model, test_data, verbose=0, deterministic=True, render=False):
    """
    Evaluate a trained agent on test data.

    Args:
        model: Trained model to evaluate (PPO or RecurrentPPO)
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

    # Create evaluation environment with realistic transaction costs
    env = TradingEnv(
        test_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 2.50),
        position_size=config["environment"].get("position_size", 1)
    )

    # Reset environment and store initial net worth
    obs, _ = env.reset()
    initial_net_worth = env.net_worth  # STORE INITIAL NET WORTH HERE

    # Check if model is RecurrentPPO (has LSTM)
    is_recurrent = hasattr(model, 'policy') and hasattr(model.policy, 'lstm')

    # Initialize LSTM states for recurrent models
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

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
        if is_recurrent:
            print("Using RecurrentPPO with LSTM state tracking")

    # Main evaluation loop
    while not done:
        # Get current step information
        current_step = env.current_step

        # Get current price
        if current_step < len(test_data):
            current_price = round(float(test_data.loc[test_data.index[current_step], close_col]), 2)
        else:
            current_price = current_price_initial  # Fallback if index out of bounds

        # Get current action (handle both recurrent and non-recurrent models)
        if is_recurrent:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic
            )
            episode_starts = np.array([done])
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        action_history.append(action)
        
        # Take action in environment
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        # Get portfolio value AFTER the step
        portfolio_value_after = float(money.format_money(env.net_worth, 2))
        portfolio_history.append(portfolio_value_after)

        # Track trade history based on position changes from the environment
        old_position = current_position
        new_position = env.position

        if new_position != old_position:
            trade_count += 1

            # Determine trade type based on position change
            if new_position == 1:
                trade_type = "Long"
            elif new_position == -1:
                trade_type = "Short"
            elif new_position == 0:
                trade_type = "Flat"
            else:
                trade_type = "Unknown"

            # Calculate if this trade was profitable (for exits)
            is_profitable = False
            if old_position != 0 and entry_price > 0:
                # We had a position and are changing it - calculate P&L
                if old_position == 1:  # Was long
                    is_profitable = current_price > entry_price
                elif old_position == -1:  # Was short
                    is_profitable = current_price < entry_price

            # For position_from/position_to:
            # - Entry (Long/Short from Flat): position_from=0, position_to=entry_price
            # - Exit (Flat from Long/Short): position_from=entry_price, position_to=0
            # - Flip (Long to Short or vice versa): position_from=old_entry, position_to=new_entry
            if old_position == 0:
                # New entry from flat
                pos_from = 0.0
                pos_to = current_price
            elif new_position == 0:
                # Exit to flat
                pos_from = float(entry_price) if entry_price > 0 else current_price
                pos_to = 0.0
            else:
                # Flip position (long to short or short to long)
                pos_from = float(entry_price) if entry_price > 0 else current_price
                pos_to = current_price

            trade_history.append({
                "date": test_data.index[current_step],
                "trade_type": trade_type,
                "price": current_price,
                "portfolio_value": portfolio_value_after,
                "profitable": is_profitable,
                "position_from": pos_from,
                "position_to": pos_to,
                "exit_reason": ""
            })

            # Update entry price for new positions
            if new_position != 0:
                entry_price = current_price
                entry_step = current_step
            else:
                # Flat - reset entry price
                entry_price = 0
                entry_step = -1

            current_position = new_position

        # Update last action
        last_action = action
        
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
    
    # Calculate hit rate (percentage of profitable trades)
    hit_rate = 0
    if trade_count > 0:
        profitable_count = sum(1 for t in trade_history if t.get("profitable", False))
        hit_rate = (profitable_count / trade_count) * 100
    
    # Calculate portfolio value
    final_portfolio_value = float(money.format_money(env.net_worth, 2))
    
    # Calculate total return
    total_return_pct = float(money.format_money(return_pct, 2))
    
    # Calculate trade history
    trade_history_df = pd.DataFrame(trade_history)
    
    # Calculate action distribution (0=Long, 1=Short, 2=Hold, 3=Flat)
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for action in action_history:
        action_counts[int(action)] += 1

    # Calculate max drawdown from portfolio history
    portfolio_array = np.array(portfolio_history)
    running_max = np.maximum.accumulate(portfolio_array)
    drawdowns = (portfolio_array - running_max) / running_max * 100  # As percentage
    max_drawdown = float(np.min(drawdowns))  # Most negative value (will be <= 0)

    # Calculate Calmar ratio (return / |max_drawdown|)
    # Higher is better - rewards high returns with low drawdowns
    if max_drawdown == 0:
        # No drawdown - use return as calmar (or a large value if positive return)
        calmar_ratio = total_return_pct if total_return_pct > 0 else 0.0
    else:
        calmar_ratio = total_return_pct / abs(max_drawdown)

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
        "action_counts": action_counts,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio
    }

    # Add additional information
    results["metric_used"] = "return"
    results["is_best"] = False
    results["profitable_trades"] = sum(1 for t in trade_history if t.get("profitable", False))
    results["profitable"] = total_return_pct > 0
    results["profitable_pct"] = total_return_pct
    results["profitable_trade_pct"] = results["profitable_trades"] / trade_count * 100 if trade_count > 0 else 0
    results["profitable_trade_return"] = results["profitable_trade_pct"]
    results["profitable_trade_return_pct"] = results["profitable_trade_pct"]
    results["profitable_trade_return_str"] = f"{results['profitable_trade_pct']:.2f}%"
    
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
                         run_hyperparameter_tuning=False, tuning_trials=30, tuning_folder=None, model_params=None):
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
        model_params (dict): Pre-tuned hyperparameters to use. If provided, skips tuning.
        
    Returns:
        tuple: (trained_model, training_stats)
    """
    # Log the start of training (debug level for detailed params)
    logger.debug(f"Training params: timesteps={initial_timesteps}, max_iter={max_iterations}, stagnant={n_stagnant_loops}")

    # Get learning rate decay configuration
    use_lr_decay = config["model"].get("use_lr_decay", False)

    # Get the evaluation metric from config
    evaluation_metric = config.get("training", {}).get("evaluation", {}).get("metric", "return")
    
    # Perform hyperparameter tuning only if enabled AND no pre-tuned parameters provided
    if run_hyperparameter_tuning and model_params is None:
        logger.info(f"Starting hyperparameter tuning with {tuning_trials} trials using {evaluation_metric} metric")
        
        # Import the hyperparameter tuning function
        from walk_forward import hyperparameter_tuning
        
        # Run hyperparameter tuning with specified metric
        tuning_results = hyperparameter_tuning(
            train_data=train_data,
            validation_data=validation_data,
            n_trials=tuning_trials,
            eval_metric=evaluation_metric
        )
        
        model_params = tuning_results["best_params"]
        logger.info(f"Hyperparameter tuning completed. Best parameters: {model_params}")
        
        # Save tuning results if a folder is provided
        if tuning_folder:
            os.makedirs(tuning_folder, exist_ok=True)
            with open(os.path.join(tuning_folder, "best_params.json"), "w") as f:
                json.dump(model_params, f, indent=4)
    elif model_params is not None:
        logger.debug(f"Using provided hyperparameters: {model_params}")
    else:
        # Use default parameters from config
        model_params = {
            "ent_coef": config["model"].get("ent_coef", 0.01),
            "learning_rate": config["model"].get("learning_rate", 0.0003),
            "n_steps": config["model"].get("n_steps", 2048),
            "batch_size": config["model"].get("batch_size", 64),
            "gamma": config["model"].get("gamma", 0.99),
            "gae_lambda": config["model"].get("gae_lambda", 0.95),
        }
        logger.debug(f"Using default parameters from config")
    
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
        model_params=model_params,
        window_folder=window_folder
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
                "metric_used": result.get("metric_used", evaluation_metric),
                "loss_info": result.get("loss_info", {})
            }
            training_stats.append(entry)

        # Save training stats
        with open(os.path.join(window_folder, "training_stats.json"), "w") as f:
            json.dump(training_stats, f, indent=4)

    # Return as dict with loss_history included
    result_dict = {
        "iterations": training_stats,
        "loss_history": validation_results.get("loss_history", [])
    }

    return model, result_dict

def main():
    import torch, os
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # CUDA 10.2+
    set_global_seed(config["seed"])
    
    # Ensure learning rate decay parameters exist in config
    if "model" not in config:
        config["model"] = {}
    
    # Set default learning rate decay parameters if not present
    if "use_lr_decay" not in config["model"]:
        config["model"]["use_lr_decay"] = True  # Enable by default
    
    if "learning_rate" not in config["model"]:
        config["model"]["learning_rate"] = 0.0003  # Default initial learning rate
    
    if "final_learning_rate" not in config["model"]:
        config["model"]["final_learning_rate"] = 1e-5  # Default final learning rate
    
    # Load data using settings from YAML with three-way split
    train_data, validation_data, test_data = get_data(
        symbol=config["data"].get("symbol", "NQ=F"),
        period=config["data"].get("period", "60d"),
        interval=config["data"].get("interval", "5m"),
        train_ratio=config["data"].get("train_ratio", 0.6),
        validation_ratio=config["data"].get("validation_ratio", 0.2),
        use_yfinance=config["data"].get("use_yfinance", False)  # Default to local CSV
    )

    # Normalize data to prevent look-ahead bias (fit scaler ONLY on train data)
    from normalization import scale_window, get_standardized_column_names
    cols_to_scale = get_standardized_column_names(train_data)
    logger.info(f"Normalizing {len(cols_to_scale)} columns using train-only fitted scaler to prevent look-ahead bias")
    scaler, train_data, validation_data, test_data = scale_window(
        train_data=train_data,
        val_data=validation_data,
        test_data=test_data,
        cols_to_scale=cols_to_scale,
        feature_range=(-1, 1)
    )
    logger.info("Data normalization complete (scaler fitted only on training data)")

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
        additional_timesteps=additional_timesteps,
        window_folder=None  # Pass None as window_folder
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

