# Standard library imports
import json
import logging
import os
import tempfile
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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# RecurrentPPO for LSTM support
try:
    from sb3_contrib import RecurrentPPO
    RECURRENT_PPO_AVAILABLE = True
except ImportError:
    RECURRENT_PPO_AVAILABLE = False

# Local application imports
from action_space import action_label
from config import config
from environment import TradingEnv
from get_data import get_data
from utils.seeding import enable_full_determinism, set_global_seed
from utils.device import get_device
import money
from utils.log_format import (
    ANSI_BOLD,
    ANSI_GREEN,
    ANSI_RED,
    ANSI_RESET,
    bold,
    color_value,
    format_action_distribution,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEE_RATE = 0.0  # No trading fees


def _checkpoint_prefix(window_folder: str | None) -> str:
    """Return a writable prefix for transient best-model checkpoints."""
    if window_folder:
        os.makedirs(window_folder, exist_ok=True)
        return os.path.join(window_folder, "best_model")

    tmpdir = tempfile.mkdtemp(prefix="deeptrade_best_model_")
    return os.path.join(tmpdir, "best_model")


def _extract_metric_value(results: dict, evaluation_metric: str) -> tuple[float, str]:
    """Map the configured evaluation metric to its corresponding result field."""
    if evaluation_metric == "prediction_accuracy":
        return results["prediction_accuracy"], "prediction_accuracy"
    if evaluation_metric == "hit_rate":
        return results["hit_rate"], "hit_rate"
    if evaluation_metric == "calmar":
        return results["calmar_ratio"], "calmar"
    if evaluation_metric == "sortino":
        return results["sortino_ratio"], "sortino"
    return results["total_return_pct"], "return"


def _with_training_diagnostics(
    results: dict,
    *,
    metric_name: str,
    metric_value: float,
    loss_info: dict,
    min_trades: int,
    best_metric_value: float | None = None,
) -> dict:
    """Attach policy-collapse diagnostics to an evaluation result."""
    enriched = dict(results)
    action_counts = {int(k): int(v) for k, v in enriched.get("action_counts", {}).items()}
    total_actions = max(1, sum(action_counts.values()))
    long_action_pct = 100.0 * sum(action_counts.get(i, 0) for i in (0, 1, 2)) / total_actions
    short_action_pct = 100.0 * sum(action_counts.get(i, 0) for i in (3, 4, 5)) / total_actions
    flat_action_pct = 100.0 * action_counts.get(6, 0) / total_actions
    dominant_action_pct = max(long_action_pct, short_action_pct, flat_action_pct)
    active_actions = sum(1 for count in action_counts.values() if count > 0)
    max_flat_action_pct = float(config.get("training", {}).get("max_flat_action_pct", 80.0))
    max_allowed_drawdown_pct = float(config.get("training", {}).get("max_allowed_drawdown_pct", 40.0))
    max_drawdown_abs = abs(float(enriched.get("max_drawdown", 0.0)))
    metric_drop_from_best = None if best_metric_value is None else metric_value - best_metric_value

    collapse_flags = []
    if dominant_action_pct >= 90.0:
        collapse_flags.append("dominant_action")
    if flat_action_pct >= max_flat_action_pct:
        collapse_flags.append("flat_dominance")
    if active_actions <= 1:
        collapse_flags.append("single_action_policy")
    if max_drawdown_abs > max_allowed_drawdown_pct:
        collapse_flags.append("excessive_drawdown")
    if enriched.get("trade_count", 0) < min_trades:
        collapse_flags.append("too_few_trades")
    if metric_drop_from_best is not None and metric_drop_from_best <= -2.0:
        collapse_flags.append("metric_drop")

    has_enough_action_mix = (
        flat_action_pct < max_flat_action_pct
        and active_actions > 1
        and dominant_action_pct < 95.0
    )
    has_acceptable_drawdown = max_drawdown_abs <= max_allowed_drawdown_pct

    enriched.update({
        "metric_used": metric_name,
        "loss_info": loss_info,
        "has_enough_trades": enriched.get("trade_count", 0) >= min_trades,
        "has_enough_action_mix": has_enough_action_mix,
        "has_acceptable_drawdown": has_acceptable_drawdown,
        "min_trades_required": min_trades,
        "max_flat_action_pct": max_flat_action_pct,
        "max_allowed_drawdown_pct": max_allowed_drawdown_pct,
        "max_drawdown_abs": max_drawdown_abs,
        "action_counts": action_counts,
        "long_action_pct": long_action_pct,
        "short_action_pct": short_action_pct,
        "flat_action_pct": flat_action_pct,
        "dominant_action_pct": dominant_action_pct,
        "active_action_count": active_actions,
        "metric_drop_from_best": metric_drop_from_best,
        "collapse_flags": collapse_flags,
        "warning_policy_collapse": any(
            flag in collapse_flags for flag in ("dominant_action", "flat_dominance", "single_action_policy", "excessive_drawdown")
        ),
        "warning_metric_drop": "metric_drop" in collapse_flags,
        "action_distribution": {
            action_label(i): 100.0 * action_counts.get(i, 0) / total_actions for i in range(7)
        },
    })
    return enriched


def _infer_periods_per_year(index) -> float:
    """Infer evaluation periods per year from a datetime index."""
    if index is None or len(index) < 2:
        return 252.0

    dt_index = pd.DatetimeIndex(index)
    deltas = dt_index.to_series().diff().dropna().dt.total_seconds()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return 252.0

    median_seconds = float(deltas.median())
    if median_seconds <= 0:
        return 252.0
    seconds_per_year = 365.25 * 24 * 60 * 60
    return seconds_per_year / median_seconds


def _calculate_sortino_ratio(portfolio_array: np.ndarray, index) -> float:
    """Calculate a frequency-aware Sortino ratio from the portfolio path."""
    if portfolio_array is None or len(portfolio_array) < 2:
        return 0.0

    portfolio_returns = np.diff(portfolio_array) / portfolio_array[:-1]
    portfolio_returns = portfolio_returns[np.isfinite(portfolio_returns)]
    if portfolio_returns.size == 0:
        return 0.0

    periods_per_year = _infer_periods_per_year(index)
    downside_component = np.minimum(portfolio_returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside_component))))

    positive_portfolio = portfolio_array[portfolio_array > 0]
    if positive_portfolio.size >= 2:
        log_returns = np.diff(np.log(positive_portfolio))
        log_returns = log_returns[np.isfinite(log_returns)]
        annualized_return = float(np.mean(log_returns)) * periods_per_year if log_returns.size > 0 else 0.0
    else:
        annualized_return = float(np.mean(portfolio_returns)) * periods_per_year

    if downside_deviation <= 0:
        return max(0.0, annualized_return * 100.0)

    annualized_downside = downside_deviation * np.sqrt(periods_per_year)
    if annualized_downside <= 0:
        return 0.0

    return float(annualized_return / annualized_downside)


def _fallback_candidate_score(results: dict) -> float:
    """Score invalid checkpoints so we can still pick the least-bad fallback."""
    metric_value = float(
        results.get("sortino_ratio", results.get("calmar_ratio", results.get("total_return_pct", 0.0)))
    )
    trade_count = float(results.get("trade_count", 0))
    flat_action_pct = float(results.get("flat_action_pct", 100.0))
    active_action_count = float(results.get("active_action_count", 0))
    collapse_flags = set(results.get("collapse_flags", []))

    score = metric_value
    score += min(trade_count, 100.0) * 0.05
    score -= max(0.0, flat_action_pct - 70.0) * 0.15
    score += active_action_count * 0.5

    if "single_action_policy" in collapse_flags:
        score -= 4.0
    if "too_few_trades" in collapse_flags:
        score -= 2.0
    if "dominant_action" in collapse_flags:
        score -= 1.5

    return float(score)


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


class EntropyDecayCallback(BaseCallback):
    """
    Callback to decay entropy coefficient during training.
    Works with both PPO and RecurrentPPO (which doesn't support callable ent_coef).
    """

    def __init__(self, initial_ent: float, final_ent: float, total_timesteps: int, verbose=0):
        super().__init__(verbose)
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.total_timesteps = total_timesteps

    def _on_step(self):
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        self.model.ent_coef = self.initial_ent + (self.final_ent - self.initial_ent) * progress
        return True


def train_agent(train_data, total_timesteps: int):
    """
    Train a PPO model based on training data.

    Args:
        train_data (pd.DataFrame): Training dataset.
        total_timesteps (int): Number of training timesteps.

    Returns:
        model: Trained PPO or RecurrentPPO model.
    """
    base_seed = int(config.get("seed", 42))

    env_kwargs = dict(
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 2.50),
        position_size=config["environment"].get("position_size", 1),
    )

    # Validate with a single env first
    single_env = TradingEnv(train_data, **env_kwargs)
    check_env(single_env, skip_render_check=True)

    # Create vectorized environment for faster rollout collection
    # Get sequence model config
    seq_config = config.get("sequence_model", {})
    use_recurrent = seq_config.get("enabled", False) and RECURRENT_PPO_AVAILABLE
    n_envs = config.get("training", {}).get("n_envs", 4)
    env_kwargs["random_start_pct"] = float(config.get("training", {}).get("random_start_pct", 0.2))
    if use_recurrent:
        env_kwargs["min_episode_steps"] = max(2, int(config["model"].get("n_steps", 2048)))

    def make_env(data, **kwargs):
        return lambda: TradingEnv(data.copy(), **kwargs)

    if n_envs > 1:
        env = SubprocVecEnv([make_env(train_data, **env_kwargs) for _ in range(n_envs)])
        logger.info(f"Using SubprocVecEnv with {n_envs} processes for faster rollouts")
    else:
        env = DummyVecEnv([make_env(train_data, **env_kwargs)])
        logger.info(f"Using DummyVecEnv (single process)")
    env.seed(base_seed)

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
    use_ent_decay = config["model"].get("ent_coef_decay", False)
    entropy_callback = None
    if use_ent_decay and not use_recurrent:
        initial_ent = config["model"].get("ent_coef", 0.01)
        final_ent = config["model"].get("final_ent_coef", 0.001)
        from stable_baselines3.common.utils import LinearSchedule
        ent_coef = LinearSchedule(initial_ent, final_ent, 1.0)
        logger.debug(f"Entropy decay (schedule): {initial_ent} -> {final_ent}")
    elif use_ent_decay and use_recurrent:
        # RecurrentPPO doesn't support callable ent_coef, use callback instead
        initial_ent = config["model"].get("ent_coef", 0.01)
        final_ent = config["model"].get("final_ent_coef", 0.001)
        ent_coef = initial_ent  # Start with initial value
        entropy_callback = EntropyDecayCallback(initial_ent, final_ent, total_timesteps)
        logger.debug(f"Entropy decay (callback): {initial_ent} -> {final_ent}")
    else:
        ent_coef = config["model"].get("ent_coef", 0.01)

    # Initialize model with configured parameters
    if use_recurrent:
        logger.info("Using RecurrentPPO with LSTM policy for temporal sequence learning")
        shared_lstm = seq_config.get("shared_lstm", False)
        rollout_steps = int(config["model"].get("n_steps", 2048))
        recurrent_batch_size = rollout_steps * max(1, int(n_envs))
        model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            n_steps=rollout_steps,
            batch_size=recurrent_batch_size,
            seed=config.get('seed'),
            device=device,
            policy_kwargs={
                "lstm_hidden_size": seq_config.get("lstm_hidden_size", 256),
                "n_lstm_layers": seq_config.get("n_lstm_layers", 1),
                "shared_lstm": shared_lstm,
                "enable_critic_lstm": not shared_lstm,
                "net_arch": {"pi": [128, 64], "vf": [128, 64]},
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
            policy_kwargs={"net_arch": [128, 64]},
        )

    callbacks = [cb for cb in [entropy_callback] if cb is not None]
    logger.info("Starting training for %d timesteps", total_timesteps)
    model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=callbacks if callbacks else None)
    logger.info("Training completed")
    return model

def train_agent_iteratively(train_data, validation_data, initial_timesteps: int, max_iterations: int = 20,
                           n_stagnant_loops: int = 3, improvement_threshold: float = 0.1, additional_timesteps: int = 10000,
                           evaluation_metric: str = "return", model_params: dict = None, window_folder: str = None,
                           window_label: str = ""):
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
    base_seed = int(config.get("seed", 42))

    # Initialize training environment with realistic transaction costs
    env_kwargs = dict(
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 2.50),
        position_size=config["environment"].get("position_size", 1),
    )

    # Validate with a single env first
    single_env = TradingEnv(train_data, **env_kwargs)
    check_env(single_env, skip_render_check=True)
    obs, _ = single_env.reset(seed=base_seed)
    logger.debug(f"Observation space: {single_env.observation_space.shape}, indicators: {len(single_env.technical_indicators)}")

    # Create vectorized environment for faster rollout collection
    n_envs = config.get("training", {}).get("n_envs", 4)
    env_kwargs["random_start_pct"] = float(config.get("training", {}).get("random_start_pct", 0.2))

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
    
    best_model_path = _checkpoint_prefix(window_folder)
    fallback_model_path = f"{best_model_path}_fallback"
    fallback_score = float("-inf")
    fallback_results = None

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
    if use_recurrent:
        env_kwargs["min_episode_steps"] = max(2, int(model_params.get("n_steps", config["model"].get("n_steps", 2048))))

    def make_env(data, **kwargs):
        return lambda: TradingEnv(data.copy(), **kwargs)

    if n_envs > 1:
        train_env = SubprocVecEnv([make_env(train_data, **env_kwargs) for _ in range(n_envs)])
        logger.info(f"{window_label}Using SubprocVecEnv with {n_envs} processes for faster rollouts")
    else:
        train_env = DummyVecEnv([make_env(train_data, **env_kwargs)])
        logger.info(f"{window_label}Using DummyVecEnv (single process)")
    train_env.seed(base_seed)

    # Get entropy coefficient decay parameters from config or use defaults
    use_ent_decay = config["model"].get("ent_coef_decay", False)
    entropy_callback = None
    if use_ent_decay and not use_recurrent:
        initial_ent = model_params.get("ent_coef", 0.01)
        final_ent = config["model"].get("final_ent_coef", 0.001)
        from stable_baselines3.common.utils import LinearSchedule
        ent_coef = LinearSchedule(initial_ent, final_ent, 1.0)
    elif use_ent_decay and use_recurrent:
        # RecurrentPPO doesn't support callable ent_coef, use callback instead
        initial_ent = model_params.get("ent_coef", 0.01)
        final_ent = config["model"].get("final_ent_coef", 0.001)
        ent_coef = initial_ent
        entropy_callback = EntropyDecayCallback(initial_ent, final_ent, total_decay_timesteps)
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
        rollout_steps = int(model_params.get("n_steps", 2048))
        recurrent_batch_size = rollout_steps * max(1, int(n_envs))

        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            verbose=verbose_level,
            ent_coef=ent_coef,
            learning_rate=learning_rate,
            n_steps=rollout_steps,
            batch_size=recurrent_batch_size,
            gamma=model_params.get("gamma", 0.99),
            seed=config.get('seed'),
            gae_lambda=model_params.get("gae_lambda", 0.95),
            device=device,
            policy_kwargs={
                "lstm_hidden_size": lstm_hidden_size,
                "n_lstm_layers": n_lstm_layers,
                "shared_lstm": shared_lstm,
                "enable_critic_lstm": not shared_lstm,
                "net_arch": {"pi": [128, 64], "vf": [128, 64]},
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
            policy_kwargs={"net_arch": [128, 64]},
        )

    # Create callbacks
    loss_callback = LossTrackingCallback(verbose=verbose_level)
    callbacks = [loss_callback]
    if entropy_callback is not None:
        callbacks.append(entropy_callback)
    loss_history = []
    best_loss = float('inf')
    metric_stagnant_counter = 0  # Track validation metric stagnation, not loss

    # Initial training
    logger.info(f"{window_label}Starting initial training for {initial_timesteps} timesteps")
    model.learn(total_timesteps=initial_timesteps, progress_bar=False, callback=callbacks)

    # Get initial loss values
    initial_losses = loss_callback.get_latest_losses()
    if initial_losses["value_loss"] is not None:
        loss_history.append(initial_losses)
        best_loss = initial_losses["value_loss"]
        logger.info(f"{window_label}Iter 0 - Loss: {best_loss:.4f}")

    # Evaluate initial model on validation data using the specified metric
    if verbose_level > 0:
        logger.info(f"Evaluating model on validation data with {evaluation_metric} metric")

    # Use the appropriate evaluation function based on the metric
    if evaluation_metric == "prediction_accuracy":
        from walk_forward import evaluate_agent_prediction_accuracy
        results = evaluate_agent_prediction_accuracy(model, validation_data, verbose=verbose_level, deterministic=True)
    else:  # Default to "return"
        results = evaluate_agent(model, validation_data, verbose=verbose_level, deterministic=True)
    best_metric_value, metric_name = _extract_metric_value(results, evaluation_metric)

    # Store best model via checkpoint, not by mutable object reference.
    min_trades = max(20, len(validation_data) // 200)  # At least 1 trade per 200 bars
    results = _with_training_diagnostics(
        results,
        metric_name=metric_name,
        metric_value=best_metric_value,
        loss_info=initial_losses,
        min_trades=min_trades,
    )
    initial_candidate_ok = (
        results["has_enough_trades"]
        and results["has_enough_action_mix"]
        and results["has_acceptable_drawdown"]
    )
    results["is_best"] = initial_candidate_ok
    best_results = dict(results)
    fallback_score = _fallback_candidate_score(results)
    fallback_results = dict(results)
    model.save(fallback_model_path)
    if not initial_candidate_ok:
        best_metric_value = float("-inf")
    model.save(best_model_path)

    # Log evaluation results based on metric (color-coded by sign)
    logger.info(
        f"{window_label}Initial training completed. "
        f"Validation {metric_name.replace('_', ' ').title()}: {color_value(best_metric_value)}, "
        f"Validation Portfolio: ${results['final_portfolio_value']:.2f}"
    )
    
    # Store all results for comparison
    all_results = [dict(results)]

    # Evaluate every iteration by default to catch sharp post-peak degradation.
    eval_frequency = config.get("training", {}).get("eval_frequency", 1)

    # Continue training until max_iterations or loss plateau
    for iteration in range(1, max_iterations + 1):
        # Reset callback for this iteration
        loss_callback.reset()

        # Train for additional timesteps
        model.learn(total_timesteps=additional_timesteps, progress_bar=False, callback=callbacks)

        # Get loss values for this iteration
        iter_losses = loss_callback.get_latest_losses()
        current_loss = iter_losses["value_loss"] if iter_losses["value_loss"] is not None else float('inf')
        if iter_losses["value_loss"] is not None:
            loss_history.append(iter_losses)

        # Skip evaluation on non-evaluation iterations (except the last one)
        if eval_frequency > 1 and iteration % eval_frequency != 0 and iteration < max_iterations:
            logger.info(f"{window_label}Iter {iteration} - Training only (eval every {eval_frequency} iters), loss={current_loss:.4f}")
            continue

        # Evaluate the model on validation data using the specified metric
        if evaluation_metric == "prediction_accuracy":
            from walk_forward import evaluate_agent_prediction_accuracy
            results = evaluate_agent_prediction_accuracy(model, validation_data, verbose=verbose_level, deterministic=True)
        else:  # Default to "return"
            results = evaluate_agent(model, validation_data, verbose=verbose_level, deterministic=True)
        current_metric_value, _ = _extract_metric_value(results, evaluation_metric)

        results = _with_training_diagnostics(
            results,
            metric_name=metric_name,
            metric_value=current_metric_value,
            loss_info=iter_losses,
            min_trades=min_trades,
            best_metric_value=best_metric_value,
        )
        results["is_best"] = False  # Will update this if it becomes the best model

        current_fallback_score = _fallback_candidate_score(results)
        if current_fallback_score > fallback_score:
            fallback_score = current_fallback_score
            fallback_results = dict(results)
            model.save(fallback_model_path)

        # Track best loss (for logging only)
        if current_loss < best_loss:
            best_loss = current_loss

        # Model selection AND early stopping based on VALIDATION metric (higher is better for calmar/return)
        # Require minimum trade count to prevent selecting near-static models.
        has_enough_trades = results["has_enough_trades"]
        has_enough_action_mix = results["has_enough_action_mix"]
        has_acceptable_drawdown = results["has_acceptable_drawdown"]
        if current_metric_value > best_metric_value and has_enough_trades and has_enough_action_mix and has_acceptable_drawdown:
            best_metric_value = current_metric_value
            results["is_best"] = True
            best_results = dict(results)
            metric_stagnant_counter = 0  # Reset stagnation counter on improvement
            model.save(best_model_path)

            best_color = ANSI_GREEN if current_metric_value >= 0 else ANSI_RED
            new_best_tag = f"{ANSI_BOLD}{best_color}\u2713 New best{ANSI_RESET}"
            best_val_str = f"{ANSI_BOLD}{best_color}{current_metric_value:.2f}{ANSI_RESET}"
            collapse_warning = ""
            if results["warning_policy_collapse"]:
                collapse_warning = f", Warnings={results['collapse_flags']}"
            logger.info(
                f"{window_label}Iter {iteration} - {new_best_tag} "
                f"(val {metric_name}={best_val_str}): "
                f"Train loss={current_loss:.4f}, Portfolio: ${results['final_portfolio_value']:.2f}, "
                f"Trades={results['trade_count']}, Actions={{{format_action_distribution(results['action_counts'])}}}, "
                f"ActionPct(L/S/F)=({results['long_action_pct']:.1f}/{results['short_action_pct']:.1f}/{results['flat_action_pct']:.1f})"
                f"{collapse_warning}"
            )
        elif current_metric_value > best_metric_value and (not has_enough_trades or not has_enough_action_mix or not has_acceptable_drawdown):
            metric_stagnant_counter += 1
            reject_reasons = []
            if not has_enough_trades:
                reject_reasons.append(f"too few trades ({results['trade_count']}<{min_trades})")
            if not has_enough_action_mix:
                reject_reasons.append(
                    f"degenerate action mix (flat={results['flat_action_pct']:.1f}%, active_actions={results['active_action_count']})"
                )
            if not has_acceptable_drawdown:
                reject_reasons.append(
                    f"max drawdown too high ({results['max_drawdown_abs']:.1f}%>{results['max_allowed_drawdown_pct']:.1f}%)"
                )
            logger.info(
                f"{window_label}Iter {iteration} - Val {metric_name}={color_value(current_metric_value)} "
                f"but rejected for {'; '.join(reject_reasons)} "
                f"[no improvement {metric_stagnant_counter}/{n_stagnant_loops}]"
            )
        else:
            metric_stagnant_counter += 1
            warnings = f", Warnings={results['collapse_flags']}" if results["collapse_flags"] else ""
            logger.info(
                f"{window_label}Iter {iteration} - Val {metric_name}={color_value(current_metric_value)}, "
                f"Train loss={current_loss:.4f} [no improvement {metric_stagnant_counter}/{n_stagnant_loops}], "
                f"Portfolio=${results['final_portfolio_value']:.2f}, Trades={results['trade_count']}, "
                f"Actions={{{format_action_distribution(results['action_counts'])}}}, "
                f"ActionPct(L/S/F)=({results['long_action_pct']:.1f}/{results['short_action_pct']:.1f}/{results['flat_action_pct']:.1f})"
                f"{warnings}"
            )

        all_results.append(dict(results))

        # Early stopping based on validation metric plateau (not loss)
        if metric_stagnant_counter >= n_stagnant_loops:
            logger.info(f"{window_label}Early stop: validation {metric_name} plateau for {n_stagnant_loops} iterations")
            break

    # Add loss history to best_results for saving
    best_results["loss_history"] = loss_history
    if best_metric_value == float("-inf"):
        logger.warning(
            f"{window_label}No validation checkpoint satisfied trade/action-mix gates. "
            f"Using least-bad fallback checkpoint instead."
        )
        best_results = dict(fallback_results or best_results)
        best_results["selected_via_fallback"] = True
        best_results["fallback_score"] = fallback_score
        best_model = model.__class__.load(fallback_model_path, env=train_env, device=device)
    else:
        best_results["selected_via_fallback"] = False
        best_model = model.__class__.load(best_model_path, env=train_env, device=device)

    total_iterations = iteration if 'iteration' in dir() else 0
    logger.info(f"{window_label}Training complete after {total_iterations} iterations. Best {metric_name}: {best_metric_value:.2f}")
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

        # Calculate close_norm using min/max scaling to [0, 1]
        close_min = test_data[close_col].min()
        close_max = test_data[close_col].max()
        close_range = close_max - close_min
        if close_range > 0:
            test_data['close_norm'] = ((test_data[close_col] - close_min) / close_range).clip(0, 1)
        else:
            test_data['close_norm'] = 0.5

    # Create evaluation environment with realistic transaction costs
    env = TradingEnv(
        test_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 2.50),
        position_size=config["environment"].get("position_size", 1)
    )

    # Reset environment and store initial net worth
    obs, _ = env.reset(seed=int(config.get("seed", 42)))
    initial_net_worth = env.net_worth  # STORE INITIAL NET WORTH HERE

    # Check if model is RecurrentPPO (has LSTM)
    is_recurrent = hasattr(model, 'policy') and hasattr(model.policy, 'lstm')

    # Initialize LSTM states for recurrent models
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    # Track portfolio values and actions over time
    portfolio_history = [float(env.net_worth)]
    action_history = []
    position_history = [int(env.position)]
    drawdown_history = [0.0]
    evaluation_dates = []
    price_history = []

    # Start evaluation
    done = False
    total_reward = 0
    step_count = 0
    trade_count = 0

    # Track trading state
    current_position = 0  # 0 = no position, 1 = long, -1 = short
    current_contracts = 0

    # Track entry points for trades
    entry_price = Decimal('0')
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

    # Get initial price for reference from the environment's sanitized price path.
    current_price_initial = round(float(env._get_current_price()), 2)

    if verbose > 0:
        print(f"Starting evaluation with initial price: ${current_price_initial}")
        if is_recurrent:
            print("Using RecurrentPPO with LSTM state tracking")

    # Main evaluation loop
    while not done:
        # Get current step information
        current_step = env.current_step

        # Get current price from the environment so trade accounting uses sanitized prices.
        if current_step <= env.total_steps:
            current_price = round(float(env._get_current_price()), 2)
        else:
            current_price = current_price_initial  # Fallback if index out of bounds
        current_timestamp = test_data.index[current_step] if current_step < len(test_data) else test_data.index[-1]
        evaluation_dates.append(current_timestamp)
        price_history.append(current_price)

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

        # Handle MultiDiscrete action space (dynamic SL/TP)
        # The environment's step() method handles both int and array actions
        # Use np.ndim to safely check if action is an array (avoids len() on scalar)
        if isinstance(action, np.ndarray) and action.ndim > 0 and action.size > 1:
            # MultiDiscrete action: [position_action, sl_idx, tp_idx]
            action_for_history = int(action[0])  # Store position action for history
        else:
            action_for_history = int(action)
        action_history.append(action_for_history)

        # Take action in environment (pass action as-is, env handles both types)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Get portfolio value AFTER the step
        portfolio_value_after = float(money.format_money(env.net_worth, 2))
        portfolio_history.append(portfolio_value_after)
        position_history.append(int(env.position))
        if env.max_net_worth > 0:
            drawdown_history.append(float((env.net_worth - env.max_net_worth) / env.max_net_worth * 100))
        else:
            drawdown_history.append(0.0)

        # Track trade history based on actual environment rebalances.
        position_changed = bool(info.get("position_changed", False))
        old_position = int(info.get("old_position", current_position))
        new_position = int(info.get("position", env.position))
        old_contracts = int(info.get("old_contracts", current_contracts))
        new_contracts = int(info.get("current_contracts", current_contracts))

        if position_changed:
            trade_count += 1
            realized_trade = (
                old_contracts != 0 and (
                    new_contracts == 0
                    or np.sign(old_contracts) != np.sign(new_contracts)
                    or abs(new_contracts) < abs(old_contracts)
                )
            )

            if old_contracts == 0 and new_contracts != 0:
                trade_type = "Long Entry" if new_contracts > 0 else "Short Entry"
            elif old_contracts != 0 and new_contracts == 0:
                trade_type = "Exit"
            elif old_contracts != 0 and new_contracts != 0 and np.sign(old_contracts) != np.sign(new_contracts):
                trade_type = "Flip"
            elif abs(new_contracts) > abs(old_contracts):
                trade_type = "Scale In"
            elif abs(new_contracts) < abs(old_contracts):
                trade_type = "Scale Out"
            else:
                trade_type = "Rebalance"

            # Only reductions/exits/flips realize P&L in a meaningful way for hit-rate tracking.
            is_profitable = False
            if realized_trade and entry_price > 0:
                if old_position == 1:  # Was long
                    is_profitable = current_price > entry_price
                elif old_position == -1:  # Was short
                    is_profitable = current_price < entry_price

            if old_contracts == 0:
                pos_from = 0.0
                pos_to = current_price
            elif new_contracts == 0:
                pos_from = float(entry_price) if entry_price > 0 else float(current_price)
                pos_to = 0.0
            else:
                pos_from = float(entry_price) if entry_price > 0 else float(current_price)
                pos_to = current_price

            trade_history.append({
                "date": test_data.index[current_step],
                "trade_type": trade_type,
                "price": current_price,
                "portfolio_value": portfolio_value_after,
                "profitable": is_profitable,
                "position_from": pos_from,
                "position_to": pos_to,
                "old_contracts": old_contracts,
                "new_contracts": new_contracts,
                "avg_entry_price": float(info.get("avg_entry_price", 0.0)),
                "realized_trade": realized_trade,
                "exit_reason": ""
            })

            if new_contracts != 0:
                entry_price = money.to_decimal(info.get("avg_entry_price", current_price))
                entry_step = current_step
            else:
                entry_price = Decimal('0')
                entry_step = -1

            current_position = new_position
            current_contracts = new_contracts

        current_position = new_position
        current_contracts = new_contracts

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
    completed_trades = sum(1 for t in trade_history if t.get("realized_trade", False))
    profitable_count = sum(1 for t in trade_history if t.get("profitable", False))
    hit_rate = (profitable_count / completed_trades) * 100 if completed_trades > 0 else 0
    
    # Calculate portfolio value
    final_portfolio_value = float(money.format_money(env.net_worth, 2))
    
    # Calculate total return
    total_return_pct = float(money.format_money(return_pct, 2))
    
    # Calculate trade history
    trade_history_df = pd.DataFrame(trade_history)
    
    # Calculate action distribution across the 7 target-allocation actions.
    action_counts = {i: 0 for i in range(7)}
    for action in action_history:
        action_val = int(action)
        if action_val in action_counts:
            action_counts[action_val] += 1
        else:
            action_counts[action_val] = 1  # Handle unexpected actions

    # Calculate max drawdown from portfolio history
    portfolio_array = np.array(portfolio_history)
    running_max = np.maximum.accumulate(portfolio_array)
    drawdowns = (portfolio_array - running_max) / running_max * 100  # As percentage
    max_drawdown = float(np.min(drawdowns))  # Most negative value (will be <= 0)

    # Calculate Calmar ratio (return / |max_drawdown|)
    # Higher is better - rewards high returns with low drawdowns
    if max_drawdown == 0:
        calmar_ratio = total_return_pct if total_return_pct > 0 else 0.0
    else:
        calmar_ratio = total_return_pct / abs(max_drawdown)

    sortino_ratio = _calculate_sortino_ratio(portfolio_array, evaluation_dates)

    # Prepare results
    results = {
        "final_portfolio_value": final_portfolio_value,
        "total_return_pct": total_return_pct,
        "trade_count": trade_count,
        "hit_rate": hit_rate,
        "final_position": env.position,
        "dates": evaluation_dates,
        "price_history": price_history,
        "portfolio_history": portfolio_history,
        "position_history": position_history,
        "drawdown_history": drawdown_history,
        "trade_history": trade_history_df.to_dict(orient="records"),
        "buy_dates": [],
        "buy_prices": [],
        "sell_dates": [],
        "sell_prices": [],
        "action_counts": action_counts,
        "completed_trades": completed_trades,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "sortino_ratio": sortino_ratio
    }

    # Add additional information
    results["metric_used"] = "return"
    results["is_best"] = False
    results["profitable_trades"] = profitable_count
    results["profitable"] = total_return_pct > 0
    results["profitable_pct"] = total_return_pct
    results["profitable_trade_pct"] = results["profitable_trades"] / completed_trades * 100 if completed_trades > 0 else 0
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
    portfolio_dates = dates
    if len(portfolio_history) == len(dates) + 1 and dates:
        portfolio_dates = [dates[0]] + list(dates)
    elif len(portfolio_history) != len(dates):
        portfolio_dates = list(dates)[:len(portfolio_history)]

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
            x=portfolio_dates,
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
                         run_hyperparameter_tuning=False, tuning_trials=30, tuning_folder=None, model_params=None,
                         window_label: str = ""):
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
        window_folder=window_folder,
        window_label=window_label
    )
    
    training_stats = []

    # Prepare training stats for all iterations
    for i, result in enumerate(all_results):
        entry = {
            "iteration": i,
            "return_pct": result.get("total_return_pct", 0),
            "portfolio_value": result.get("final_portfolio_value", 0),
            "hit_rate": result.get("hit_rate", 0),
            "prediction_accuracy": result.get("prediction_accuracy", 0),
            "sortino_ratio": result.get("sortino_ratio", 0),
            "calmar_ratio": result.get("calmar_ratio", 0),
            "max_drawdown": result.get("max_drawdown", 0),
            "trade_count": result.get("trade_count", 0),
            "is_best": result.get("is_best", False),
            "metric_used": result.get("metric_used", evaluation_metric),
            "has_enough_trades": result.get("has_enough_trades", False),
            "has_enough_action_mix": result.get("has_enough_action_mix", False),
            "min_trades_required": result.get("min_trades_required", 0),
            "max_flat_action_pct": result.get("max_flat_action_pct", 0),
            "action_counts": result.get("action_counts", {}),
            "long_action_pct": result.get("long_action_pct", 0),
            "short_action_pct": result.get("short_action_pct", 0),
            "flat_action_pct": result.get("flat_action_pct", 0),
            "dominant_action_pct": result.get("dominant_action_pct", 0),
            "active_action_count": result.get("active_action_count", 0),
            "metric_drop_from_best": result.get("metric_drop_from_best"),
            "collapse_flags": result.get("collapse_flags", []),
            "warning_policy_collapse": result.get("warning_policy_collapse", False),
            "warning_metric_drop": result.get("warning_metric_drop", False),
            "loss_info": result.get("loss_info", {})
        }
        training_stats.append(entry)

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
    enable_full_determinism(int(config["seed"]))
    
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
