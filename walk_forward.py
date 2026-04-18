import logging
import re
import pandas as pd
import numpy as np
import os
import json
import yaml
import copy
from datetime import datetime, time, timedelta
import pytz
from typing import List, Dict, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _sanitize_cuda_library_path() -> None:
    """Prefer this repo's bundled NVIDIA libs over incompatible foreign venvs."""
    venv_root = os.path.join(os.path.dirname(__file__), "venv", "lib", "python3.10", "site-packages", "nvidia")
    local_lib_dirs = [
        "nvjitlink/lib",
        "cuda_runtime/lib",
        "cuda_nvrtc/lib",
        "curand/lib",
        "cusparse/lib",
        "cusolver/lib",
        "cudnn/lib",
        "nccl/lib",
        "cufft/lib",
        "cublas/lib",
        "cusparselt/lib",
        "nvshmem/lib",
    ]
    preferred = [os.path.join(venv_root, rel) for rel in local_lib_dirs if os.path.isdir(os.path.join(venv_root, rel))]
    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    filtered = [p for p in existing if "/home/orion/Documents/loki/.venv/lib/python3.10/site-packages/nvidia/" not in p]
    deduped = []
    for path in preferred + filtered:
        if path and path not in deduped:
            deduped.append(path)
    os.environ["LD_LIBRARY_PATH"] = ":".join(deduped)


_sanitize_cuda_library_path()

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import optuna
import math
from decimal import Decimal
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import pickle
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import train as train_module

# RecurrentPPO for LSTM support
try:
    from sb3_contrib import RecurrentPPO
    RECURRENT_PPO_AVAILABLE = True
except ImportError:
    RECURRENT_PPO_AVAILABLE = False

from action_space import ACTION_COUNT, FLAT_ACTION, LONG_ACTIONS, SHORT_ACTIONS, action_direction, action_label
from environment import TradingEnv, get_market_feature_columns, get_representation_mode
from get_data import ensure_numeric, get_data, process_technical_indicators, select_session_dominant_contract
from train import evaluate_agent, get_configured_algorithm, plot_results, train_walk_forward_model
from trade import trade_with_risk_management, save_trade_history  # Import trade_with_risk_management and save_trade_history
from config import config
import money
from utils.seeding import enable_full_determinism, seed_worker, set_global_seed  # Import the seed helpers
from utils.device import get_device  # Import device utility
from normalization import scale_window, get_standardized_column_names  # Import both functions from normalization
from indicators.lstm_features import LSTMFeatureGenerator, tune_lstm_hyperparameters
from tuning_prune import normalize_action_counts as _normalize_action_counts
from tuning_prune import should_hard_prune_trial as _should_hard_prune_trial
from tuning_prune import tuning_prune_diagnostics as _tuning_prune_diagnostics
from utils.synthetic_bears import augment_with_synthetic_bears, extract_ohlcv_frame
from utils.session_context import build_session_context, session_config_from_mapping

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
from reporting.walk_forward_report import generate_walk_forward_report

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


class _ConstantActionModel:
    """Minimal model adapter for scoring trivial constant-action baselines."""

    def __init__(self, action: int):
        self.action = int(action)

    def predict(self, obs, deterministic=True):
        return self.action, None

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


def _session_timestamp() -> str:
    """Return a high-resolution timestamp for unique walk-forward session folders."""
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def _narrow_hp_config(base_hp_config: Dict, best_params: Dict, shrink: float) -> Dict:
    """Create a narrower stage-2 search space around a stage-1 winner."""
    narrowed = copy.deepcopy(base_hp_config)
    shrink = min(max(float(shrink), 0.05), 0.95)
    for name, value in best_params.items():
        cfg = narrowed.get(name)
        if not isinstance(cfg, dict):
            continue
        if "min" in cfg and "max" in cfg:
            old_min = float(cfg["min"])
            old_max = float(cfg["max"])
            center = float(value)
            if cfg.get("log", False) and old_min > 0 and old_max > 0 and center > 0:
                log_min = np.log(old_min)
                log_max = np.log(old_max)
                log_center = np.log(center)
                half_width = max((log_max - log_min) * shrink * 0.5, 1e-6)
                cfg["min"] = float(np.exp(max(log_min, log_center - half_width)))
                cfg["max"] = float(np.exp(min(log_max, log_center + half_width)))
            else:
                half_width = max((old_max - old_min) * shrink * 0.5, 1e-9)
                cfg["min"] = max(old_min, center - half_width)
                cfg["max"] = min(old_max, center + half_width)
    return narrowed


def _sample_stage_params(trial, hp_config: Dict) -> Dict[str, float]:
    """Sample the configured PPO/reward/augmentation tuning parameters."""
    params: Dict[str, float] = {}
    integer_params = {"n_steps", "reward_flat_time_grace_steps"}
    for name in (
        "learning_rate",
        "n_steps",
        "ent_coef",
        "gamma",
        "gae_lambda",
        "reward_turnover_penalty",
        "reward_loss_multiplier",
        "reward_drawdown_penalty",
        "reward_drawdown_penalty_threshold",
        "reward_flat_time_penalty",
        "reward_flat_time_grace_steps",
        "synthetic_oversample_ratio",
    ):
        cfg = hp_config.get(name, {})
        if not isinstance(cfg, dict):
            continue
        if "choices" in cfg:
            params[name] = trial.suggest_categorical(name, list(cfg["choices"]))
        elif name in integer_params:
            params[name] = trial.suggest_int(
                name,
                int(cfg.get("min", 1)),
                int(cfg.get("max", 64)),
                log=cfg.get("log", False),
            )
        else:
            params[name] = trial.suggest_float(
                name,
                float(cfg.get("min", 1e-5)),
                float(cfg.get("max", 1e-2)),
                log=cfg.get("log", False),
            )
    return params


def _resolve_qrdqn_hp_config(tuning_config: Dict) -> Dict:
    """Resolve a minimal QR-DQN search space from config with safe defaults."""
    shared = tuning_config.get("parameters", {})
    qrdqn_cfg = copy.deepcopy(tuning_config.get("qrdqn_parameters", {}))
    config_map = {
        "learning_rate": copy.deepcopy(
            qrdqn_cfg.get(
                "learning_rate",
                shared.get("learning_rate", {"min": 1e-5, "max": 5e-4, "log": True}),
            )
        ),
        "batch_size": copy.deepcopy(
            qrdqn_cfg.get(
                "batch_size",
                {"choices": [32, 64, 128, 256]},
            )
        ),
        "gamma": copy.deepcopy(
            qrdqn_cfg.get(
                "gamma",
                shared.get("gamma", {"min": 0.985, "max": 0.999, "log": False}),
            )
        ),
        "buffer_size": copy.deepcopy(
            qrdqn_cfg.get(
                "buffer_size",
                {"choices": [50000, 100000, 200000, 500000]},
            )
        ),
        "learning_starts": copy.deepcopy(
            qrdqn_cfg.get(
                "learning_starts",
                {"choices": [2000, 5000, 10000]},
            )
        ),
        "train_freq": copy.deepcopy(
            qrdqn_cfg.get(
                "train_freq",
                {"choices": [4, 8, 16, 32]},
            )
        ),
        "gradient_steps": copy.deepcopy(
            qrdqn_cfg.get(
                "gradient_steps",
                {"choices": [1, 2, 4, 8]},
            )
        ),
        "target_update_interval": copy.deepcopy(
            qrdqn_cfg.get(
                "target_update_interval",
                {"choices": [1000, 2000, 5000, 10000]},
            )
        ),
        "exploration_fraction": copy.deepcopy(
            qrdqn_cfg.get(
                "exploration_fraction",
                {"min": 0.05, "max": 0.20, "log": False},
            )
        ),
        "exploration_final_eps": copy.deepcopy(
            qrdqn_cfg.get(
                "exploration_final_eps",
                {"min": 0.02, "max": 0.10, "log": False},
            )
        ),
    }
    for name in (
        "reward_turnover_penalty",
        "reward_loss_multiplier",
        "reward_drawdown_penalty",
        "reward_drawdown_penalty_threshold",
        "reward_flat_time_penalty",
        "reward_flat_time_grace_steps",
        "synthetic_oversample_ratio",
    ):
        shared_cfg = shared.get(name)
        if isinstance(shared_cfg, dict):
            config_map[name] = copy.deepcopy(shared_cfg)
    return config_map


def _sample_qrdqn_params(trial, hp_config: Dict) -> Dict[str, float]:
    """Sample QR-DQN-specific hyperparameters."""
    params: Dict[str, float] = {}
    integer_params = {
        "batch_size",
        "buffer_size",
        "learning_starts",
        "train_freq",
        "gradient_steps",
        "target_update_interval",
        "reward_flat_time_grace_steps",
    }
    for name in (
        "learning_rate",
        "batch_size",
        "gamma",
        "buffer_size",
        "learning_starts",
        "train_freq",
        "gradient_steps",
        "target_update_interval",
        "exploration_fraction",
        "exploration_final_eps",
        "reward_turnover_penalty",
        "reward_loss_multiplier",
        "reward_drawdown_penalty",
        "reward_drawdown_penalty_threshold",
        "reward_flat_time_penalty",
        "reward_flat_time_grace_steps",
        "synthetic_oversample_ratio",
    ):
        cfg = hp_config.get(name, {})
        if not isinstance(cfg, dict):
            continue
        if "choices" in cfg:
            params[name] = trial.suggest_categorical(name, list(cfg["choices"]))
        elif name in integer_params:
            params[name] = trial.suggest_int(
                name,
                int(cfg.get("min", 32)),
                int(cfg.get("max", 256 if name == "batch_size" else 10000)),
                log=cfg.get("log", False),
            )
        else:
            params[name] = trial.suggest_float(
                name,
                float(cfg.get("min", 1e-5)),
                float(cfg.get("max", 1e-2)),
                log=cfg.get("log", False),
            )
    return params


def _run_tuning_stage(
    *,
    stage_name: str,
    hp_config: Dict,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    evaluation_sets: List[pd.DataFrame],
    reference_columns: List[str],
    baseline_scores: List[Dict[int, float]],
    baseline_margin: float,
    n_trials: int,
    tuning_timesteps: int,
    window_folder: str | None,
    use_parallel: bool,
    n_jobs: int,
) -> Dict:
    """Run one tuning stage and return study details plus best params."""
    tuning_config = config.get("hyperparameter_tuning", {})
    holdout_validation_sets = evaluation_sets[1:]
    seq_config = config.get("sequence_model", {})
    use_recurrent = seq_config.get("enabled", False) and RECURRENT_PPO_AVAILABLE

    study_name = f"{stage_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    sampler = optuna.samplers.TPESampler(
        seed=config.get("seed", 42),
        multivariate=True,
        n_startup_trials=min(10, max(5, n_trials // 3)),
        n_ei_candidates=24,
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(8, max(4, n_trials // 4)), n_warmup_steps=2, interval_steps=1)

    storage = None
    if use_parallel:
        if window_folder:
            storage = f"sqlite:///{window_folder}/{stage_name}_optuna.db"
        else:
            storage = f"sqlite:///{stage_name}_optuna.db"

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial):
        sampled = _sample_stage_params(trial, hp_config)
        learning_rate = float(sampled["learning_rate"])
        n_steps = int(sampled["n_steps"])
        ent_coef = float(sampled["ent_coef"])
        gamma = float(sampled.get("gamma", config.get("model", {}).get("gamma", 0.99)))
        gae_lambda = float(sampled.get("gae_lambda", config.get("model", {}).get("gae_lambda", 0.95)))
        lstm_hidden_size = int(seq_config.get("lstm_hidden_size", 256))
        n_lstm_layers = int(seq_config.get("n_lstm_layers", 1))

        trial_overrides = {}
        for key in (
            "reward_turnover_penalty",
            "reward_loss_multiplier",
            "reward_drawdown_penalty",
            "reward_drawdown_penalty_threshold",
            "reward_flat_time_penalty",
            "reward_flat_time_grace_steps",
            "synthetic_oversample_ratio",
        ):
            if key in sampled:
                trial_overrides[key] = sampled[key]

        original_overrides = _apply_trial_config_overrides(trial_overrides)
        min_episode_steps = max(2, n_steps) if use_recurrent else 2
        tuning_random_start_pct = 0.0 if use_recurrent else float(config.get("training", {}).get("random_start_pct", 0.2))
        train_env, n_envs = _build_tuning_env(
            train_data,
            random_start_pct=tuning_random_start_pct,
            min_episode_steps=min_episode_steps,
        )
        batch_size = n_steps * n_envs if use_recurrent else min(64, n_steps * n_envs)
        device = get_device(seq_config.get("device", "auto"), for_recurrent=use_recurrent)

        if use_recurrent:
            shared_lstm = seq_config.get("shared_lstm", False)
            model = RecurrentPPO(
                "MlpLstmPolicy",
                train_env,
                verbose=0,
                learning_rate=learning_rate,
                n_steps=n_steps,
                ent_coef=ent_coef,
                batch_size=batch_size,
                gamma=gamma,
                seed=config.get('seed'),
                gae_lambda=gae_lambda,
                max_grad_norm=config["model"].get("max_grad_norm", 0.5),
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
            model = PPO(
                "MlpPolicy",
                train_env,
                verbose=0,
                learning_rate=learning_rate,
                n_steps=n_steps,
                ent_coef=ent_coef,
                batch_size=batch_size,
                gamma=gamma,
                seed=config.get('seed'),
                gae_lambda=gae_lambda,
                max_grad_norm=config["model"].get("max_grad_norm", 0.5),
                device=device,
                policy_kwargs={"net_arch": [128, 64]},
            )

        prune_interval = int(tuning_config.get("pruning_eval_steps", 5000))
        prune_interval = max(1000, min(prune_interval, tuning_timesteps))
        step_results = None
        try:
            timesteps_completed = 0
            prune_step = 0
            while timesteps_completed < tuning_timesteps:
                chunk = min(prune_interval, tuning_timesteps - timesteps_completed)
                model.learn(total_timesteps=chunk, reset_num_timesteps=False)
                timesteps_completed += chunk
                prune_step += 1

                primary_results = evaluate_agent(model, validation_data, verbose=0, deterministic=True)
                primary_score, _ = _score_tuning_trial(primary_results, len(validation_data))
                trial.report(primary_score, prune_step)

                should_prune, prune_reasons = _should_hard_prune_trial(
                    primary_results,
                    tuning_config,
                    prune_step=prune_step,
                )
                action_dist = format_action_distribution(
                    _normalize_action_counts(primary_results.get("action_counts"))
                )
                if should_prune:
                    trial.set_user_attr("hard_prune_reasons", prune_reasons)
                    logger.info(
                        f"{stage_name} Trial {trial.number} pruned at step {prune_step} "
                        f"({timesteps_completed}/{tuning_timesteps} steps): "
                        f"score={primary_score:.2f}, reasons={prune_reasons}, actions=[{action_dist}]"
                    )
                    raise optuna.TrialPruned()
                if timesteps_completed < tuning_timesteps and trial.should_prune():
                    logger.info(
                        f"{stage_name} Trial {trial.number} pruned by Optuna median pruner at step {prune_step} "
                        f"({timesteps_completed}/{tuning_timesteps} steps): "
                        f"score={primary_score:.2f}, actions=[{action_dist}]"
                    )
                    raise optuna.TrialPruned()

            evaluation_scores = []
            evaluation_results = []
            for dataset in evaluation_sets:
                aligned_dataset = _align_feature_columns(reference_columns, dataset)
                results = evaluate_agent(model, aligned_dataset, verbose=0, deterministic=True)
                score, diagnostics = _score_tuning_trial(results, len(aligned_dataset))
                evaluation_results.append((results, diagnostics))
                evaluation_scores.append(score)
            composite_score, aggregate_diag = _summarize_trial_scores(
                evaluation_scores,
                baseline_scores,
                baseline_margin=baseline_margin,
            )
            results, tuning_diag = evaluation_results[0]
            step_results = {
                "primary_results": results,
                "primary_diag": tuning_diag,
                "aggregate_diag": aggregate_diag,
            }
        except (ValueError, RuntimeError) as e:
            logger.warning(f"{stage_name} trial {trial.number} failed during training: {e}")
            return -100.0
        finally:
            train_env.close()
            _restore_trial_config_overrides(original_overrides)

        results = step_results["primary_results"]
        tuning_diag = step_results["primary_diag"]
        aggregate_diag = step_results["aggregate_diag"]
        composite_score = float(aggregate_diag["aggregate_score"] - aggregate_diag["baseline_penalty"])
        trial.set_user_attr("tuning_diagnostics", {
            "stage": stage_name,
            "base_score": round(tuning_diag["base_score"], 6),
            "total_penalty": round(tuning_diag["total_penalty"], 6),
            "trade_count": tuning_diag["trade_count"],
            "min_trades_required": tuning_diag["min_trades_required"],
            "dominant_action_pct": round(tuning_diag["dominant_action_pct"], 4),
            "flat_action_pct": round(tuning_diag["flat_action_pct"], 4),
            "collapse_flags": tuning_diag["collapse_flags"],
            "primary_score": round(aggregate_diag["primary_score"], 6),
            "holdout_mean": round(aggregate_diag["holdout_mean"], 6),
            "holdout_std": round(aggregate_diag["holdout_std"], 6),
            "best_baseline_score": round(aggregate_diag["best_baseline_score"], 6),
            "baseline_gap": round(aggregate_diag["baseline_gap"], 6),
            "baseline_penalty": round(aggregate_diag["baseline_penalty"], 6),
        })

        action_dist = format_action_distribution(tuning_diag["action_counts"])
        sortino_text = bold(f"{float(results.get('sortino_ratio', 0.0)):.2f}")
        log_msg = (
            f"{stage_name} Trial {trial.number}: composite={composite_score:.2f}, "
            f"return={color_pct(float(results.get('total_return_pct', 0.0)))}, "
            f"sortino={sortino_text}, "
            f"calmar={float(results.get('calmar_ratio', 0.0)):.2f}, "
            f"maxDD={abs(float(results.get('max_drawdown', 0.0))):.2f}%, "
            f"actions=[{action_dist}], lr={learning_rate:.6f}, n_steps={n_steps}, "
            f"ent_coef={ent_coef:.4f}, batch={batch_size}"
        )
        if tuning_diag["total_penalty"] > 0:
            log_msg += (
                f", base={tuning_diag['base_score']:.2f}, penalty={tuning_diag['total_penalty']:.2f}, "
                f"trades={tuning_diag['trade_count']}/{tuning_diag['min_trades_required']}, "
                f"flags={tuning_diag['collapse_flags']}"
            )
        if holdout_validation_sets:
            log_msg += (
                f", holdout_mean={aggregate_diag['holdout_mean']:.2f}, holdout_std={aggregate_diag['holdout_std']:.2f}"
            )
        if aggregate_diag["baseline_penalty"] > 0:
            log_msg += (
                f", baseline={aggregate_diag['best_baseline_score']:.2f}, "
                f"baseline_penalty={aggregate_diag['baseline_penalty']:.2f}"
            )
        log_msg += (
            f", gamma={gamma:.4f}, gae={gae_lambda:.4f}, "
            f"turnover={float(trial_overrides.get('reward_turnover_penalty', config['reward']['turnover_penalty'])):.4f}, "
            f"loss_mult={float(trial_overrides.get('reward_loss_multiplier', config['reward']['loss_multiplier'])):.3f}, "
            f"dd_pen={float(trial_overrides.get('reward_drawdown_penalty', config['reward']['drawdown_penalty'])):.3f}, "
            f"flat_pen={float(trial_overrides.get('reward_flat_time_penalty', config['reward']['flat_time_penalty'])):.4f}, "
            f"flat_grace={int(trial_overrides.get('reward_flat_time_grace_steps', config['reward']['flat_time_grace_steps']))}"
        )
        logger.info(log_msg)
        return composite_score

    if n_trials <= 0:
        return {"best_params": {}, "best_value": float("-inf"), "study": None}

    if use_parallel:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
    else:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    try:
        best_params = study.best_params
        best_value = study.best_value
    except ValueError:
        completed_trials = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned_trials = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.PRUNED
        ]
        failed_trials = [
            trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.FAIL
        ]
        logger.warning(
            "%s tuning produced no completed trials (pruned=%d, failed=%d). Returning empty result.",
            stage_name,
            len(pruned_trials),
            len(failed_trials),
        )
        best_params = {}
        best_value = float("-inf")

    return {
        "best_params": best_params,
        "best_value": best_value,
        "study": study,
    }


def _run_qrdqn_tuning(
    *,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    n_trials: int,
    window_folder: str | None,
    holdout_validation_sets: List[pd.DataFrame],
) -> Dict:
    """Run a minimal QR-DQN hyperparameter tuning study."""
    tuning_config = config.get("hyperparameter_tuning", {})
    parallel_config = tuning_config.get("parallel_processing", {})
    use_parallel = parallel_config.get("enabled", True)
    n_jobs = _resolve_tuning_n_jobs()
    if not use_parallel:
        n_jobs = 1

    reference_columns = list(train_data.columns)
    evaluation_sets = [validation_data] + list(holdout_validation_sets or [])
    baseline_margin = float(tuning_config.get("baseline_margin", 0.25))
    baseline_scores = _score_constant_action_baselines(evaluation_sets, reference_columns)
    logger.info(
        "QR-DQN trivial baseline scores: %s",
        [round(max(scores.values()), 2) for scores in baseline_scores],
    )

    hp_config = _resolve_qrdqn_hp_config(tuning_config)
    tuning_timesteps = int(
        tuning_config.get(
            "qrdqn_tuning_timesteps",
            tuning_config.get("tuning_timesteps", tuning_config.get("stage1_timesteps", 20000)),
        )
    )

    study_name = f"qrdqn_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    sampler = optuna.samplers.TPESampler(
        seed=config.get("seed", 42),
        multivariate=True,
        n_startup_trials=min(10, max(5, n_trials // 3)),
        n_ei_candidates=24,
    )

    storage = None
    if use_parallel:
        if window_folder:
            storage = f"sqlite:///{window_folder}/qrdqn_optuna.db"
        else:
            storage = "sqlite:///qrdqn_optuna.db"

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial):
        sampled = _sample_qrdqn_params(trial, hp_config)
        learning_rate = float(sampled["learning_rate"])
        batch_size = int(sampled["batch_size"])
        gamma = float(sampled["gamma"])
        buffer_size = int(sampled["buffer_size"])
        learning_starts = int(sampled["learning_starts"])
        train_freq = int(sampled["train_freq"])
        gradient_steps = int(sampled["gradient_steps"])
        target_update_interval = int(sampled["target_update_interval"])
        exploration_fraction = float(sampled["exploration_fraction"])
        exploration_final_eps = float(sampled["exploration_final_eps"])
        trial_overrides = {}
        for key in (
            "reward_turnover_penalty",
            "reward_loss_multiplier",
            "reward_drawdown_penalty",
            "reward_drawdown_penalty_threshold",
            "reward_flat_time_penalty",
            "reward_flat_time_grace_steps",
            "synthetic_oversample_ratio",
        ):
            if key in sampled:
                trial_overrides[key] = sampled[key]
        original_overrides = _apply_trial_config_overrides(trial_overrides)
        device = get_device(config.get("sequence_model", {}).get("device", "auto"), for_recurrent=False)
        train_env, _ = _build_tuning_env(train_data, random_start_pct=float(config.get("training", {}).get("random_start_pct", 0.2)), min_episode_steps=2, n_envs_override=1)

        model = train_module.QRDQN(
            "MlpPolicy",
            train_env,
            verbose=0,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gamma=gamma,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=float(config.get("model", {}).get("exploration_initial_eps", 1.0)),
            exploration_final_eps=exploration_final_eps,
            seed=config.get("seed"),
            device=device,
            policy_kwargs={"net_arch": [256, 256]},
        )

        try:
            model.learn(total_timesteps=tuning_timesteps, progress_bar=False)
            evaluation_scores = []
            evaluation_results = []
            for dataset in evaluation_sets:
                aligned_dataset = _align_feature_columns(reference_columns, dataset)
                results = evaluate_agent(model, aligned_dataset, verbose=0, deterministic=True)
                score, diagnostics = _score_tuning_trial(results, len(aligned_dataset))
                evaluation_results.append((results, diagnostics))
                evaluation_scores.append(score)
            composite_score, aggregate_diag = _summarize_trial_scores(
                evaluation_scores,
                baseline_scores,
                baseline_margin=baseline_margin,
            )
        except (ValueError, RuntimeError) as e:
            logger.warning("qrdqn trial %d failed during training: %s", trial.number, e)
            return -100.0
        finally:
            train_env.close()
            _restore_trial_config_overrides(original_overrides)

        results, tuning_diag = evaluation_results[0]
        trial.set_user_attr(
            "tuning_diagnostics",
            {
                "stage": "qrdqn",
                "base_score": round(tuning_diag["base_score"], 6),
                "total_penalty": round(tuning_diag["total_penalty"], 6),
                "trade_count": tuning_diag["trade_count"],
                "min_trades_required": tuning_diag["min_trades_required"],
                "dominant_action_pct": round(tuning_diag["dominant_action_pct"], 4),
                "flat_action_pct": round(tuning_diag["flat_action_pct"], 4),
                "collapse_flags": tuning_diag["collapse_flags"],
                "primary_score": round(aggregate_diag["primary_score"], 6),
                "holdout_mean": round(aggregate_diag["holdout_mean"], 6),
                "holdout_std": round(aggregate_diag["holdout_std"], 6),
                "best_baseline_score": round(aggregate_diag["best_baseline_score"], 6),
                "baseline_gap": round(aggregate_diag["baseline_gap"], 6),
                "baseline_penalty": round(aggregate_diag["baseline_penalty"], 6),
            },
        )

        action_dist = format_action_distribution(tuning_diag["action_counts"])
        sortino_text = bold(f"{float(results.get('sortino_ratio', 0.0)):.2f}")
        log_msg = (
            f"qrdqn Trial {trial.number}: composite={composite_score:.2f}, "
            f"return={color_pct(float(results.get('total_return_pct', 0.0)))}, "
            f"sortino={sortino_text}, "
            f"calmar={float(results.get('calmar_ratio', 0.0)):.2f}, "
            f"maxDD={abs(float(results.get('max_drawdown', 0.0))):.2f}%, "
            f"actions=[{action_dist}], lr={learning_rate:.6f}, batch={batch_size}, gamma={gamma:.4f}, "
            f"buffer={buffer_size}, starts={learning_starts}, train_freq={train_freq}, "
            f"grad_steps={gradient_steps}, target_update={target_update_interval}, "
            f"eps_frac={exploration_fraction:.3f}, eps_final={exploration_final_eps:.3f}"
        )
        if aggregate_diag["holdout_std"] > 0 or len(evaluation_sets) > 1:
            log_msg += (
                f", holdout_mean={aggregate_diag['holdout_mean']:.2f}, "
                f"holdout_std={aggregate_diag['holdout_std']:.2f}"
            )
        if aggregate_diag["baseline_penalty"] > 0:
            log_msg += (
                f", baseline={aggregate_diag['best_baseline_score']:.2f}, "
                f"baseline_penalty={aggregate_diag['baseline_penalty']:.2f}"
            )
        log_msg += (
            f", turnover={float(trial_overrides.get('reward_turnover_penalty', config['reward']['turnover_penalty'])):.4f}, "
            f"loss_mult={float(trial_overrides.get('reward_loss_multiplier', config['reward']['loss_multiplier'])):.3f}, "
            f"dd_pen={float(trial_overrides.get('reward_drawdown_penalty', config['reward']['drawdown_penalty'])):.3f}, "
            f"flat_pen={float(trial_overrides.get('reward_flat_time_penalty', config['reward']['flat_time_penalty'])):.4f}, "
            f"flat_grace={int(trial_overrides.get('reward_flat_time_grace_steps', config['reward']['flat_time_grace_steps']))}"
        )
        if "synthetic_oversample_ratio" in trial_overrides:
            log_msg += f", synth={float(trial_overrides['synthetic_oversample_ratio']):.3f}"
        logger.info(log_msg)
        return float(composite_score)

    if n_trials <= 0:
        return {
            "best_params": {},
            "best_value": float("-inf"),
            "selected_stage": "qrdqn",
            "stage1_best_params": {},
            "stage1_best_value": float("-inf"),
            "stage2_best_params": {},
            "stage2_best_value": float("-inf"),
            "stage2_search_space": {},
            "study": None,
        }

    if use_parallel:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
    else:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    try:
        best_params = study.best_params
        best_value = study.best_value
    except ValueError:
        logger.warning("QR-DQN tuning produced no completed trials. Returning empty result.")
        best_params = {}
        best_value = float("-inf")

    return {
        "best_params": best_params,
        "best_value": float(best_value),
        "selected_stage": "qrdqn",
        "stage1_best_params": dict(best_params),
        "stage1_best_value": float(best_value),
        "stage2_best_params": {},
        "stage2_best_value": float("-inf"),
        "stage2_search_space": {},
        "study": study,
    }


def _score_tuning_trial(results: Dict, validation_bars: int) -> Tuple[float, Dict]:
    """Score a tuning trial and penalize degenerate one-sided policies."""
    return_pct = float(results.get("total_return_pct", 0.0))
    calmar = float(results.get("calmar_ratio", 0.0))
    sortino = float(results.get("sortino_ratio", 0.0))
    max_dd = abs(float(results.get("max_drawdown", 0.0)))
    max_allowed_drawdown_pct = float(config.get("training", {}).get("max_allowed_drawdown_pct", 40.0))
    trade_count = int(
        results.get(
            "economic_trade_count",
            results.get("completed_trades", results.get("trade_count", results.get("num_trades", 0))),
        )
    )
    action_counts = _normalize_action_counts(results.get("action_counts"))
    total_actions = max(1, sum(action_counts.values()))
    long_action_pct = 100.0 * sum(action_counts.get(i, 0) for i in LONG_ACTIONS) / total_actions
    short_action_pct = 100.0 * sum(action_counts.get(i, 0) for i in SHORT_ACTIONS) / total_actions
    flat_action_pct = 100.0 * action_counts.get(FLAT_ACTION, 0) / total_actions
    dominant_action_pct = max(long_action_pct, short_action_pct, flat_action_pct)
    active_actions = sum(1 for count in action_counts.values() if count > 0)
    min_trades = max(5, validation_bars // 250)

    base_score = (calmar * 0.45) + (sortino * 0.30) + (return_pct * 0.15) - (max_dd * 0.40)

    penalties = []
    collapse_flags = []
    if trade_count < min_trades:
        shortfall_ratio = (min_trades - trade_count) / max(1, min_trades)
        penalty = 3.0 + (3.0 * shortfall_ratio)
        penalties.append(("too_few_trades", penalty))
        collapse_flags.append("too_few_trades")
    if dominant_action_pct >= 90.0:
        penalty = ((dominant_action_pct - 90.0) / 10.0) * 3.0
        penalties.append(("dominant_action", penalty))
        collapse_flags.append("dominant_action")
    if active_actions <= 1:
        penalties.append(("single_action_policy", 3.0))
        collapse_flags.append("single_action_policy")
    if short_action_pct >= 95.0:
        penalties.append(("always_short", 3.0))
        collapse_flags.append("always_short")
    if long_action_pct >= 95.0:
        penalties.append(("always_long", 3.0))
        collapse_flags.append("always_long")
    if flat_action_pct >= 95.0:
        penalties.append(("always_flat", 3.0))
        collapse_flags.append("always_flat")
    if max_dd > max_allowed_drawdown_pct:
        penalty = 6.0 + max(0.0, max_dd - max_allowed_drawdown_pct) * 0.2
        penalties.append(("excessive_drawdown", penalty))
        collapse_flags.append("excessive_drawdown")

    total_penalty = float(sum(penalty for _, penalty in penalties))
    composite_score = base_score - total_penalty
    diagnostics = {
        "base_score": base_score,
        "total_penalty": total_penalty,
        "penalties": penalties,
        "collapse_flags": collapse_flags,
        "trade_count": trade_count,
        "min_trades_required": min_trades,
        "action_counts": action_counts,
        "long_action_pct": long_action_pct,
        "short_action_pct": short_action_pct,
        "flat_action_pct": flat_action_pct,
        "dominant_action_pct": dominant_action_pct,
        "active_action_count": active_actions,
        "max_allowed_drawdown_pct": max_allowed_drawdown_pct,
    }
    return composite_score, diagnostics


def _apply_trial_config_overrides(trial_overrides: Dict[str, float]) -> Dict[str, object]:
    """Apply per-trial reward/augmentation overrides and return originals."""
    originals: Dict[str, object] = {}
    reward_cfg = config.setdefault("reward", {})
    if "reward_loss_multiplier" in trial_overrides:
        originals["reward_loss_multiplier"] = reward_cfg.get("loss_multiplier")
        reward_cfg["loss_multiplier"] = float(trial_overrides["reward_loss_multiplier"])
    if "reward_turnover_penalty" in trial_overrides:
        originals["reward_turnover_penalty"] = reward_cfg.get("turnover_penalty")
        reward_cfg["turnover_penalty"] = float(trial_overrides["reward_turnover_penalty"])
    if "reward_drawdown_penalty" in trial_overrides:
        originals["reward_drawdown_penalty"] = reward_cfg.get("drawdown_penalty")
        reward_cfg["drawdown_penalty"] = float(trial_overrides["reward_drawdown_penalty"])
    if "reward_drawdown_penalty_threshold" in trial_overrides:
        originals["reward_drawdown_penalty_threshold"] = reward_cfg.get("drawdown_penalty_threshold")
        reward_cfg["drawdown_penalty_threshold"] = float(trial_overrides["reward_drawdown_penalty_threshold"])
    if "reward_flat_time_penalty" in trial_overrides:
        originals["reward_flat_time_penalty"] = reward_cfg.get("flat_time_penalty")
        reward_cfg["flat_time_penalty"] = float(trial_overrides["reward_flat_time_penalty"])
    if "reward_flat_time_grace_steps" in trial_overrides:
        originals["reward_flat_time_grace_steps"] = reward_cfg.get("flat_time_grace_steps")
        reward_cfg["flat_time_grace_steps"] = int(trial_overrides["reward_flat_time_grace_steps"])

    aug_cfg = config.setdefault("augmentation", {}).setdefault("synthetic_bears", {})
    if "synthetic_oversample_ratio" in trial_overrides:
        originals["synthetic_oversample_ratio"] = aug_cfg.get("oversample_ratio")
        aug_cfg["oversample_ratio"] = float(trial_overrides["synthetic_oversample_ratio"])

    return originals


def _restore_trial_config_overrides(originals: Dict[str, object]) -> None:
    """Restore config values that were temporarily overridden for a trial."""
    reward_cfg = config.setdefault("reward", {})
    if "reward_loss_multiplier" in originals:
        reward_cfg["loss_multiplier"] = originals["reward_loss_multiplier"]
    if "reward_turnover_penalty" in originals:
        reward_cfg["turnover_penalty"] = originals["reward_turnover_penalty"]
    if "reward_drawdown_penalty" in originals:
        reward_cfg["drawdown_penalty"] = originals["reward_drawdown_penalty"]
    if "reward_drawdown_penalty_threshold" in originals:
        reward_cfg["drawdown_penalty_threshold"] = originals["reward_drawdown_penalty_threshold"]
    if "reward_flat_time_penalty" in originals:
        reward_cfg["flat_time_penalty"] = originals["reward_flat_time_penalty"]
    if "reward_flat_time_grace_steps" in originals:
        reward_cfg["flat_time_grace_steps"] = originals["reward_flat_time_grace_steps"]

    aug_cfg = config.setdefault("augmentation", {}).setdefault("synthetic_bears", {})
    if "synthetic_oversample_ratio" in originals:
        aug_cfg["oversample_ratio"] = originals["synthetic_oversample_ratio"]


def _apply_best_hyperparameters_to_config(best_hyperparameters: Dict) -> None:
    """Apply tuned reward/augmentation/model params to the in-memory config."""
    reward_cfg = config.setdefault("reward", {})
    model_cfg = config.setdefault("model", {})
    aug_cfg = config.setdefault("augmentation", {}).setdefault("synthetic_bears", {})
    if "learning_rate" in best_hyperparameters:
        model_cfg["learning_rate"] = float(best_hyperparameters["learning_rate"])
    if "batch_size" in best_hyperparameters:
        model_cfg["batch_size"] = int(best_hyperparameters["batch_size"])
    if "buffer_size" in best_hyperparameters:
        model_cfg["buffer_size"] = int(best_hyperparameters["buffer_size"])
    if "learning_starts" in best_hyperparameters:
        model_cfg["learning_starts"] = int(best_hyperparameters["learning_starts"])
    if "train_freq" in best_hyperparameters:
        model_cfg["train_freq"] = int(best_hyperparameters["train_freq"])
    if "gradient_steps" in best_hyperparameters:
        model_cfg["gradient_steps"] = int(best_hyperparameters["gradient_steps"])
    if "target_update_interval" in best_hyperparameters:
        model_cfg["target_update_interval"] = int(best_hyperparameters["target_update_interval"])
    if "exploration_fraction" in best_hyperparameters:
        model_cfg["exploration_fraction"] = float(best_hyperparameters["exploration_fraction"])
    if "exploration_final_eps" in best_hyperparameters:
        model_cfg["exploration_final_eps"] = float(best_hyperparameters["exploration_final_eps"])
    if "n_steps" in best_hyperparameters:
        model_cfg["n_steps"] = int(best_hyperparameters["n_steps"])
    if "ent_coef" in best_hyperparameters:
        model_cfg["ent_coef"] = float(best_hyperparameters["ent_coef"])
    if "reward_loss_multiplier" in best_hyperparameters:
        reward_cfg["loss_multiplier"] = float(best_hyperparameters["reward_loss_multiplier"])
    if "reward_turnover_penalty" in best_hyperparameters:
        reward_cfg["turnover_penalty"] = float(best_hyperparameters["reward_turnover_penalty"])
    if "reward_drawdown_penalty" in best_hyperparameters:
        reward_cfg["drawdown_penalty"] = float(best_hyperparameters["reward_drawdown_penalty"])
    if "reward_drawdown_penalty_threshold" in best_hyperparameters:
        reward_cfg["drawdown_penalty_threshold"] = float(best_hyperparameters["reward_drawdown_penalty_threshold"])
    if "reward_flat_time_penalty" in best_hyperparameters:
        reward_cfg["flat_time_penalty"] = float(best_hyperparameters["reward_flat_time_penalty"])
    if "reward_flat_time_grace_steps" in best_hyperparameters:
        reward_cfg["flat_time_grace_steps"] = int(best_hyperparameters["reward_flat_time_grace_steps"])
    if "synthetic_oversample_ratio" in best_hyperparameters:
        aug_cfg["oversample_ratio"] = float(best_hyperparameters["synthetic_oversample_ratio"])
    if "gamma" in best_hyperparameters:
        model_cfg["gamma"] = float(best_hyperparameters["gamma"])
    if "gae_lambda" in best_hyperparameters:
        model_cfg["gae_lambda"] = float(best_hyperparameters["gae_lambda"])


def _candidate_trials_from_study(study, top_k: int) -> List[optuna.trial.FrozenTrial]:
    """Return the strongest completed Optuna trials in descending score order."""
    if study is None or top_k <= 0:
        return []
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda trial: float(trial.value), reverse=True)
    return completed[:top_k]


def _evaluate_finalist_candidates(
    *,
    candidates: List[Dict],
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    holdout_validation_sets: List[pd.DataFrame],
    baseline_scores: List[Dict[int, float]],
    baseline_margin: float,
    finalist_timesteps: int,
    finalist_num_seeds: int,
) -> Dict | None:
    """Re-score finalist parameter sets across multiple seeds and a larger budget."""
    if not candidates or finalist_num_seeds <= 0 or finalist_timesteps <= 0:
        return None

    training_cfg = config.get("training", {})
    original_seed = config.get("seed", 42)
    evaluation_sets = [validation_data] + holdout_validation_sets
    finalists = []

    for candidate in candidates:
        params = dict(candidate.get("params", {}))
        seed_scores = []
        worst_seed_drawdown = float("-inf")
        for seed_offset in range(finalist_num_seeds):
            seed_value = int(original_seed) + seed_offset
            config["seed"] = seed_value
            overrides = _apply_trial_config_overrides(params)
            try:
                model, _ = train_walk_forward_model(
                    train_data=train_data,
                    validation_data=validation_data,
                    initial_timesteps=finalist_timesteps,
                    additional_timesteps=0,
                    max_iterations=1,
                    n_stagnant_loops=1,
                    improvement_threshold=0.0,
                    window_folder=None,
                    run_hyperparameter_tuning=False,
                    model_params=params,
                )
                evaluation_scores = []
                seed_drawdowns = []
                for dataset in evaluation_sets:
                    results = evaluate_agent(model, dataset, verbose=0, deterministic=True)
                    score, _ = _score_tuning_trial(results, len(dataset))
                    evaluation_scores.append(score)
                    seed_drawdowns.append(abs(float(results.get("max_drawdown", 0.0))))
                composite_score, aggregate_diag = _summarize_trial_scores(
                    evaluation_scores,
                    baseline_scores,
                    baseline_margin=baseline_margin,
                )
                seed_scores.append(composite_score)
                worst_seed_drawdown = max(worst_seed_drawdown, max(seed_drawdowns) if seed_drawdowns else 0.0)
            finally:
                _restore_trial_config_overrides(overrides)
        if seed_scores:
            finalists.append(
                {
                    "params": params,
                    "median_score": float(np.median(seed_scores)),
                    "mean_score": float(np.mean(seed_scores)),
                    "seed_scores": [float(score) for score in seed_scores],
                    "worst_drawdown": float(worst_seed_drawdown if worst_seed_drawdown != float("-inf") else 0.0),
                }
            )

    config["seed"] = original_seed
    if not finalists:
        return None

    finalists.sort(key=lambda item: (item["median_score"], item["mean_score"], -item["worst_drawdown"]), reverse=True)
    return finalists[0]


def _align_feature_columns(reference_columns: List[str], df: pd.DataFrame) -> pd.DataFrame:
    """Align a dataframe to a model's expected feature columns."""
    return df.reindex(columns=reference_columns, fill_value=0.0).copy()


def _score_constant_action_baselines(
    evaluation_sets: List[pd.DataFrame],
    reference_columns: List[str],
) -> List[Dict[str, float]]:
    """Score trivial constant policies across the full 7-action space."""
    baseline_scores = []
    for dataset in evaluation_sets:
        aligned = _align_feature_columns(reference_columns, dataset)
        action_scores = {}
        for action in range(ACTION_COUNT):
            try:
                results = evaluate_agent(_ConstantActionModel(action), aligned, verbose=0, deterministic=True)
            except Exception as exc:
                logger.warning(f"Constant-action baseline {action_label(action)} failed: {exc}")
                results = {
                    "total_return_pct": -100.0,
                    "sortino_ratio": -10.0,
                    "calmar_ratio": -1.0,
                    "max_drawdown": -100.0,
                    "trade_count": 0,
                    "action_counts": {action: len(aligned)},
                }
            score, _ = _score_tuning_trial(results, len(aligned))
            action_scores[action] = score
        baseline_scores.append(action_scores)
    return baseline_scores


def _summarize_trial_scores(
    evaluation_scores: List[float],
    baseline_scores: List[Dict[int, float]],
    *,
    baseline_margin: float,
) -> Tuple[float, Dict]:
    """Aggregate per-dataset scores and penalize trials that barely beat trivial baselines."""
    primary_score = float(evaluation_scores[0])
    holdout_scores = [float(score) for score in evaluation_scores[1:]]
    holdout_mean = float(np.mean(holdout_scores)) if holdout_scores else primary_score
    holdout_std = float(np.std(holdout_scores)) if holdout_scores else 0.0
    aggregate_score = (primary_score * 0.7) + (holdout_mean * 0.3) - (0.2 * holdout_std)

    per_dataset_best_baseline = [max(scores.values()) for scores in baseline_scores]
    best_baseline_score = float(np.mean(per_dataset_best_baseline)) if per_dataset_best_baseline else 0.0
    baseline_gap = aggregate_score - best_baseline_score
    baseline_penalty = max(0.0, baseline_margin - baseline_gap)
    final_score = aggregate_score - baseline_penalty
    diagnostics = {
        "primary_score": primary_score,
        "holdout_mean": holdout_mean,
        "holdout_std": holdout_std,
        "aggregate_score": aggregate_score,
        "best_baseline_score": best_baseline_score,
        "baseline_gap": baseline_gap,
        "baseline_penalty": baseline_penalty,
    }
    return final_score, diagnostics


def _resolve_close_column(df: pd.DataFrame) -> str:
    """Return the close-like price column used by a dataframe."""
    for candidate in ("Close", "CLOSE", "close"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No close column found in dataframe")


def _sanitize_ohlc_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Replace implausible OHLC outliers with nearby valid prices.

    Some vendor files contain positive but obviously bad crypto prints
    (for example BTC bars at 145.0 in an otherwise 25k-30k regime). Those
    values do not trip simple non-finite/<=0 guards, but they blow up
    contract sizing and downstream portfolio accounting. Clean them before
    feature generation so both observations and valuation use the same price path.
    """
    if df.empty:
        return df

    close_col = _resolve_close_column(df)
    sanitized = df.copy()
    close_series = pd.to_numeric(sanitized[close_col], errors="coerce")
    finite_positive = close_series[np.isfinite(close_series) & (close_series > 0)]
    if finite_positive.empty:
        return sanitized

    median_price = float(finite_positive.median())
    if not np.isfinite(median_price) or median_price <= 0:
        return sanitized

    low_cutoff = median_price * 0.1
    high_cutoff = median_price * 10.0

    def _clean_price_series(series: pd.Series, *, replacement: pd.Series | None = None) -> tuple[pd.Series, int]:
        numeric = pd.to_numeric(series, errors="coerce")
        invalid_mask = (~np.isfinite(numeric)) | (numeric <= 0) | (numeric < low_cutoff) | (numeric > high_cutoff)
        invalid_count = int(invalid_mask.sum())
        if not invalid_count:
            return numeric, 0

        cleaned = numeric.copy()
        cleaned.loc[invalid_mask] = np.nan
        cleaned = cleaned.ffill().bfill()
        if replacement is not None:
            cleaned.loc[cleaned.isna()] = replacement.loc[cleaned.isna()]
        cleaned = cleaned.fillna(median_price)
        return cleaned.astype(float), invalid_count

    clean_close, invalid_close = _clean_price_series(close_series)
    if invalid_close:
        logger.warning(
            "Sanitized %d implausible '%s' values outside [%s, %s]",
            invalid_close,
            close_col,
            round(low_cutoff, 2),
            round(high_cutoff, 2),
        )
    sanitized[close_col] = clean_close

    ohlc_variants = {
        close_col,
        "open", "Open", "OPEN",
        "high", "High", "HIGH",
        "low", "Low", "LOW",
    }
    for col in [c for c in ohlc_variants if c in sanitized.columns and c != close_col]:
        clean_col, invalid_count = _clean_price_series(sanitized[col], replacement=clean_close)
        if invalid_count:
            logger.warning(
                "Sanitized %d implausible '%s' values outside [%s, %s]",
                invalid_count,
                col,
                round(low_cutoff, 2),
                round(high_cutoff, 2),
            )
        sanitized[col] = clean_col

    return sanitized


def _recompute_clean_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild indicators from cleaned raw OHLCV for a window slice."""
    try:
        raw = extract_ohlcv_frame(df)
    except KeyError:
        logger.warning("Window slice missing raw OHLCV columns; using provided frame without indicator recompute")
        return df.copy()
    raw = _sanitize_ohlc_outliers(raw)
    return process_technical_indicators(raw)


def build_window_report_payload(
    *,
    window_idx: int,
    test_data: pd.DataFrame,
    test_results: Dict,
    training_stats: Dict,
    validation_results: Dict,
    window_periods: Dict,
    evaluation_metric: str,
) -> Dict:
    """Build a report-grade window payload for HTML report generation."""
    close_col = _resolve_close_column(test_data)
    test_timestamps = list(test_data.index)
    test_prices = [float(v) for v in test_data[close_col].tolist()]

    raw_dates = test_results.get("dates") or test_timestamps
    raw_price_history = test_results.get("price_history") or test_prices
    raw_portfolio_history = [float(v) for v in test_results.get("portfolio_history", [])]
    raw_action_history = [int(v) for v in test_results.get("action_history", [])]
    raw_position_history = [int(v) for v in test_results.get("position_history", [])]
    raw_drawdown_history = [float(v) for v in test_results.get("drawdown_history", [])]

    if not raw_dates:
        raw_dates = test_timestamps
    if not raw_price_history:
        raw_price_history = test_prices

    timestamps = [pd.Timestamp(ts) for ts in raw_dates]
    price_history = [float(v) for v in raw_price_history]
    portfolio_timestamps = list(timestamps)
    portfolio_history = raw_portfolio_history
    if len(portfolio_history) == len(timestamps) + 1 and timestamps:
        portfolio_timestamps = [timestamps[0]] + list(timestamps)
    elif len(portfolio_history) != len(timestamps):
        limit = min(len(portfolio_history), len(timestamps))
        portfolio_history = portfolio_history[:limit]
        portfolio_timestamps = portfolio_timestamps[:limit]

    if raw_position_history:
        position_timestamps = list(portfolio_timestamps)
        position_history = raw_position_history
        if len(position_history) != len(position_timestamps):
            limit = min(len(position_history), len(position_timestamps))
            position_history = position_history[:limit]
            position_timestamps = position_timestamps[:limit]
    else:
        position_history = []
        position_timestamps = []

    if raw_drawdown_history:
        drawdown_timestamps = list(portfolio_timestamps)
        drawdown_history = raw_drawdown_history
        if len(drawdown_history) != len(drawdown_timestamps):
            limit = min(len(drawdown_history), len(drawdown_timestamps))
            drawdown_history = drawdown_history[:limit]
            drawdown_timestamps = drawdown_timestamps[:limit]
    else:
        drawdown_history = []
        drawdown_timestamps = []

    training_iterations = training_stats.get("iterations", training_stats)
    best_iteration = None
    for iteration in training_iterations:
        if iteration.get("is_best"):
            best_iteration = iteration.get("iteration")
            break

    return {
        "window": window_idx,
        "evaluation_metric": evaluation_metric,
        "metrics": {
            "return_pct": float(test_results.get("total_return_pct", 0.0)),
            "final_portfolio_value": float(test_results.get("final_portfolio_value", 0.0)),
            "trade_count": int(test_results.get("trade_count", 0)),
            "rebalance_count": int(test_results.get("rebalance_count", test_results.get("trade_count", 0))),
            "completed_trades": int(
                test_results.get("completed_trades", test_results.get("economic_trade_count", 0))
            ),
            "economic_trade_count": int(
                test_results.get("economic_trade_count", test_results.get("completed_trades", 0))
            ),
            "hit_rate": float(test_results.get("hit_rate", 0.0)),
            "profitable_trades": int(test_results.get("profitable_trades", 0)),
            "prediction_accuracy": float(test_results.get("prediction_accuracy", 0.0)),
            "correct_predictions": int(test_results.get("correct_predictions", 0)),
            "total_predictions": int(test_results.get("total_predictions", 0)),
            "max_drawdown": float(test_results.get("max_drawdown", 0.0)),
            "calmar_ratio": float(test_results.get("calmar_ratio", 0.0)),
            "sortino_ratio": float(test_results.get("sortino_ratio", 0.0)),
            "final_position": int(test_results.get("final_position", 0)),
        },
        "window_periods": window_periods,
        "series": {
            "timestamps": timestamps,
            "prices": price_history,
            "portfolio_timestamps": portfolio_timestamps,
            "portfolio_values": portfolio_history,
            "position_timestamps": position_timestamps,
            "position_values": position_history,
            "drawdown_timestamps": drawdown_timestamps,
            "drawdown_values": drawdown_history,
            "action_values": raw_action_history,
        },
        "trade_history": test_results.get("trade_history", []),
        "action_counts": test_results.get("action_counts", {}),
        "training_iterations": training_iterations,
        "validation_results": validation_results,
        "best_iteration": best_iteration,
    }


def _augment_training_slice(train_data: pd.DataFrame, window_folder: str | None = None) -> Tuple[pd.DataFrame, Dict | None]:
    """Augment raw OHLCV and recompute indicators for the train slice."""
    aug_config = config.get("augmentation", {}).get("synthetic_bears", {})
    try:
        raw_train = extract_ohlcv_frame(train_data)
    except KeyError:
        logger.warning("Training slice missing raw OHLCV columns; skipping augmentation/recompute fallback")
        return train_data.copy(), None
    raw_train = _sanitize_ohlc_outliers(raw_train)
    if not aug_config.get("enabled", False):
        return process_technical_indicators(raw_train), None

    augmented_raw, metadata = augment_with_synthetic_bears(
        raw_train,
        oversample_ratio=aug_config.get("oversample_ratio", 0.3),
        segment_length_pct=aug_config.get("segment_length_pct", 0.15),
        seed=int(config.get("seed", 42)),
        return_metadata=True,
    )
    augmented_raw = _sanitize_ohlc_outliers(augmented_raw)
    recomputed_train = process_technical_indicators(augmented_raw)

    if metadata and window_folder:
        save_json(metadata, f"{window_folder}/synthetic_bears_report.json")

    logger.info(
        f"Augmented train data to {len(recomputed_train)} rows with synthetic bears "
        f"before LSTM/normalization"
    )
    return recomputed_train, metadata


def _drop_unused_model_columns(datasets: List[pd.DataFrame]) -> None:
    """Drop redundant raw columns after feature generation/scaling."""
    risk_cfg = config.get("risk_management", {})
    needs_ohlc = (
        risk_cfg.get("enabled", False)
        or risk_cfg.get("dynamic_sl_tp", {}).get("enabled", False)
        or risk_cfg.get("stop_loss", {}).get("enabled", False)
        or risk_cfg.get("take_profit", {}).get("enabled", False)
        or risk_cfg.get("trailing_stop", {}).get("enabled", False)
    )
    cols_to_drop = ['volume', 'Volume', 'SMA', 'EMA', 'VWAP', 'PSAR', 'OBV', 'VOLUME_NORM', 'DOW', 'position']
    if not needs_ohlc:
        cols_to_drop.extend(['open', 'Open', 'OPEN', 'high', 'low', 'High', 'Low', 'HIGH', 'LOW'])
    for df in datasets:
        cols_present = [c for c in cols_to_drop if c in df.columns]
        if cols_present:
            df.drop(columns=cols_present, inplace=True)


def _current_lstm_tuning_scope() -> str:
    """Return whether LSTM tuning happens once per session or per window."""
    lstm_tuning_cfg = config.get("indicators", {}).get("lstm_features", {}).get("tuning", {})
    scope = str(lstm_tuning_cfg.get("scope", "session")).lower()
    if scope not in {"session", "per_window"}:
        return "session"
    return scope


def _prepare_scaled_window_inputs(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame | None = None,
    *,
    window_folder: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, Dict | None]:
    """Apply augmentation/recompute/scaling before any VAE or PPO fitting."""
    prepared_train, synth_metadata = _augment_training_slice(train_data.copy(), window_folder)
    prepared_val = _recompute_clean_slice(validation_data.copy())
    prepared_test = _recompute_clean_slice(test_data.copy()) if test_data is not None else None

    scaling_test_frame = prepared_test.copy() if prepared_test is not None else prepared_val.copy()
    scaling_cols = get_standardized_column_names(prepared_train)
    scaler_type = config.get("normalization", {}).get("scaler_type", "robust")
    _, prepared_train, prepared_val, scaled_test = scale_window(
        train_data=prepared_train,
        val_data=prepared_val,
        test_data=scaling_test_frame,
        cols_to_scale=scaling_cols,
        feature_range=(-1, 1),
        window_folder=window_folder,
        scaler_type=scaler_type,
    )
    if prepared_test is not None:
        prepared_test = scaled_test
    return prepared_train, prepared_val, prepared_test, synth_metadata


def _attach_latent_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    *,
    lstm_params: Dict,
    feature_columns: List[str],
    checkpoint_path: str | None = None,
    save_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, List[str]]:
    """Fit a VAE on the canonical market features and attach LATENT_F* columns."""
    lstm_generator = LSTMFeatureGenerator(
        lookback=lstm_params["lookback"],
        hidden_size=lstm_params["hidden_size"],
        num_layers=lstm_params["num_layers"],
        output_size=lstm_params["output_size"],
        feature_columns=feature_columns,
        beta=lstm_params.get("beta", 0.001),
        kl_warmup_epochs=lstm_params.get("kl_warmup_epochs", 10),
        pretrain_epochs=lstm_params.get("pretrain_epochs", 50),
        pretrain_lr=lstm_params.get("pretrain_lr", 0.001),
        pretrain_batch_size=lstm_params.get("pretrain_batch_size", 64),
        pretrain_patience=lstm_params.get("pretrain_patience", 10),
        pretrain_min_delta=lstm_params.get("pretrain_min_delta", 0.0001),
    )
    lstm_generator.fit(
        train_df=train_df[feature_columns],
        validation_df=validation_df[feature_columns],
        checkpoint_path=checkpoint_path,
    )

    datasets = [train_df, validation_df] + ([test_df] if test_df is not None else [])
    latent_columns = [f"LATENT_F{i}" for i in range(lstm_generator.output_size)]
    for dataset in datasets:
        latent_frame = lstm_generator.transform(dataset[feature_columns].copy())
        for col in latent_columns:
            dataset[col] = latent_frame[col]

    if save_path:
        lstm_generator.save(save_path)

    return train_df, validation_df, test_df, latent_columns


def _apply_representation_mode(
    datasets: List[pd.DataFrame],
    *,
    market_feature_columns: List[str],
    latent_columns: List[str],
) -> None:
    """Drop columns according to the configured representation mode."""
    mode = get_representation_mode()
    if mode == "engineered_only":
        drop_columns = latent_columns
    elif mode == "latent_only":
        drop_columns = market_feature_columns
    else:
        drop_columns = []

    if not drop_columns:
        return

    for dataset in datasets:
        cols_present = [col for col in drop_columns if col in dataset.columns]
        if cols_present:
            dataset.drop(columns=cols_present, inplace=True)


def _prepare_tuning_inputs(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    *,
    window_folder: str | None = None,
    lstm_params: Dict | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict | None]:
    """Prepare train/validation inputs for PPO tuning using the window pipeline."""
    prepared_train, prepared_val, _, synth_metadata = _prepare_scaled_window_inputs(
        train_data,
        validation_data,
        window_folder=window_folder,
    )
    market_feature_columns = get_market_feature_columns(prepared_train, representation_mode="hybrid")
    latent_columns: List[str] = []

    if lstm_params is not None and get_representation_mode() != "engineered_only" and market_feature_columns:
        checkpoint_path = f"{window_folder}/lstm_autoencoder_tuning_checkpoint.pt" if window_folder else None
        prepared_train, prepared_val, _, latent_columns = _attach_latent_features(
            prepared_train,
            prepared_val,
            None,
            lstm_params=lstm_params,
            feature_columns=market_feature_columns,
            checkpoint_path=checkpoint_path,
            save_path=None,
        )

    _apply_representation_mode(
        [prepared_train, prepared_val],
        market_feature_columns=market_feature_columns,
        latent_columns=latent_columns,
    )
    _drop_unused_model_columns([prepared_train, prepared_val])
    return prepared_train, prepared_val, synth_metadata


def _current_lstm_params_from_config() -> Dict | None:
    lstm_config = config.get("indicators", {}).get("lstm_features", {})
    if not lstm_config.get("enabled", False):
        return None
    return {
        "lookback": lstm_config.get("lookback", 10),
        "hidden_size": lstm_config.get("hidden_size", 64),
        "num_layers": lstm_config.get("num_layers", 1),
        "output_size": lstm_config.get("output_size", 8),
        "beta": lstm_config.get("beta", 0.001),
        "kl_warmup_epochs": lstm_config.get("kl_warmup_epochs", 10),
        "pretrain_epochs": lstm_config.get("pretrain_epochs", 50),
        "pretrain_lr": lstm_config.get("pretrain_lr", 0.001),
        "pretrain_batch_size": lstm_config.get("pretrain_batch_size", 64),
        "pretrain_patience": lstm_config.get("pretrain_patience", 10),
        "pretrain_min_delta": lstm_config.get("pretrain_min_delta", 0.0001),
    }


def run_augmentation_sweep(
    window_data_list: List[Dict],
    *,
    session_folder: str,
    eval_metric: str,
) -> Dict:
    """Evaluate synthetic oversample candidates on rebuilt window data."""
    tuning_cfg = config.get("augmentation", {}).get("tuning", {})
    candidate_ratios = [float(x) for x in tuning_cfg.get("candidate_oversample_ratios", [0.0, 0.08, 0.12, 0.18, 0.24])]
    max_windows = max(1, int(tuning_cfg.get("max_windows", 3)))
    sweep_timesteps = int(tuning_cfg.get("timesteps", 10000))
    candidate_windows = window_data_list[:max_windows]
    lstm_params = _current_lstm_params_from_config()

    reports = []
    aug_cfg = config.setdefault("augmentation", {}).setdefault("synthetic_bears", {})
    original_ratio = float(aug_cfg.get("oversample_ratio", 0.0))

    try:
        for ratio in candidate_ratios:
            aug_cfg["oversample_ratio"] = ratio
            logger.info(f"Augmentation sweep: testing synthetic oversample ratio {ratio:.3f}")
            window_reports = []

            for window_data_dict in candidate_windows:
                window_idx = int(window_data_dict["window_idx"])
                sweep_folder = (
                    f"{session_folder}/reports/augmentation_sweep/"
                    f"ratio_{str(ratio).replace('.', '_')}/window_{window_idx}"
                )
                os.makedirs(sweep_folder, exist_ok=True)

                prepared_train, prepared_val, synth_metadata = _prepare_tuning_inputs(
                    window_data_dict["train_data"].copy(),
                    window_data_dict["validation_data"].copy(),
                    window_folder=sweep_folder,
                    lstm_params=lstm_params,
                )

                model, _ = train_walk_forward_model(
                    prepared_train,
                    prepared_val,
                    initial_timesteps=sweep_timesteps,
                    additional_timesteps=0,
                    max_iterations=0,
                    n_stagnant_loops=0,
                    improvement_threshold=0.0,
                    window_folder=sweep_folder,
                    run_hyperparameter_tuning=False,
                    model_params=None,
                    window_label=f"[AUG {ratio:.3f} W{window_idx}] ",
                )
                validation_results = evaluate_agent(model, prepared_val, verbose=0, deterministic=True)
                validation_score, diagnostics = _score_tuning_trial(validation_results, len(prepared_val))

                window_reports.append({
                    "window_idx": window_idx,
                    "score": float(validation_score),
                    "return_pct": float(validation_results.get("total_return_pct", 0.0)),
                    "sortino_ratio": float(validation_results.get("sortino_ratio", 0.0)),
                    "calmar_ratio": float(validation_results.get("calmar_ratio", 0.0)),
                    "max_drawdown": float(validation_results.get("max_drawdown", 0.0)),
                    "trade_count": int(validation_results.get("trade_count", 0)),
                    "action_counts": diagnostics["action_counts"],
                    "collapse_flags": diagnostics["collapse_flags"],
                    "synthetic_bars": int((synth_metadata or {}).get("synthetic_bars", 0)),
                })

            reports.append({
                "oversample_ratio": ratio,
                "avg_score": float(np.mean([entry["score"] for entry in window_reports])) if window_reports else float("-inf"),
                "avg_return_pct": float(np.mean([entry["return_pct"] for entry in window_reports])) if window_reports else 0.0,
                "windows": window_reports,
            })
    finally:
        aug_cfg["oversample_ratio"] = original_ratio

    reports.sort(key=lambda item: item["avg_score"], reverse=True)
    summary = {
        "evaluation_metric": eval_metric,
        "timesteps_per_window": sweep_timesteps,
        "max_windows": max_windows,
        "candidates": reports,
        "best_ratio": reports[0]["oversample_ratio"] if reports else original_ratio,
    }
    save_json(summary, f"{session_folder}/reports/augmentation_tuning_results.json")
    logger.info(f"Saved augmentation sweep results to {session_folder}/reports/augmentation_tuning_results.json")
    return summary

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

        # Update model section with tuned hyperparameters
        if 'model' not in config_data:
            config_data['model'] = {}

        # Map tuned params to config locations
        model_params = ['learning_rate', 'n_steps', 'batch_size', 'gamma', 'gae_lambda', 'ent_coef']
        for param in model_params:
            if param in best_params:
                config_data['model'][param] = float(best_params[param])

        # Update sequence_model section for LSTM params
        if 'sequence_model' not in config_data:
            config_data['sequence_model'] = {}

        lstm_params = ['lstm_hidden_size', 'n_lstm_layers']
        for param in lstm_params:
            if param in best_params:
                config_data['sequence_model'][param] = int(best_params[param])

        # Write back to config
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved best hyperparameters to {config_path}")
        logger.info(f"Updated params: {list(best_params.keys())}")

    except Exception as e:
        logger.error(f"Failed to save hyperparameters to config: {e}")


def _build_tuning_env(
    data: pd.DataFrame,
    *,
    random_start_pct: float | None = None,
    min_episode_steps: int = 2,
    n_envs_override: int | None = None,
) -> Tuple[object, int]:
    """Mirror train.py rollout collection during policy tuning."""
    base_seed = int(config.get("seed", 42))
    if random_start_pct is None:
        random_start_pct = float(config.get("training", {}).get("random_start_pct", 0.2))
    env_kwargs = dict(
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 2.50),
        position_size=config["environment"].get("position_size", 1),
        random_start_pct=random_start_pct,
        min_episode_steps=min_episode_steps,
    )
    tuning_config = config.get("hyperparameter_tuning", {})
    n_envs = int(n_envs_override if n_envs_override is not None else tuning_config.get("n_envs", config.get("training", {}).get("n_envs", 4)))
    n_envs = max(1, n_envs)

    def make_env(df, **kwargs):
        return lambda: TradingEnv(df.copy(), **kwargs)

    if n_envs > 1:
        env = SubprocVecEnv([make_env(data, **env_kwargs) for _ in range(n_envs)])
        env.seed(base_seed + 1000)
        return env, n_envs

    env = DummyVecEnv([make_env(data, **env_kwargs)])
    env.seed(base_seed + 1000)
    return env, 1


def _deterministic_mode_enabled() -> bool:
    """Whether walk-forward should favor reproducible execution over throughput."""
    return bool(config.get("reproducibility", {}).get("deterministic_mode", True))


def _resolve_tuning_n_jobs() -> int:
    """Choose a machine-safe Optuna parallelism level for policy tuning."""
    if _deterministic_mode_enabled():
        return 1

    parallel_config = config.get("hyperparameter_tuning", {}).get("parallel_processing", {})
    use_parallel = parallel_config.get("enabled", True)
    if not use_parallel:
        return 1

    requested_n_jobs = int(parallel_config.get("n_jobs", 0))
    if requested_n_jobs > 0:
        return requested_n_jobs

    cpu_count = multiprocessing.cpu_count()
    tuning_n_envs = int(config.get("hyperparameter_tuning", {}).get("n_envs", config.get("training", {}).get("n_envs", 4)))
    tuning_n_envs = max(1, tuning_n_envs)
    reserve_cores = max(0, int(parallel_config.get("reserve_cores", 1)))
    max_auto_jobs = int(parallel_config.get("max_auto_jobs", 2))
    max_auto_jobs = max(1, max_auto_jobs)

    # Each trial may spawn multiple rollout env workers. Keep auto-detection conservative.
    available_cores = max(1, cpu_count - reserve_cores)
    env_limited_jobs = max(1, available_cores // tuning_n_envs)
    return max(1, min(max_auto_jobs, env_limited_jobs))


def _valid_batch_size_choices(total_rollout_steps: int) -> List[int]:
    """Prefer minibatches that evenly divide the rollout buffer."""
    candidates = [16, 32, 64, 128, 256, 512]
    valid = [size for size in candidates if size <= total_rollout_steps and total_rollout_steps % size == 0]
    if valid:
        return valid
    fallback = [size for size in candidates if size <= total_rollout_steps]
    return fallback if fallback else [total_rollout_steps]


def _resolve_staged_trial_counts(n_trials: int, stage1_trials: int, stage2_trials: int) -> Tuple[int, int]:
    """Fit staged tuning trial counts inside the requested total budget."""
    n_trials = max(0, int(n_trials))
    stage1_trials = max(0, int(stage1_trials))
    stage2_trials = max(0, int(stage2_trials))

    if n_trials <= 0:
        return 0, 0

    total_requested = stage1_trials + stage2_trials
    if total_requested <= 0:
        return n_trials, 0
    if total_requested <= n_trials:
        return stage1_trials, stage2_trials

    if stage1_trials == 0:
        return 0, n_trials
    if stage2_trials == 0:
        return n_trials, 0

    stage1_share = stage1_trials / total_requested
    resolved_stage1 = max(1, int(round(n_trials * stage1_share)))
    resolved_stage1 = min(resolved_stage1, n_trials - 1) if n_trials > 1 else n_trials
    resolved_stage2 = max(0, n_trials - resolved_stage1)
    return resolved_stage1, resolved_stage2

def load_tradingview_data(csv_filepath: str | None = None) -> pd.DataFrame:
    """
    Load and process data from a TradingView CSV export file.
    
    Args:
        csv_filepath: Path to the TradingView CSV file
        
    Returns:
        DataFrame: Processed data with technical indicators
    """
    if csv_filepath is None:
        csv_filepath = config.get("data", {}).get("csv_path", "data/NQ_2024_unix.csv")

    logger.info(f"Loading TradingView data from {csv_filepath}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_filepath)
        
        # Debug information
        logger.info(f"Raw TradingView data columns: {df.columns.tolist()}")
        
        # Check if we have the required columns
        required_cols = ['open', 'high', 'low', 'close']
        time_cols = ['time', 'timestamp', 'ts_event']  # Accept common source timestamp columns
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
            df = df.sort_index()
            df = select_session_dominant_contract(df, keep='last')
            df = ensure_numeric(df, ['open', 'high', 'low', 'close', 'volume'], drop_invalid=True)
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

# Import shared data-session helpers
from utils.data_utils import filter_market_hours, market_hours_only_enabled

def get_trading_days(data: pd.DataFrame) -> List[str]:
    """
    Extract unique trading days from a DataFrame with filtered market hours.
    
    Args:
        data: DataFrame with DatetimeIndex in UTC, already filtered to market hours
        
    Returns:
        List[str]: List of unique trading days in YYYY-MM-DD format
    """
    session_cfg = session_config_from_mapping(config.get("data", {}).get("session", {}))
    session_ctx = build_session_context(data.index, session_cfg)
    unique_days = sorted(set(session_ctx["session_anchor"].date.astype(str)))
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
    current_contracts = 0
    entry_price = Decimal('0')  # Track entry price for P&L calculation

    # Trade tracking
    trade_history = []
    trade_count = 0

    # Track action history for plotting
    action_history = []
    portfolio_history = [float(initial_balance)]
    
    # Reset environment to start evaluation
    obs, _ = env.reset(seed=int(config.get("seed", 42)))
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

        new_position = action_direction(position_action)
        
        # Check if we're at the last step or not
        # We need the next price to determine if prediction was correct
        if current_step < len(test_data) - 1:
            # Get next price
            next_price = money.to_decimal(test_data.iloc[current_step + 1][close_col])
            price_change = next_price - current_price
            
            # Evaluate prediction accuracy based on action type
            prediction_correct = False
            
            if new_position == 1:
                prediction_correct = price_change > 0
            elif new_position == -1:
                prediction_correct = price_change < 0
            else:
                # Flat is correct if we avoided a losing move
                if current_position == 1:  # Was long, flat is correct if price went down
                    prediction_correct = price_change < 0
                elif current_position == -1:  # Was short, flat is correct if price went up
                    prediction_correct = price_change > 0
                else:  # Already flat, correct if price doesn't move much
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
        
        old_position = current_position

        # Take step in environment
        new_obs, reward, done, truncated, info = env.step(action)
        obs = new_obs
        done = done or truncated

        # Track actual environment rebalances, not just direction changes.
        position_changed = bool(info.get("position_changed", False))
        env_old_position = int(info.get("old_position", current_position))
        env_new_position = int(info.get("position", env.position))
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

            is_profitable = False
            trade_pnl = Decimal('0')
            if realized_trade and entry_price > 0:
                if env_old_position == 1:
                    trade_pnl = current_price - entry_price
                    is_profitable = trade_pnl > 0
                elif env_old_position == -1:
                    trade_pnl = entry_price - current_price
                    is_profitable = trade_pnl > 0

            trade_info = {
                "step": current_step,
                "action": action_label(int(position_action)),
                "price": float(current_price),
                "timestamp": test_data.index[current_step].strftime('%Y-%m-%d %H:%M:%S') if hasattr(test_data.index[current_step], 'strftime') else str(test_data.index[current_step]),
                "old_position": env_old_position,
                "new_position": env_new_position,
                "old_contracts": old_contracts,
                "new_contracts": new_contracts,
                "entry_price": float(entry_price) if entry_price > 0 else None,
                "trade_pnl_points": float(trade_pnl),
                "realized_trade": realized_trade,
                "profitable": is_profitable
            }
            trade_history.append(trade_info)

            if new_contracts != 0:
                entry_price = money.to_decimal(info.get("avg_entry_price", current_price))
            else:
                entry_price = Decimal('0')

            current_position = env_new_position
            current_contracts = new_contracts

        # Update portfolio from environment
        current_portfolio = env.net_worth
        # Sample portfolio history (every 10 steps or on position change) to reduce memory
        if position_changed or len(portfolio_history) == 0 or current_step % 10 == 0:
            portfolio_history.append(float(current_portfolio))

        current_position = env_new_position
        current_contracts = new_contracts
    
    # Calculate final metrics
    prediction_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    total_return_pct = money.calculate_return_pct(current_portfolio, initial_balance)
    
    # Calculate hit rate from realized trade events only.
    profitable_trades = sum(1 for t in trade_history if t.get("profitable", False))
    completed_trades = sum(1 for t in trade_history if t.get("realized_trade", False))
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
        "completed_trades": completed_trades,
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


def _run_window_hyperparameter_tuning(
    *,
    window_data_list: List[Dict],
    anchor_idx: int,
    tuning_trials: int,
    eval_metric: str,
    session_lstm_params: Dict | None,
    lstm_tuning_scope: str,
    session_folder: str,
) -> Tuple[Dict, Dict]:
    """Tune PPO/RecurrentPPO on a specific walk-forward anchor window."""
    lstm_config = config.get("indicators", {}).get("lstm_features", {})
    lstm_tuning_config = lstm_config.get("tuning", {})
    anchor_window = window_data_list[anchor_idx]
    anchor_window_num = anchor_window["window_idx"]

    logger.info("Starting hyperparameter tuning on window %d data", anchor_window_num)
    tuning_train_raw = anchor_window["train_data"].copy()
    tuning_val_raw = anchor_window["validation_data"].copy()

    tuning_lstm_params = session_lstm_params
    if (
        lstm_config.get("enabled", False)
        and get_representation_mode() != "engineered_only"
        and lstm_tuning_config.get("enabled", False)
        and lstm_tuning_scope == "per_window"
    ):
        logger.info("Tuning LSTM hyperparameters on prepared window %d before PPO tuning", anchor_window_num)
        prep_train, prep_val, _, prep_synth = _prepare_scaled_window_inputs(
            tuning_train_raw,
            tuning_val_raw,
            window_folder=anchor_window["window_folder"],
        )
        if prep_synth:
            logger.info(
                f"Window {anchor_window_num} PPO-tuning prep augmentation added {prep_synth.get('synthetic_bars', 0)} synthetic bars "
                f"across {prep_synth.get('num_segments', 0)} segments"
            )
        prep_feature_columns = get_market_feature_columns(prep_train, representation_mode="hybrid")
        if prep_feature_columns:
            tuning_lstm_params = tune_lstm_hyperparameters(
                train_data=prep_train[prep_feature_columns],
                validation_data=prep_val[prep_feature_columns],
                tuning_config=lstm_tuning_config,
                base_config=lstm_config,
                feature_columns=prep_feature_columns,
                window_folder=anchor_window["window_folder"],
            )

    tuning_train, tuning_val, tuning_synth_metadata = _prepare_tuning_inputs(
        tuning_train_raw,
        tuning_val_raw,
        window_folder=anchor_window["window_folder"],
        lstm_params=tuning_lstm_params,
    )
    if tuning_synth_metadata:
        logger.info(
            f"Window {anchor_window_num} tuning augmentation added {tuning_synth_metadata.get('synthetic_bars', 0)} synthetic bars "
            f"across {tuning_synth_metadata.get('num_segments', 0)} segments"
        )

    tuning_holdout_sets: List[pd.DataFrame] = []
    holdout_window_limit = int(config.get("hyperparameter_tuning", {}).get("holdout_windows", 2))
    for holdout_window in window_data_list[anchor_idx + 1:anchor_idx + 1 + holdout_window_limit]:
        _, holdout_val, _ = _prepare_tuning_inputs(
            holdout_window["train_data"],
            holdout_window["validation_data"],
            window_folder=holdout_window["window_folder"],
            lstm_params=tuning_lstm_params,
        )
        tuning_holdout_sets.append(_align_feature_columns(list(tuning_train.columns), holdout_val))

    tuning_result = hyperparameter_tuning(
        train_data=tuning_train,
        validation_data=tuning_val,
        n_trials=tuning_trials,
        window_folder=anchor_window["window_folder"],
        eval_metric=eval_metric,
        holdout_validation_sets=tuning_holdout_sets,
    )
    best_hyperparameters = tuning_result.get("best_params", {})
    _apply_best_hyperparameters_to_config(best_hyperparameters)

    tuning_summary = {k: v for k, v in tuning_result.items() if k != "study"}
    save_json(best_hyperparameters, f"{session_folder}/reports/best_hyperparameters_window_{anchor_window_num}.json")
    save_json(tuning_summary, f"{session_folder}/reports/hyperparameter_tuning_window_{anchor_window_num}.json")
    save_json(best_hyperparameters, f"{session_folder}/reports/best_hyperparameters.json")
    save_json(tuning_summary, f"{session_folder}/reports/hyperparameter_tuning_summary.json")
    logger.info("Best hyperparameters for window %d anchor: %s", anchor_window_num, best_hyperparameters)
    return best_hyperparameters, tuning_summary

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
    resolved_lstm_params: Dict | None = None,
    lstm_tuning_scope: str = "session",
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

    train_data, validation_data, test_data, synthetic_bears_metadata = _prepare_scaled_window_inputs(
        train_data,
        validation_data,
        test_data,
        window_folder=window_folder,
    )

    representation_mode = get_representation_mode()
    lstm_config = config.get("indicators", {}).get("lstm_features", {})
    market_feature_columns = get_market_feature_columns(train_data, representation_mode="hybrid")
    latent_columns: List[str] = []

    if lstm_config.get("enabled", False) and representation_mode != "engineered_only" and market_feature_columns:
        lstm_tuning_config = lstm_config.get("tuning", {})
        lstm_params = resolved_lstm_params

        if lstm_tuning_config.get("enabled", False) and lstm_tuning_scope == "per_window":
            window_logger.info("Tuning LSTM hyperparameters for this window on canonical market features")
            lstm_params = tune_lstm_hyperparameters(
                train_data=train_data[market_feature_columns],
                validation_data=validation_data[market_feature_columns],
                tuning_config=lstm_tuning_config,
                base_config=lstm_config,
                feature_columns=market_feature_columns,
                window_folder=window_folder
            )
            window_logger.info(f"Best LSTM params: hidden={lstm_params['hidden_size']}, "
                             f"layers={lstm_params['num_layers']}, output={lstm_params['output_size']}, "
                             f"lookback={lstm_params['lookback']}, lr={lstm_params['pretrain_lr']:.6f}")
        elif lstm_params is None:
            lstm_params = _current_lstm_params_from_config()

        if lstm_params is not None:
            window_logger.info(
                "Training LSTM autoencoder for this window on %d market features (%s mode)",
                len(market_feature_columns),
                representation_mode,
            )
            checkpoint_path = f"{window_folder}/lstm_autoencoder_checkpoint.pt"
            generator_path = f"{window_folder}/lstm_generator.pkl"
            train_data, validation_data, test_data, latent_columns = _attach_latent_features(
                train_data,
                validation_data,
                test_data,
                lstm_params=lstm_params,
                feature_columns=market_feature_columns,
                checkpoint_path=checkpoint_path,
                save_path=generator_path,
            )
            window_logger.info(f"Added {len(latent_columns)} latent features")

    _apply_representation_mode(
        [train_data, validation_data, test_data],
        market_feature_columns=market_feature_columns,
        latent_columns=latent_columns,
    )
    _drop_unused_model_columns([train_data, validation_data, test_data])

    logger.info(f"Model input columns ({len(train_data.columns)}): {train_data.columns.tolist()}")

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
        window_label=f"[W{window_idx}/{num_windows}] "
    )

    # Save loss history if available
    if "loss_history" in training_stats and training_stats["loss_history"]:
        loss_history_path = f"{window_folder}/loss_history.json"
        save_json(training_stats["loss_history"], loss_history_path)

    # Plot training progress (pass iterations list)
    plot_training_progress(training_stats.get("iterations", training_stats), window_folder)

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

    validation_results_path = f"{window_folder}/validation_results.json"
    validation_results = {}
    if os.path.exists(validation_results_path):
        with open(validation_results_path, "r") as f:
            validation_results = json.load(f)

    report_payload = build_window_report_payload(
        window_idx=window_idx,
        test_data=test_data,
        test_results=test_results,
        training_stats=training_stats,
        validation_results=validation_results,
        window_periods=window_periods,
        evaluation_metric=config.get("training", {}).get("evaluation", {}).get("metric", "return"),
    )
    save_json(report_payload, f"{window_folder}/report_data.json")
    
    # Compile window result
    window_result = {
        "window": window_idx,
        "window_folder": window_folder,
        "report_data_path": f"{window_folder}/report_data.json",
        "return": test_results["total_return_pct"],
        "portfolio_value": test_results["final_portfolio_value"],
        "trade_count": test_results["trade_count"],
        "rebalance_count": test_results.get("rebalance_count", test_results["trade_count"]),
        "economic_trade_count": test_results.get(
            "economic_trade_count",
            test_results.get("completed_trades", 0),
        ),
        "action_history": test_results.get("action_history", []),
        "trade_history": test_results.get("trade_history", []),
        "final_position": test_results["final_position"],
        "train_start": train_data.index[0],
        "train_end": train_data.index[-1],
        "test_start": test_data.index[0],
        "test_end": test_data.index[-1]
    }
    if synthetic_bears_metadata:
        window_result["synthetic_bars"] = synthetic_bears_metadata.get("synthetic_bars", 0)
        window_result["synthetic_segments"] = synthetic_bears_metadata.get("num_segments", 0)
    
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
    action_dist = format_action_distribution(test_results.get('action_history') or test_results.get('action_counts'))
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
    max_windows: int = 0,
    tune_augmentation_only: bool = False,
) -> Dict:
    """
    Perform walk-forward testing with anchored walk-forward analysis.
    """
    if _deterministic_mode_enabled():
        enable_full_determinism(int(config.get("seed", 42)))
    else:
        set_global_seed(int(config.get("seed", 42)))

    # Create session folder within models directory
    timestamp = _session_timestamp()
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
    market_hours_only = market_hours_only_enabled(config)
    if market_hours_only:
        logger.info("Filtering data to NYSE RTH only")
        data = filter_market_hours(data)
    else:
        logger.info("Using full-session data without NYSE RTH filtering")
    
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

    tuning_config = config.get("hyperparameter_tuning", {})
    retune_every_windows = int(tuning_config.get("retune_every_windows", 0))
    if run_hyperparameter_tuning and use_parallel and retune_every_windows > 0:
        logger.warning(
            "Per-window retuning requires sequential walk-forward processing. Disabling parallel window execution."
        )
        use_parallel = False
    
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
        "market_hours_only": market_hours_only,
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
    session_params["representation_mode"] = get_representation_mode()
    session_params["lstm_tuning_scope"] = _current_lstm_tuning_scope()
    
    save_json(session_params, f'{session_folder}/reports/session_parameters.json')
    
    # Prepare window data for all windows
    window_data_list = []
    best_hyperparameters = None  # Store the best hyperparameters from first window
    
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
        
        session_cfg = session_config_from_mapping(config.get("data", {}).get("session", {}))
        session_ctx = build_session_context(data.index, session_cfg)
        session_days = pd.Index(session_ctx["session_anchor"].date.astype(str))
        window_mask = (session_days >= start_day) & (session_days <= end_day)
        window_data = data.loc[window_mask].copy()
        
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

    lstm_config = config.get("indicators", {}).get("lstm_features", {})
    lstm_tuning_config = lstm_config.get("tuning", {})
    lstm_tuning_scope = _current_lstm_tuning_scope()
    session_lstm_params = None
    session_encoder_feature_columns = None
    if lstm_config.get("enabled", False) and get_representation_mode() != "engineered_only":
        session_lstm_params = _current_lstm_params_from_config()
        if lstm_tuning_config.get("enabled", False) and lstm_tuning_scope == "session" and window_data_list:
            first_window = window_data_list[0]
            logger.info("Running session-level LSTM tuning on the first prepared window")
            session_tune_train, session_tune_val, _, session_tune_synth = _prepare_scaled_window_inputs(
                first_window["train_data"],
                first_window["validation_data"],
                window_folder=first_window["window_folder"],
            )
            if session_tune_synth:
                logger.info(
                    f"First-window LSTM tuning augmentation added {session_tune_synth.get('synthetic_bars', 0)} synthetic bars "
                    f"across {session_tune_synth.get('num_segments', 0)} segments"
                )
            session_feature_columns = get_market_feature_columns(session_tune_train, representation_mode="hybrid")
            if session_feature_columns:
                session_encoder_feature_columns = list(session_feature_columns)
                session_lstm_params = tune_lstm_hyperparameters(
                    train_data=session_tune_train[session_feature_columns],
                    validation_data=session_tune_val[session_feature_columns],
                    tuning_config=lstm_tuning_config,
                    base_config=lstm_config,
                    feature_columns=session_feature_columns,
                    window_folder=first_window["window_folder"],
                )
                logger.info(
                    "Resolved session LSTM params: hidden=%s layers=%s output=%s lookback=%s lr=%.6f",
                    session_lstm_params["hidden_size"],
                    session_lstm_params["num_layers"],
                    session_lstm_params["output_size"],
                    session_lstm_params["lookback"],
                    session_lstm_params["pretrain_lr"],
                )
        elif window_data_list:
            session_preview_train, session_preview_val, _, _ = _prepare_scaled_window_inputs(
                window_data_list[0]["train_data"],
                window_data_list[0]["validation_data"],
                window_folder=None,
            )
            session_encoder_feature_columns = get_market_feature_columns(
                session_preview_train,
                representation_mode="hybrid",
            )
    session_params["resolved_lstm_params"] = session_lstm_params
    session_params["encoder_feature_columns"] = session_encoder_feature_columns
    session_params["latent_width"] = int(session_lstm_params["output_size"]) if session_lstm_params else 0
    save_json(session_params, f'{session_folder}/reports/session_parameters.json')
    
    if tune_augmentation_only:
        summary = run_augmentation_sweep(
            window_data_list,
            session_folder=session_folder,
            eval_metric=eval_metric,
        )
        summary["timestamp"] = timestamp
        return summary

    retune_during_loop = run_hyperparameter_tuning and not use_parallel and retune_every_windows > 0
    if run_hyperparameter_tuning and not retune_during_loop:
        best_hyperparameters, _ = _run_window_hyperparameter_tuning(
            window_data_list=window_data_list,
            anchor_idx=0,
            tuning_trials=tuning_trials,
            eval_metric=eval_metric,
            session_lstm_params=session_lstm_params,
            lstm_tuning_scope=lstm_tuning_scope,
            session_folder=session_folder,
        )
    
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
                        best_hyperparameters,  # Pass the best hyperparameters found
                        session_lstm_params,
                        lstm_tuning_scope,
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
        for list_idx, window_data_dict in enumerate(window_data_list):
            try:
                if run_hyperparameter_tuning and retune_during_loop:
                    if best_hyperparameters is None or (list_idx % retune_every_windows) == 0:
                        best_hyperparameters, _ = _run_window_hyperparameter_tuning(
                            window_data_list=window_data_list,
                            anchor_idx=list_idx,
                            tuning_trials=tuning_trials,
                            eval_metric=eval_metric,
                            session_lstm_params=session_lstm_params,
                            lstm_tuning_scope=lstm_tuning_scope,
                            session_folder=session_folder,
                        )
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
                    best_hyperparameters,  # Pass the best hyperparameters found
                    session_lstm_params,
                    lstm_tuning_scope,
                )
                all_window_results.append(window_result)
            except Exception as e:
                logger.error(f"Error processing window {window_data_dict['window_idx']}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # Aggregate results across all windows
    returns = [res["return"] for res in all_window_results]
    portfolio_values = [res["portfolio_value"] for res in all_window_results]
    trade_counts = [res["trade_count"] for res in all_window_results]
    rebalance_counts = [res.get("rebalance_count", res["trade_count"]) for res in all_window_results]
    economic_trade_counts = [res.get("economic_trade_count", 0) for res in all_window_results]
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
    avg_rebalances = np.mean(rebalance_counts)
    avg_completed_trades = np.mean(economic_trade_counts) if economic_trade_counts else 0
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
    logger.info(f"Average completed trades: {avg_completed_trades:.2f}")
    logger.info(f"Average rebalances: {avg_rebalances:.2f}")
    
    # Save summary results
    summary_results = {
        "avg_return": avg_return,
        "avg_sortino": avg_sortino,
        "avg_hit_rate": avg_hit_rate,
        "avg_prediction_accuracy": avg_prediction_accuracy,
        "avg_portfolio": avg_portfolio,
        "avg_trades": avg_trades,
        "avg_completed_trades": avg_completed_trades,
        "avg_rebalances": avg_rebalances,
        "avg_profitable_trades": avg_profitable_trades,
        "avg_correct_predictions": avg_correct_predictions,
        "avg_total_predictions": avg_total_predictions,
        "num_windows": num_windows,
        "timestamp": timestamp,
        "evaluation_metric": eval_metric
    }
    
    # Don't include all window results in the summary JSON (they may contain non-serializable objects)
    json_safe_summary = dict(summary_results)
    save_json(json_safe_summary, f'{session_folder}/reports/summary_results.json')

    # Add window results back for the return value (not for JSON serialization)
    summary_results["all_window_results"] = all_window_results
    
    # Plot results
    plot_walk_forward_results(all_window_results, session_folder, eval_metric)
    
    # Export consolidated trade history
    export_consolidated_trade_history(all_window_results, session_folder)

    # Generate engineer-facing HTML report when window artifacts are available.
    try:
        report_path = generate_walk_forward_report(session_folder)
        summary_results["html_report_path"] = report_path
        logger.info(f"Generated walk-forward HTML report: {report_path}")
    except Exception as e:
        logger.error(f"Failed to generate walk-forward HTML report: {e}")
        import traceback
        logger.error(traceback.format_exc())

    json_safe_summary = {k: v for k, v in summary_results.items() if k != "all_window_results"}
    save_json(json_safe_summary, f'{session_folder}/reports/summary_results.json')
    
    return summary_results

def hyperparameter_tuning(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    n_trials: int = 30,
    window_folder: str = None,
    eval_metric: str = "return",
    hit_rate_min_trades: int = 5,
    holdout_validation_sets: List[pd.DataFrame] | None = None,
) -> Dict:
    """
    Run hyperparameter tuning using Optuna with parallel processing.
    
    Args:
        train_data: Training data DataFrame
        validation_data: Validation data DataFrame
        n_trials: Number of trials for Optuna optimization
        window_folder: Folder to save visualization results
        eval_metric: Evaluation metric to optimize for
        hit_rate_min_trades: Minimum number of trades required for hit rate to be meaningful
        holdout_validation_sets: Optional extra validation datasets for lightweight holdout scoring
        
    Returns:
        Dict: Best hyperparameters and optimization results
    """
    logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
    logger.info(f"Evaluation metric: {eval_metric}")
    holdout_validation_sets = holdout_validation_sets or []
    if get_configured_algorithm() == "qrdqn":
        logger.info("Using QR-DQN benchmark hyperparameter tuning")
        return _run_qrdqn_tuning(
            train_data=train_data,
            validation_data=validation_data,
            n_trials=n_trials,
            window_folder=window_folder,
            holdout_validation_sets=holdout_validation_sets,
        )

    tuning_config = config.get("hyperparameter_tuning", {})
    parallel_config = tuning_config.get("parallel_processing", {})
    use_parallel = parallel_config.get("enabled", True)
    n_jobs = _resolve_tuning_n_jobs()
    if use_parallel:
        logger.info(f"Using parallel processing with {n_jobs} workers for hyperparameter tuning")
    else:
        logger.info("Parallel processing disabled for hyperparameter tuning")
        n_jobs = 1

    risk_enabled = config.get("risk_management", {}).get("enabled", False)
    logger.info("Risk management is %s", "enabled" if risk_enabled else "disabled")

    hp_config = tuning_config.get("parameters", {})
    reference_columns = list(train_data.columns)
    baseline_margin = float(tuning_config.get("baseline_margin", 0.25))

    requested_stage1_trials = int(tuning_config.get("stage1_trials", n_trials))
    requested_stage2_trials = int(tuning_config.get("stage2_trials", max(0, n_trials - requested_stage1_trials)))
    stage1_trials, stage2_trials = _resolve_staged_trial_counts(
        n_trials,
        requested_stage1_trials,
        requested_stage2_trials,
    )
    if stage1_trials + stage2_trials <= 0:
        stage1_trials = max(1, n_trials)
        stage2_trials = 0

    stage1_timesteps = int(tuning_config.get("stage1_timesteps", tuning_config.get("tuning_timesteps", 20000)))
    stage2_timesteps = int(tuning_config.get("stage2_timesteps", tuning_config.get("tuning_timesteps", 20000)))
    stage2_window_shrink = float(tuning_config.get("stage2_window_shrink", 0.35))

    if (requested_stage1_trials, requested_stage2_trials) != (stage1_trials, stage2_trials):
        logger.info(
            "Adjusted staged tuning trials to fit requested budget: stage1=%d, stage2=%d, total=%d",
            stage1_trials,
            stage2_trials,
            n_trials,
        )
    logger.info(
        "Staged PPO tuning: stage1=%d trials @ %d steps, stage2=%d trials @ %d steps",
        stage1_trials,
        stage1_timesteps,
        stage2_trials,
        stage2_timesteps,
    )
    if holdout_validation_sets:
        logger.info(
            "Using %d additional holdout validation window(s) in stage 2",
            len(holdout_validation_sets),
        )

    stage1_sets = [validation_data]
    stage1_baselines = _score_constant_action_baselines(stage1_sets, reference_columns)
    selection_sets = stage1_sets
    selection_baselines = stage1_baselines
    logger.info(
        "Stage 1 trivial baseline scores: %s",
        [round(max(scores.values()), 2) for scores in stage1_baselines],
    )
    stage1_result = _run_tuning_stage(
        stage_name="stage1",
        hp_config=hp_config,
        train_data=train_data,
        validation_data=validation_data,
        evaluation_sets=stage1_sets,
        reference_columns=reference_columns,
        baseline_scores=stage1_baselines,
        baseline_margin=baseline_margin,
        n_trials=stage1_trials,
        tuning_timesteps=stage1_timesteps,
        window_folder=window_folder,
        use_parallel=use_parallel,
        n_jobs=n_jobs,
    )

    final_result = stage1_result
    stage2_result = {"best_params": {}, "best_value": float("-inf"), "study": None}
    stage2_search_space = {}

    if stage2_trials > 0:
        stage2_sets = [validation_data] + holdout_validation_sets
        stage2_baselines = _score_constant_action_baselines(stage2_sets, reference_columns)
        logger.info(
            "Stage 2 trivial baseline scores: %s",
            [round(max(scores.values()), 2) for scores in stage2_baselines],
        )
        stage2_hp_config = _narrow_hp_config(hp_config, stage1_result.get("best_params", {}), stage2_window_shrink)
        stage2_search_space = copy.deepcopy(stage2_hp_config)
        stage2_result = _run_tuning_stage(
            stage_name="stage2",
            hp_config=stage2_hp_config,
            train_data=train_data,
            validation_data=validation_data,
            evaluation_sets=stage2_sets,
            reference_columns=reference_columns,
            baseline_scores=stage2_baselines,
            baseline_margin=baseline_margin,
            n_trials=stage2_trials,
            tuning_timesteps=stage2_timesteps,
            window_folder=window_folder,
            use_parallel=use_parallel,
            n_jobs=n_jobs,
        )
        if stage2_result.get("best_params"):
            final_result = stage2_result
            selection_sets = stage2_sets
            selection_baselines = stage2_baselines
        else:
            selection_sets = stage2_sets
            selection_baselines = stage2_baselines

    selected_stage = "stage2" if stage2_result.get("best_params") else "stage1"
    finalist_result = None
    finalist_top_k = int(tuning_config.get("finalist_top_k", 0))
    finalist_num_seeds = int(tuning_config.get("finalist_num_seeds", 0))
    finalist_timesteps = int(tuning_config.get("finalist_timesteps", 0))
    final_study = final_result.get("study")
    finalist_trials = _candidate_trials_from_study(final_study, finalist_top_k)
    finalist_candidates = [{"params": dict(trial.params), "score": float(trial.value)} for trial in finalist_trials]
    if finalist_candidates:
        logger.info(
            "Re-ranking %d finalist candidate(s) across %d seed(s) at %d timesteps",
            len(finalist_candidates),
            finalist_num_seeds,
            finalist_timesteps,
        )
        finalist_result = _evaluate_finalist_candidates(
            candidates=finalist_candidates,
            train_data=train_data,
            validation_data=validation_data,
            holdout_validation_sets=selection_sets[1:],
            baseline_scores=selection_baselines,
            baseline_margin=baseline_margin,
            finalist_timesteps=finalist_timesteps,
            finalist_num_seeds=finalist_num_seeds,
        )
        if finalist_result is not None:
            best_params = finalist_result["params"]
            best_value = float(finalist_result["median_score"])
            selected_stage = f"{selected_stage}_finalists"
        else:
            finalist_candidates = []
    else:
        finalist_candidates = []

    if finalist_result is None:
        best_params = final_result.get("best_params", {})
        best_value = float(final_result.get("best_value", float("-inf")))

    logger.info("\n" + "=" * 80)
    logger.info("Hyperparameter Tuning Results:")
    logger.info(f"Selected stage: {selected_stage}")
    logger.info(f"Best composite score: {best_value:.2f}")
    logger.info("Best parameters:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")

    if window_folder and final_study is not None:
        try:
            os.makedirs('models/plots/tuning', exist_ok=True)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            try:
                import plotly

                fig1 = optuna.visualization.plot_optimization_history(final_study)
                fig1.write_image(f'models/plots/tuning/optimization_history_{timestamp}.png')

                fig2 = optuna.visualization.plot_param_importances(final_study)
                fig2.write_image(f'models/plots/tuning/param_importances_{timestamp}.png')

                logger.info("Saved optimization visualizations to models/plots/tuning/")
            except ImportError:
                logger.warning("Plotly is not installed. Skipping optimization visualization plots.")
        except Exception as e:
            logger.warning(f"Could not save optimization visualizations: {e}")

    return {
        "best_params": best_params,
        "best_value": best_value,
        "selected_stage": selected_stage,
        "stage1_best_params": stage1_result.get("best_params", {}),
        "stage1_best_value": float(stage1_result.get("best_value", float("-inf"))),
        "stage2_best_params": stage2_result.get("best_params", {}),
        "stage2_best_value": float(stage2_result.get("best_value", float("-inf"))),
        "stage2_search_space": stage2_search_space,
        "finalist_result": finalist_result,
        "finalist_candidates": finalist_candidates,
        "study": final_study,
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
    if _deterministic_mode_enabled():
        enable_full_determinism(int(config.get("seed", 42)))
    else:
        set_global_seed(int(config.get("seed", 42)))

    parser = argparse.ArgumentParser(description="Walk-forward testing")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Scenario name from scenarios.yaml (e.g. steady_bull_2021)")
    parser.add_argument("--tune-augmentation", action="store_true",
                        help="Run a dedicated synthetic augmentation sweep and exit")
    args = parser.parse_args()

    # Create model directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/logs', exist_ok=True)

    # Load scenario config if specified
    data_file = config.get("data", {}).get("csv_path", "data/NQ_2024_unix.csv")
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
        start_ts = pd.Timestamp(start_date)
        if getattr(full_data.index, "tz", None) is not None and start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        full_data = full_data[full_data.index >= start_ts]
    if end_date:
        end_ts = pd.Timestamp(end_date)
        if getattr(full_data.index, "tz", None) is not None and end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        full_data = full_data[full_data.index <= end_ts + pd.Timedelta(days=1)]

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
        max_windows=config.get("walk_forward", {}).get("max_windows", 0),
        tune_augmentation_only=args.tune_augmentation,
    )

    if args.tune_augmentation:
        print("\nAugmentation Sweep Summary:")
        print(f"Best oversample ratio: {results.get('best_ratio')}")
        print(f"Results saved to models/session_{results['timestamp']}/reports/augmentation_tuning_results.json")
        return
    
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
    if results.get("html_report_path"):
        print(f"HTML report: {results['html_report_path']}")
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
    
    
    if market_hours_only_enabled(config):
        print("Note: All references to 'days' now indicate NYSE RTH trading days, not calendar days")
        print("Note: Training and evaluation were performed only on NYSE RTH data (09:30-16:00 America/New_York, Monday-Friday)")
    else:
        print("Note: All references to 'days' now indicate trading days derived from the loaded dataset, not calendar days")

if __name__ == "__main__":
    main() 
