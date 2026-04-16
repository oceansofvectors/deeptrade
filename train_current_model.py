#!/usr/bin/env python3
"""Train a single RecurrentPPO model on the most-recent walk-forward window.

Freshness contract (the "most-current" window is defined to match walk-forward):
    window_size  = config.walk_forward.window_size  (default 120 trading days)
    split        = config.data.{train,validation,test}_ratio  (default 60/20/20)
    embargo      = config.data.embargo_days  (default 1)

Rollover cadence: retrain every config.walk_forward.step_size trading days
(default 24, ~monthly). Retrain sooner if drift triggers fire:
    - 5-day realized vol leaves the training-window 10-90th VOL_PERCENTILE band
    - Live daily drawdown exceeds 1.5x training max DD for 3 sessions
    - ROLLING_DD in live bars crosses a regime threshold

Usage:
    python train_current_model.py                       # full training on data/NQ_live.csv
    python train_current_model.py --skip-training       # data pipeline smoke test
    python train_current_model.py --data data/NQ_5min_2020_2025.csv

Output bundle (self-contained, portable to live):
    models/current/<timestamp>/
        best_model.zip
        indicator_scaler.pkl
        close_norm_params.pkl
        lstm_generator.pkl
        lstm_autoencoder_checkpoint.pt
        feature_order.json
        training_manifest.json
    models/current/latest  ->  models/current/<timestamp>
"""

import argparse
import json
import logging
import multiprocessing
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# macOS Python 3.13 + numpy spawn-mode is broken ("Could not parse python long
# as longdouble") — force fork so SubprocVecEnv workers can import numpy.
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

import pandas as pd
import pytz

from config import config
from environment import TradingEnv
from get_data import process_technical_indicators
from indicators.lstm_features import LSTMFeatureGenerator
from normalization import get_standardized_column_names, scale_window

import train as _train_module  # noqa: E402
from train import evaluate_agent, train_agent_iteratively  # noqa: E402

# SB3's SubprocVecEnv defaults to forkserver/spawn on macOS, which crashes
# in Python 3.13 with "Could not parse python long as longdouble". Force
# fork by replacing the SubprocVecEnv symbol inside the `train` module.
if sys.platform == "darwin":
    from stable_baselines3.common.vec_env import SubprocVecEnv as _OriginalSubproc

    def _fork_subproc(env_fns, start_method=None):
        return _OriginalSubproc(env_fns, start_method="fork")

    _train_module.SubprocVecEnv = _fork_subproc
from utils.data_utils import filter_market_hours
from utils.seeding import set_global_seed
from utils.synthetic_bears import augment_with_synthetic_bears, extract_ohlcv_frame
from walk_forward import get_trading_days, load_tradingview_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def slice_last_window(data: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Return the last `window_size` trading days of data, matching walk_forward slicing."""
    trading_days = get_trading_days(data)
    if len(trading_days) == 0:
        raise ValueError("No trading days found after market-hours filter")

    if len(trading_days) < window_size:
        logger.warning(
            f"Only {len(trading_days)} trading days in dataset, requested {window_size}. "
            f"Using all available days."
        )
        start_day = trading_days[0]
    else:
        start_day = trading_days[-window_size]
    end_day = trading_days[-1]

    eastern = pytz.timezone("US/Eastern")
    data_eastern = data.copy()
    if data_eastern.index.tz is None:
        data_eastern.index = data_eastern.index.tz_localize("UTC").tz_convert(eastern)
    else:
        data_eastern.index = data_eastern.index.tz_convert(eastern)

    mask = (data_eastern.index.date.astype(str) >= start_day) & (
        data_eastern.index.date.astype(str) <= end_day
    )
    window = data_eastern[mask].copy()

    if data.index.tz is None:
        window.index = window.index.tz_localize(None)
    else:
        window.index = window.index.tz_convert("UTC")

    logger.info(
        f"Window: {window.index[0]} -> {window.index[-1]} "
        f"({len(window)} bars across {len(trading_days[-window_size:])} trading days)"
    )
    return window


def split_with_embargo(window_data: pd.DataFrame, train_ratio: float, validation_ratio: float, embargo_days: int):
    """Mirror walk_forward.py lines ~1098-1120."""
    train_idx = int(len(window_data) * train_ratio)
    validation_idx = train_idx + int(len(window_data) * validation_ratio)

    if len(window_data) > 1:
        time_diff = (window_data.index[1] - window_data.index[0]).total_seconds()
        bars_per_day = int(24 * 60 * 60 / time_diff) if time_diff > 0 else 288
    else:
        bars_per_day = 288
    embargo_bars = embargo_days * bars_per_day

    train_data = window_data.iloc[:train_idx].copy()
    validation_data = window_data.iloc[train_idx:validation_idx].copy()

    test_start_idx = validation_idx + embargo_bars
    if test_start_idx >= len(window_data):
        min_test_bars = max(int(len(window_data) * 0.05), 10)
        test_start_idx = max(validation_idx + 1, len(window_data) - min_test_bars)
        logger.warning(f"Embargo reduced: test_start_idx={test_start_idx}, len={len(window_data)}")
    test_data = window_data.iloc[test_start_idx:].copy()

    logger.info(
        f"Train {len(train_data)} | Val {len(validation_data)} | Test {len(test_data)} bars "
        f"(embargo {embargo_bars} bars)"
    )
    return train_data, validation_data, test_data


def maybe_apply_lstm_features(train_data, validation_data, test_data, output_dir: Path):
    """Mirror walk_forward.py lines 619-668."""
    lstm_config = config.get("indicators", {}).get("lstm_features", {})
    if not lstm_config.get("enabled", False):
        logger.info("LSTM features disabled in config; skipping")
        return train_data, validation_data, test_data

    lstm_params = {
        "lookback": lstm_config.get("lookback", 20),
        "hidden_size": lstm_config.get("hidden_size", 32),
        "num_layers": lstm_config.get("num_layers", 1),
        "output_size": lstm_config.get("output_size", 8),
        "pretrain_epochs": lstm_config.get("pretrain_epochs", 50),
        "pretrain_lr": lstm_config.get("pretrain_lr", 0.001),
        "pretrain_batch_size": lstm_config.get("pretrain_batch_size", 64),
        "pretrain_patience": lstm_config.get("pretrain_patience", 10),
        "pretrain_min_delta": lstm_config.get("pretrain_min_delta", 0.0001),
    }
    logger.info(f"Fitting LSTMFeatureGenerator: {lstm_params}")
    generator = LSTMFeatureGenerator(**lstm_params)
    checkpoint_path = str(output_dir / "lstm_autoencoder_checkpoint.pt")
    generator.fit(
        train_df=train_data,
        validation_df=validation_data,
        checkpoint_path=checkpoint_path
    )
    train_data = generator.transform(train_data)
    validation_data = generator.transform(validation_data)
    test_data = generator.transform(test_data)
    generator.save(str(output_dir / "lstm_generator.pkl"))
    logger.info(f"Added {generator.output_size} LSTM features")
    return train_data, validation_data, test_data


def maybe_apply_synthetic_bears(train_data: pd.DataFrame, output_dir: Path | None = None) -> pd.DataFrame:
    aug_config = config.get("augmentation", {}).get("synthetic_bears", {})
    if not aug_config.get("enabled", False):
        return train_data
    raw_train = extract_ohlcv_frame(train_data)
    augmented_raw, metadata = augment_with_synthetic_bears(
        raw_train,
        oversample_ratio=aug_config.get("oversample_ratio", 0.3),
        segment_length_pct=aug_config.get("segment_length_pct", 0.15),
        seed=int(config.get("seed", 42)),
        return_metadata=True,
    )
    train_data = process_technical_indicators(augmented_raw)
    if metadata and output_dir is not None:
        with open(output_dir / "synthetic_bears_report.json", "w") as f:
            json.dump(metadata, f, indent=2)
    logger.info(
        f"Augmented train data to {len(train_data)} rows with synthetic bears "
        f"before LSTM/normalization"
    )
    return train_data


def drop_redundant_columns(*dfs):
    """Mirror walk_forward.py lines 703-721."""
    cols_to_drop = ["volume", "Volume", "SMA", "EMA", "VWAP", "PSAR", "OBV", "VOLUME_NORM", "DOW", "position"]
    risk_config = config.get("risk_management", {})
    risk_enabled = risk_config.get("enabled", False)
    dynamic_sl_tp = risk_config.get("dynamic_sl_tp", {}).get("enabled", False)
    sl_atr = risk_config.get("stop_loss", {}).get("mode") == "atr"
    tp_atr = risk_config.get("take_profit", {}).get("mode") == "atr"
    needs_ohlc = risk_enabled or dynamic_sl_tp or sl_atr or tp_atr
    if not needs_ohlc:
        cols_to_drop.extend(
            ["open", "Open", "OPEN", "high", "low", "High", "Low", "HIGH", "LOW", "close", "Close", "CLOSE"]
        )
    for df in dfs:
        present = [c for c in cols_to_drop if c in df.columns]
        if present:
            df.drop(columns=present, inplace=True)


def build_feature_order(train_data: pd.DataFrame) -> dict:
    """Build the authoritative obs column layout by instantiating TradingEnv on the
    post-scaling, post-drop train frame. Live side uses this to assemble the obs
    vector in exactly the same order."""
    env = TradingEnv(
        train_data,
        initial_balance=config["environment"]["initial_balance"],
        transaction_cost=config["environment"].get("transaction_cost", 2.5),
    )
    indicators = list(env.technical_indicators)
    layout = {
        "close_norm": True,
        "indicators": indicators,
        "state_features": ["signed_exposure", "unrealized_pnl_norm", "time_in_position_norm", "drawdown_pct"],
        "observation_dim": int(env.observation_space.shape[0]),
    }
    expected_dim = 1 + len(indicators) + 4
    assert expected_dim == layout["observation_dim"], (
        f"Layout mismatch: computed {expected_dim} vs env {layout['observation_dim']}"
    )
    return layout


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def update_latest_symlink(output_dir: Path) -> None:
    latest = Path("models/current/latest")
    latest.parent.mkdir(parents=True, exist_ok=True)
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    target = os.path.relpath(output_dir.resolve(), latest.parent.resolve())
    latest.symlink_to(target, target_is_directory=True)
    logger.info(f"models/current/latest -> {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data",
        default=config.get("data", {}).get("csv_path", "data/NQ_live.csv"),
        help="Source CSV (OHLCV + timestamp)",
    )
    parser.add_argument("--output-dir", default=None, help="Bundle dir (default: models/current/<ts>)")
    parser.add_argument("--window-size", type=int, default=None, help="Trading days (default: config)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: config.seed)")
    parser.add_argument("--skip-training", action="store_true", help="Smoke-test data pipeline only")
    parser.add_argument("--no-symlink", action="store_true", help="Skip updating models/current/latest")
    parser.add_argument("--timesteps", type=int, default=None, help="Override initial_timesteps")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override max_iterations")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_global_seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"models/current/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Bundle dir: {output_dir}")

    logger.info(f"Loading data from {args.data}")
    full_data = load_tradingview_data(args.data)
    if full_data is None or len(full_data) == 0:
        logger.error("Data loading failed")
        sys.exit(1)
    logger.info(f"Loaded {len(full_data)} bars from {full_data.index[0]} to {full_data.index[-1]}")

    # Drop non-indicator leftover columns from the source CSV that would otherwise
    # pollute the scaler fit (e.g. NQ_live.csv has price_volatility/returns/source).
    # Live preprocess_bar never produces these, so they must not be in the training
    # feature set either.
    junk_cols = [c for c in ("source", "price_volatility", "returns", "Source") if c in full_data.columns]
    if junk_cols:
        logger.info(f"Dropping non-indicator source-CSV columns: {junk_cols}")
        full_data = full_data.drop(columns=junk_cols)

    if config.get("data", {}).get("market_hours_only", True):
        full_data = filter_market_hours(full_data)
        logger.info(f"After market-hours filter: {len(full_data)} bars")

    window_size = args.window_size or config.get("walk_forward", {}).get("window_size", 120)
    window_data = slice_last_window(full_data, window_size)

    train_data, validation_data, test_data = split_with_embargo(
        window_data,
        train_ratio=config["data"].get("train_ratio", 0.6),
        validation_ratio=config["data"].get("validation_ratio", 0.2),
        embargo_days=config["data"].get("embargo_days", 1),
    )

    train_data = maybe_apply_synthetic_bears(train_data, output_dir)

    train_data, validation_data, test_data = maybe_apply_lstm_features(
        train_data, validation_data, test_data, output_dir
    )

    cols_to_scale = get_standardized_column_names(train_data)
    scaler_type = config.get("normalization", {}).get("scaler_type", "robust")
    scaler, train_data, validation_data, test_data = scale_window(
        train_data=train_data,
        val_data=validation_data,
        test_data=test_data,
        cols_to_scale=cols_to_scale,
        feature_range=(-1, 1),
        window_folder=str(output_dir),
        scaler_type=scaler_type,
    )

    drop_redundant_columns(train_data, validation_data, test_data)

    feature_layout = build_feature_order(train_data)
    with open(output_dir / "feature_order.json", "w") as f:
        json.dump(feature_layout, f, indent=2)
    logger.info(
        f"feature_order.json: obs_dim={feature_layout['observation_dim']}, "
        f"{len(feature_layout['indicators'])} indicators"
    )

    manifest = {
        "timestamp": timestamp,
        "git_sha": git_sha(),
        "source_csv": args.data,
        "window_size_trading_days": window_size,
        "rollover_step_trading_days": config.get("walk_forward", {}).get("step_size", 24),
        "train_bars": len(train_data),
        "validation_bars": len(validation_data),
        "test_bars": len(test_data),
        "train_start": str(train_data.index[0]),
        "train_end": str(train_data.index[-1]),
        "validation_start": str(validation_data.index[0]),
        "validation_end": str(validation_data.index[-1]),
        "test_start": str(test_data.index[0]),
        "test_end": str(test_data.index[-1]),
        "scaler_type": scaler_type,
        "observation_dim": feature_layout["observation_dim"],
        "train_columns": train_data.columns.tolist(),
        "seed": seed,
        "recurrent": config.get("sequence_model", {}).get("enabled", False),
    }

    if args.skip_training:
        logger.info("--skip-training set; stopping after data pipeline")
        manifest["skipped_training"] = True
        with open(output_dir / "training_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        if not args.no_symlink:
            update_latest_symlink(output_dir)
        return

    eval_metric = config.get("training", {}).get("evaluation", {}).get("metric", "sortino")
    initial_timesteps = args.timesteps or config["training"].get("total_timesteps", 100_000)
    additional_timesteps = config["training"].get("additional_timesteps", 20_000)
    max_iterations = args.max_iterations or config["training"].get("max_iterations", 20)
    n_stagnant_loops = config["training"].get("n_stagnant_loops", 3)
    improvement_threshold = config["training"].get("improvement_threshold", 0.1)

    logger.info(
        f"Training: initial_timesteps={initial_timesteps}, max_iter={max_iterations}, "
        f"metric={eval_metric}, recurrent={manifest['recurrent']}"
    )

    model, val_results, _ = train_agent_iteratively(
        train_data=train_data,
        validation_data=validation_data,
        initial_timesteps=initial_timesteps,
        additional_timesteps=additional_timesteps,
        max_iterations=max_iterations,
        n_stagnant_loops=n_stagnant_loops,
        improvement_threshold=improvement_threshold,
        evaluation_metric=eval_metric,
        model_params=None,
        window_folder=str(output_dir),
        window_label="[current] ",
        prev_model_path=None,
    )

    test_results = evaluate_agent(model, test_data, deterministic=True)
    logger.info(
        f"Test: return={test_results.get('total_return_pct', 0):.2f}%, "
        f"sortino={test_results.get('sortino_ratio', 0):.2f}, "
        f"max_dd={test_results.get('max_drawdown', 0):.2f}%, "
        f"trades={test_results.get('trade_count', 0)}"
    )

    manifest["metrics"] = {
        "validation_return_pct": val_results.get("total_return_pct", 0),
        "validation_sortino": val_results.get("sortino_ratio", 0),
        "test_return_pct": test_results.get("total_return_pct", 0),
        "test_sortino": test_results.get("sortino_ratio", 0),
        "test_max_drawdown": test_results.get("max_drawdown", 0),
        "test_trades": test_results.get("trade_count", 0),
    }
    with open(output_dir / "training_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    if not args.no_symlink:
        update_latest_symlink(output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
