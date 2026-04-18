"""
Synthetic bear episode generation for training.

The goal is to augment training windows with contiguous bearish episodes that
preserve time-series structure. These episodes should be created from raw OHLCV
data before indicators / LSTM features are recomputed upstream.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_TYPES = ("mirrored_selloff", "accelerated_downtrend", "panic_crash")
REGIME_WEIGHTS = np.array([0.45, 0.40, 0.15], dtype=float)


def extract_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy containing only canonical OHLCV columns."""
    col_map: Dict[str, str] = {}
    for target in ("open", "high", "low", "close", "volume"):
        for col in df.columns:
            if col.lower() == target:
                col_map[target] = col
                break
    required = ("open", "high", "low", "close", "volume")
    missing = [col for col in required if col not in col_map]
    if missing:
        raise KeyError(f"Missing OHLCV columns for synthetic augmentation: {missing}")

    raw = pd.DataFrame(index=df.index.copy())
    for target in required:
        raw[target] = pd.to_numeric(df[col_map[target]], errors="coerce")
    raw = raw.dropna(subset=list(required))
    return raw


def _infer_bar_delta(index: pd.Index) -> pd.Timedelta | None:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return None
    diffs = pd.Series(index[1:] - index[:-1])
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None
    return diffs.mode().iloc[0]


def _episode_index(source_index: pd.Index, end_anchor, segment_length: int, bar_delta, offset_bars: int) -> pd.Index:
    if isinstance(source_index, pd.DatetimeIndex) and bar_delta is not None:
        start = pd.Timestamp(end_anchor) + bar_delta * (offset_bars + 1)
        return pd.date_range(start=start, periods=segment_length, freq=bar_delta)
    start = int(len(source_index)) + offset_bars
    return pd.RangeIndex(start=start, stop=start + segment_length)


def _ensure_ohlc_consistency(segment: pd.DataFrame) -> pd.DataFrame:
    body_max = segment[["open", "close"]].max(axis=1)
    body_min = segment[["open", "close"]].min(axis=1)
    segment["high"] = np.maximum(segment["high"], body_max)
    segment["low"] = np.minimum(segment["low"], body_min)
    segment["low"] = np.maximum(segment["low"], 0.01)
    segment["open"] = np.maximum(segment["open"], 0.01)
    segment["high"] = np.maximum(segment["high"], 0.01)
    segment["close"] = np.maximum(segment["close"], 0.01)
    return segment


def _mirror_selloff(segment: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    transformed = segment.copy()
    anchor = float(segment["close"].iloc[0])
    for col in ("open", "high", "low", "close"):
        transformed[col] = 2.0 * anchor - segment[col].astype(float)
    drift = np.linspace(0.0, anchor * rng.uniform(0.002, 0.01), len(segment))
    for col in ("open", "high", "low", "close"):
        transformed[col] = transformed[col] - drift
    transformed["volume"] = segment["volume"].astype(float) * rng.uniform(1.2, 1.6, len(segment))
    return _ensure_ohlc_consistency(transformed)


def _build_bear_close_path(segment: pd.DataFrame, rng: np.random.Generator, *, vol_mult: float, crash: bool) -> np.ndarray:
    source_close = segment["close"].astype(float).to_numpy()
    anchor = max(source_close[0], 0.01)
    returns = np.diff(source_close) / np.clip(source_close[:-1], 0.01, None)
    abs_returns = np.abs(returns)
    if abs_returns.size == 0:
        abs_returns = np.array([0.001], dtype=float)

    closes = [anchor]
    crash_step = rng.integers(1, len(segment) - 1) if crash and len(segment) > 4 else None
    for i in range(1, len(segment)):
        scale = abs_returns[min(i - 1, len(abs_returns) - 1)]
        downward = scale * rng.uniform(1.1, vol_mult)
        noise = rng.normal(0.0, scale * 0.25)
        bounce = scale * rng.uniform(0.15, 0.5) if rng.random() < 0.12 else 0.0
        step_return = -downward + noise + bounce
        if crash_step is not None and i == crash_step:
            step_return -= rng.uniform(0.01, 0.035)
        closes.append(max(closes[-1] * (1.0 + step_return), 0.01))
    return np.array(closes, dtype=float)


def _reconstruct_ohlcv(segment: pd.DataFrame, close_path: np.ndarray, rng: np.random.Generator, *, vol_scale: float, volume_scale: Tuple[float, float]) -> pd.DataFrame:
    transformed = pd.DataFrame(index=segment.index.copy())
    base_range = (segment["high"].astype(float) - segment["low"].astype(float)).to_numpy()
    base_range = np.maximum(base_range, np.maximum(segment["close"].astype(float).to_numpy() * 0.0005, 0.25))

    opens = np.empty_like(close_path)
    highs = np.empty_like(close_path)
    lows = np.empty_like(close_path)
    volumes = segment["volume"].astype(float).to_numpy().copy()

    opens[0] = close_path[0] * (1.0 + rng.normal(0.0, 0.0008))
    for i in range(1, len(close_path)):
        gap = rng.normal(-0.0004 * vol_scale, 0.0008 * vol_scale)
        opens[i] = max(close_path[i - 1] * (1.0 + gap), 0.01)

    for i, close in enumerate(close_path):
        local_range = base_range[min(i, len(base_range) - 1)] * rng.uniform(1.0, vol_scale)
        wick_up = local_range * rng.uniform(0.15, 0.55)
        wick_down = local_range * rng.uniform(0.35, 0.9)
        highs[i] = max(opens[i], close) + wick_up
        lows[i] = max(min(opens[i], close) - wick_down, 0.01)

        neg_move = max(0.0, (opens[i] - close) / max(opens[i], 0.01))
        vol_mult = rng.uniform(*volume_scale) * (1.0 + neg_move * 12.0)
        volumes[i] = volumes[i] * vol_mult

    transformed["open"] = opens
    transformed["high"] = highs
    transformed["low"] = lows
    transformed["close"] = close_path
    transformed["volume"] = volumes
    return _ensure_ohlc_consistency(transformed)


def _accelerated_downtrend(segment: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    close_path = _build_bear_close_path(segment, rng, vol_mult=1.45, crash=False)
    return _reconstruct_ohlcv(segment, close_path, rng, vol_scale=1.35, volume_scale=(1.1, 1.45))


def _panic_crash(segment: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    close_path = _build_bear_close_path(segment, rng, vol_mult=1.9, crash=True)
    return _reconstruct_ohlcv(segment, close_path, rng, vol_scale=1.7, volume_scale=(1.25, 2.0))


def _regime_transform(segment: pd.DataFrame, regime_type: str, rng: np.random.Generator) -> pd.DataFrame:
    if regime_type == "mirrored_selloff":
        return _mirror_selloff(segment, rng)
    if regime_type == "accelerated_downtrend":
        return _accelerated_downtrend(segment, rng)
    if regime_type == "panic_crash":
        return _panic_crash(segment, rng)
    raise ValueError(f"Unknown synthetic bear regime type: {regime_type}")


def _return_stats(close_series: pd.Series) -> Dict[str, float]:
    returns = close_series.pct_change().dropna()
    if returns.empty:
        return {"mean_return": 0.0, "volatility": 0.0, "worst_bar": 0.0}
    return {
        "mean_return": float(returns.mean()),
        "volatility": float(returns.std(ddof=0)),
        "worst_bar": float(returns.min()),
    }


def augment_with_synthetic_bears(
    train_df: pd.DataFrame,
    oversample_ratio: float = 0.3,
    segment_length_pct: float = 0.15,
    seed: int = 42,
    return_metadata: bool = False,
):
    """
    Create contiguous synthetic bear episodes from raw OHLCV training data.

    The augmented output keeps the original rows first and appends synthetic
    episodes after them. No row-level shuffling is performed.
    """
    raw_df = extract_ohlcv_frame(train_df)
    n = len(raw_df)
    if n < 20:
        logger.warning("Training data too short for synthetic augmentation, skipping")
        return (raw_df, None) if return_metadata else raw_df

    segment_length = max(10, int(n * segment_length_pct))
    num_synthetic_bars = max(0, int(round(n * oversample_ratio)))
    if num_synthetic_bars < 10:
        metadata = {
            "enabled": False,
            "seed": int(seed),
            "original_bars": int(n),
            "synthetic_bars": 0,
            "target_synthetic_bars": int(num_synthetic_bars),
            "augmented_bars": int(n),
            "oversample_ratio": float(oversample_ratio),
            "segment_length_pct": float(segment_length_pct),
            "num_segments": 0,
            "regime_counts": {regime: 0 for regime in REGIME_TYPES},
            "segments": [],
            "real_close_stats": _return_stats(raw_df["close"]),
        }
        logger.info(
            "Skipping synthetic bear augmentation: target bars=%d is below minimum segment length",
            num_synthetic_bars,
        )
        return (raw_df, metadata) if return_metadata else raw_df

    num_segments = max(1, math.ceil(num_synthetic_bars / segment_length))
    rng = np.random.default_rng(seed=seed)
    bar_delta = _infer_bar_delta(raw_df.index)

    logger.info(
        f"Generating {num_segments} synthetic bear segments "
        f"({segment_length} bars each, ~{num_synthetic_bars} total bars)"
    )

    augmented_segments: List[pd.DataFrame] = []
    segment_reports: List[Dict[str, object]] = []
    if n - min(segment_length, num_synthetic_bars) <= 0:
        return (raw_df, None) if return_metadata else raw_df

    synthetic_offset = 0
    synthetic_bars_created = 0
    for episode_idx in range(num_segments):
        remaining_bars = num_synthetic_bars - synthetic_bars_created
        current_length = min(segment_length, remaining_bars)
        if current_length < 10:
            break

        max_start = n - current_length
        if max_start <= 0:
            break

        start = int(rng.integers(0, max_start + 1))
        source = raw_df.iloc[start:start + current_length].copy()
        regime_type = str(rng.choice(REGIME_TYPES, p=REGIME_WEIGHTS))
        synthetic = _regime_transform(source, regime_type, rng)
        synthetic.index = _episode_index(
            raw_df.index,
            raw_df.index[-1],
            len(source),
            bar_delta,
            synthetic_offset,
        )
        augmented_segments.append(synthetic)
        synthetic_offset += len(source)
        synthetic_bars_created += len(source)

        segment_reports.append({
            "episode_id": episode_idx + 1,
            "regime_type": regime_type,
            "source_start": str(source.index[0]),
            "source_end": str(source.index[-1]),
            "synthetic_start": str(synthetic.index[0]),
            "synthetic_end": str(synthetic.index[-1]),
            "bars": int(len(source)),
            "source_stats": _return_stats(source["close"]),
            "synthetic_stats": _return_stats(synthetic["close"]),
        })

    if not augmented_segments:
        return (raw_df, None) if return_metadata else raw_df

    result = pd.concat([raw_df] + augmented_segments, axis=0)
    result = result.sort_index(kind="stable")

    metadata = {
        "enabled": True,
        "seed": int(seed),
        "original_bars": int(n),
        "synthetic_bars": int(sum(len(seg) for seg in augmented_segments)),
        "target_synthetic_bars": int(num_synthetic_bars),
        "augmented_bars": int(len(result)),
        "oversample_ratio": float(oversample_ratio),
        "segment_length_pct": float(segment_length_pct),
        "num_segments": int(len(augmented_segments)),
        "regime_counts": {
            regime: int(sum(1 for report in segment_reports if report["regime_type"] == regime))
            for regime in REGIME_TYPES
        },
        "segments": segment_reports,
        "real_close_stats": _return_stats(raw_df["close"]),
        "synthetic_close_stats": _return_stats(pd.concat([seg["close"] for seg in augmented_segments], axis=0)),
    }

    logger.info(
        f"Augmented training data: {n} -> {len(result)} rows "
        f"(+{metadata['synthetic_bars']} synthetic, no row shuffling)"
    )

    if return_metadata:
        return result, metadata
    return result
