#!/usr/bin/env python3
"""Offline replay harness for the live trading pipeline.

Feeds rows of a CSV into the modernized ModelTrader as if they were streaming
5-minute bars, runs preprocess_bar -> get_prediction, and simulates fills so
that the live obs state (position, unrealized_pnl, time_in_position, drawdown)
evolves like it would in production — all without touching IB.

Usage:
    python scripts/replay_live.py --bundle models/current/latest \
        --data data/NQ_live.csv --warmup 250 --max-bars 1000
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trading.model_trader import ModelTrader  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("replay_live")
logger.setLevel(logging.INFO)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = "timestamp" if "timestamp" in df.columns else "time"
    if pd.api.types.is_numeric_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col], unit="s")
    else:
        df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).set_index(ts_col)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_simulated_fill(mt: ModelTrader, action: int, current_close: float) -> None:
    """Map action -> current_position transition like execute_trade does,
    but without touching IB. Updates both ModelTrader.current_position and
    the live obs state via _update_trade_state (indirectly, since preprocess_bar
    already called _update_trade_state before this — the next bar will see
    the new position)."""
    if action == 0:  # long
        target = 1
    elif action == 1:  # short
        target = -1
    else:  # 2 = flat
        target = 0
    if target != mt.current_position:
        mt.current_position = target
        mt.expected_position = abs(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", required=True, help="Path to bundle dir (e.g. models/current/latest)")
    parser.add_argument("--data", default="data/NQ_live.csv", help="Source CSV with timestamp,OHLCV")
    parser.add_argument("--warmup", type=int, default=250,
                        help="Number of bars to feed before evaluating predictions")
    parser.add_argument("--max-bars", type=int, default=None,
                        help="Stop after N bars total (default: all)")
    parser.add_argument("--start", default=None, help="ISO timestamp to start from")
    parser.add_argument("--end", default=None, help="ISO timestamp to stop at")
    parser.add_argument("--report", default=None, help="Write JSON summary here")
    args = parser.parse_args()

    logger.info(f"Loading bundle: {args.bundle}")
    mt = ModelTrader(
        ib_instance=None,
        model_path=args.bundle,
        state_file_path="/tmp/replay_state.pkl",
        use_risk_management=False,
    )
    if mt.model is None:
        logger.error("Bundle has no model; cannot run replay.")
        sys.exit(1)
    if mt.feature_layout is None:
        logger.error("Bundle missing feature_order.json; cannot run replay.")
        sys.exit(1)

    logger.info(f"Loading data: {args.data}")
    df = load_csv(args.data)
    if args.start:
        df = df[df.index >= pd.Timestamp(args.start)]
    if args.end:
        df = df[df.index <= pd.Timestamp(args.end)]
    if args.max_bars:
        df = df.iloc[: args.max_bars]
    logger.info(f"Replaying {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    action_counts = Counter()
    position_flips = 0
    first_eval_idx = args.warmup
    preprocess_failures = 0
    pnl_history = []
    obs_sample = None

    for i, (ts, row) in enumerate(df.iterrows()):
        bar = {
            "time": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0.0)),
        }
        obs = mt.preprocess_bar(bar)
        if obs is None:
            preprocess_failures += 1
            continue
        if i < first_eval_idx:
            continue

        if obs_sample is None:
            obs_sample = obs.tolist()

        action = mt.get_prediction(obs)
        if action is None:
            continue
        action_counts[int(action)] += 1
        prev_pos = mt.current_position
        apply_simulated_fill(mt, int(action), bar["close"])
        if mt.current_position != prev_pos:
            position_flips += 1
        pnl_history.append(mt.unrealized_pnl)

    summary = {
        "bundle": args.bundle,
        "data": args.data,
        "bars_replayed": len(df),
        "warmup_bars": args.warmup,
        "preprocess_failures": preprocess_failures,
        "evaluated_predictions": sum(action_counts.values()),
        "action_counts": {str(k): int(v) for k, v in action_counts.items()},
        "position_flips": position_flips,
        "final_position": mt.current_position,
        "obs_dim": int(mt.feature_layout["observation_dim"]),
        "obs_sample_first5": obs_sample[:5] if obs_sample else None,
        "obs_sample_last5": obs_sample[-5:] if obs_sample else None,
        "unrealized_pnl_final": mt.unrealized_pnl,
        "unrealized_pnl_min": float(min(pnl_history)) if pnl_history else 0.0,
        "unrealized_pnl_max": float(max(pnl_history)) if pnl_history else 0.0,
        "session_peak_equity": mt.session_peak_equity,
        "drawdown_pct_final": mt.drawdown_pct,
        "time_in_position_final": mt.time_in_position,
    }

    print(json.dumps(summary, indent=2, default=str))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Summary written to {args.report}")

    if position_flips == 0:
        logger.warning("No position changes during replay — model may be stuck on one action.")
    if preprocess_failures > 0:
        logger.warning(f"{preprocess_failures} preprocess failures during replay.")


if __name__ == "__main__":
    main()
