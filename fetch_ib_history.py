#!/usr/bin/env python3
"""Fetch historical NQ 5-minute bars from Interactive Brokers.

Uses ContFuture (continuous front-month) so we get a single stitched series
without contract-rollover bookkeeping. IB limits 5-minute historical requests
to roughly one month per call, so we pull in chunks walking backwards from
"now" until the requested window is covered.

Requires a running TWS / IB Gateway with API access enabled.

Usage:
    # Paper port (7497), last 365 days, overwrite data/NQ_live.csv
    python fetch_ib_history.py --days 365

    # Merge with the existing file instead of overwriting (dedup on timestamp)
    python fetch_ib_history.py --days 90 --merge

    # Live port (7496), custom output
    python fetch_ib_history.py --port 7496 --out data/NQ_fresh.csv --days 365

Output is CSV with columns: timestamp,open,high,low,close,volume — the format
`get_data.load_tradingview_data` / `walk_forward.load_tradingview_data` both
accept.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from ib_insync import ContFuture, IB, util

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fetch_ib_history")


def fetch_chunk(ib: IB, contract, end_dt: datetime, duration: str, bar_size: str, use_rth: int) -> pd.DataFrame:
    """Fetch one chunk of bars ending at end_dt (None = 'now'). Returns a DataFrame
    with columns [date, open, high, low, close, volume, ...]."""
    end_str = "" if end_dt is None else end_dt.strftime("%Y%m%d-%H:%M:%S")
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end_str,
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow="TRADES",
        useRTH=use_rth,
        formatDate=1,
        keepUpToDate=False,
    )
    if not bars:
        return pd.DataFrame()
    return util.df(bars)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="NQ", help="Futures root symbol")
    parser.add_argument("--exchange", default="CME", help="Exchange")
    parser.add_argument("--days", type=int, default=365, help="Total calendar days of history to pull")
    parser.add_argument("--chunk-days", type=int, default=28,
                        help="Days per reqHistoricalData call (IB caps 5-min ~1 month)")
    parser.add_argument("--bar-size", default="5 mins", help="IB bar size string")
    parser.add_argument("--use-rth", action=argparse.BooleanOptionalAction, default=True,
                        help="Regular trading hours only (default: True, matches training). "
                             "Pass --no-use-rth to include the overnight Globex session.")
    parser.add_argument("--out", default="data/NQ_live.csv", help="Output CSV path")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with existing file instead of overwriting; dedup on timestamp")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497, help="7497=paper, 7496=live")
    parser.add_argument("--client-id", type=int, default=17)
    args = parser.parse_args()

    ib = IB()
    logger.info(f"Connecting to IB at {args.host}:{args.port} (clientId={args.client_id})")
    try:
        ib.connect(args.host, args.port, clientId=args.client_id)
    except ConnectionRefusedError:
        logger.error(
            f"Connection refused on {args.host}:{args.port}. "
            f"Start TWS/Gateway in paper mode and enable API access."
        )
        sys.exit(1)
    logger.info("Connected.")

    contract = ContFuture(symbol=args.symbol, exchange=args.exchange)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        logger.error(f"Could not qualify ContFuture {args.symbol}@{args.exchange}")
        ib.disconnect()
        sys.exit(1)
    contract = qualified[0]
    logger.info(f"Using contract: {contract}")

    all_frames = []
    end_dt = None  # None => 'now'
    days_remaining = args.days
    while days_remaining > 0:
        chunk_days = min(args.chunk_days, days_remaining)
        duration = f"{chunk_days} D"
        end_label = "now" if end_dt is None else end_dt.strftime("%Y-%m-%d %H:%M")
        logger.info(f"Fetching {duration} ending {end_label}")
        try:
            df = fetch_chunk(ib, contract, end_dt, duration, args.bar_size, int(args.use_rth))
        except Exception as e:
            logger.error(f"reqHistoricalData failed: {e}")
            break

        if df.empty:
            logger.warning(f"Empty chunk at end={end_label}; stopping early")
            break

        logger.info(f"  got {len(df)} bars, range {df['date'].min()} -> {df['date'].max()}")
        all_frames.append(df)

        earliest = df["date"].min()
        if isinstance(earliest, pd.Timestamp):
            end_dt = earliest.to_pydatetime()
        else:
            end_dt = pd.Timestamp(earliest).to_pydatetime()

        days_remaining -= chunk_days
        ib.sleep(1.0)  # gentle pacing vs IB's 60-requests-per-10-minutes rule

    ib.disconnect()

    if not all_frames:
        logger.error("No data fetched; aborting")
        sys.exit(1)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.rename(columns={"date": "timestamp"})
    keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    combined = combined[[c for c in keep_cols if c in combined.columns]]
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    if combined["timestamp"].dt.tz is not None:
        combined["timestamp"] = combined["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.merge and out_path.exists():
        logger.info(f"Merging with existing {out_path}")
        existing = pd.read_csv(out_path)
        existing_ts = "timestamp" if "timestamp" in existing.columns else "time"
        existing = existing.rename(columns={existing_ts: "timestamp"})
        existing_cols = [c for c in keep_cols if c in existing.columns]
        existing = existing[existing_cols]
        existing["timestamp"] = pd.to_datetime(existing["timestamp"])
        before = len(existing)
        merged = pd.concat([existing, combined], ignore_index=True)
        merged = merged.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        merged.to_csv(out_path, index=False)
        logger.info(f"Wrote {len(merged)} rows to {out_path} (added {len(merged) - before} new)")
        final = merged
    else:
        combined.to_csv(out_path, index=False)
        logger.info(f"Wrote {len(combined)} rows to {out_path}")
        final = combined

    logger.info(f"Date range: {final['timestamp'].min()} -> {final['timestamp'].max()}")


if __name__ == "__main__":
    main()
