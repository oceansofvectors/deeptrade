"""
Data utility functions for the AlgoTrader2 system.

This module contains shared data processing utilities to avoid circular dependencies.
"""

# Standard library imports
import logging
from collections.abc import Mapping
from datetime import time

# Third-party imports
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

EASTERN = pytz.timezone("US/Eastern")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


def market_hours_only_enabled(settings: Mapping | None = None) -> bool:
    """Return whether offline training/evaluation should use NYSE RTH only.

    The repo treats this as a single source-of-truth boolean under
    ``data.market_hours_only``. Missing keys default to enabled so all
    training/evaluation paths agree on the safer market-hours-only behavior.
    """
    if settings is None:
        return True
    data_config = settings.get("data", {})
    if not isinstance(data_config, Mapping):
        return True
    return bool(data_config.get("market_hours_only", True))


def is_rth_bar(ts) -> bool:
    """Return True if the given timestamp lies within NYSE regular trading
    hours (9:30-16:00 Eastern, Mon-Fri). Accepts any tz-aware or tz-naive
    datetime/pd.Timestamp; naive values are assumed to be UTC.

    This is the single-timestamp counterpart to `filter_market_hours` — both
    must use the same spec so training and live agree on what "a bar" means.
    """
    if ts is None:
        return False
    if not hasattr(ts, "tzinfo") or ts.tzinfo is None:
        ts = pd.Timestamp(ts).tz_localize("UTC")
    et = ts.astimezone(EASTERN)
    if et.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    t = et.time()
    return MARKET_OPEN <= t <= MARKET_CLOSE


def filter_market_hours(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the data to only include NYSE market hours (9:30 AM to 4:00 PM ET, Monday to Friday).

    Args:
        data: DataFrame with DatetimeIndex in UTC

    Returns:
        DataFrame: Filtered data containing only market hours
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        logger.error("Data index is not a DatetimeIndex, cannot filter market hours")
        return data

    # Make a copy to avoid modifying the original
    filtered_data = data.copy()

    # Convert UTC times to Eastern Time
    eastern = pytz.timezone('US/Eastern')

    # Ensure the index is timezone-aware
    if filtered_data.index.tz is None:
        filtered_data.index = filtered_data.index.tz_localize('UTC')

    # Convert to Eastern Time
    filtered_data.index = filtered_data.index.tz_convert(eastern)

    # Filter for weekdays (Monday=0, Friday=4)
    weekday_mask = (filtered_data.index.dayofweek >= 0) & (filtered_data.index.dayofweek <= 4)

    # Filter for market hours (9:30 AM to 4:00 PM ET)
    market_open = time(9, 30)
    market_close = time(16, 0)

    hours_mask = (
        (filtered_data.index.time >= market_open) &
        (filtered_data.index.time <= market_close)
    )

    # Apply both filters
    market_hours_mask = weekday_mask & hours_mask
    filtered_data = filtered_data.loc[market_hours_mask]

    # Convert back to UTC for consistency with the rest of the system
    filtered_data.index = filtered_data.index.tz_convert('UTC')

    # Log filtering results
    filtered_pct = (len(filtered_data) / len(data)) * 100
    logger.info(f"Filtered data to NYSE RTH only: {len(filtered_data)} / {len(data)} rows ({filtered_pct:.2f}%)")

    return filtered_data
