"""
Minutes Since Midnight indicator module (24-hour time encoding).
"""
import pandas as pd
import numpy as np
import pytz
import logging

logger = logging.getLogger(__name__)

def calculate_minutes_since_open(df, sin_col='MSO_SIN', cos_col='MSO_COS',
                                market_open_hour=9, market_open_minute=30,
                                market_duration_minutes=390):  # Unused, kept for backwards compatibility
    """
    Calculate time-of-day indicator with circular encoding for 24-hour data.

    Uses minutes since midnight (0-1440) for full 24-hour cycle encoding,
    allowing the model to learn patterns across all trading sessions
    (pre-market, regular hours, after-hours, overnight).

    Args:
        df: DataFrame with datetime index
        sin_col: Name of the sine encoding column (default: 'MSO_SIN')
        cos_col: Name of the cosine encoding column (default: 'MSO_COS')
        market_open_hour: Unused, kept for backwards compatibility
        market_open_minute: Unused, kept for backwards compatibility
        market_duration_minutes: Unused, kept for backwards compatibility

    Returns:
        DataFrame: Original DataFrame with time-of-day columns added
    """
    try:
        eastern = pytz.timezone('US/Eastern')

        # Create a UTC tz-aware DatetimeIndex from df.index, then convert to Eastern
        idx = pd.to_datetime(df.index, utc=True)
        time_et = idx.tz_convert(eastern)

        # Calculate minutes since midnight (0-1439)
        minutes_since_midnight = time_et.hour * 60 + time_et.minute

        # Normalize to [0, 1) range for full 24-hour cycle
        minutes_norm = minutes_since_midnight / 1440.0  # 24 * 60 = 1440

        # Convert to sine and cosine representation (circular encoding)
        # This creates a smooth cycle where 23:59 is close to 00:00
        df[sin_col] = np.sin(2 * np.pi * minutes_norm)
        df[cos_col] = np.cos(2 * np.pi * minutes_norm)

        return df
    except Exception as e:
        logger.error(f"Error calculating Minutes Since Midnight: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Add the columns with default values in case of error
        if sin_col not in df.columns:
            df[sin_col] = 0
        if cos_col not in df.columns:
            df[cos_col] = 1  # cos(0) = 1
        return df 