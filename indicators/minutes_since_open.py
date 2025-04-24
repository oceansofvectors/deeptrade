"""
Minutes Since Market Open indicator module.
"""
import pandas as pd
import numpy as np
import pytz
import logging

logger = logging.getLogger(__name__)

def calculate_minutes_since_open(df, sin_col='MSO_SIN', cos_col='MSO_COS', 
                                market_open_hour=9, market_open_minute=30,
                                market_duration_minutes=390):  # 6.5 hours = 390 minutes
    """
    Calculate Minutes Since Market Open indicator with circular encoding
    
    Args:
        df: DataFrame with datetime index
        sin_col: Name of the sine encoding column (default: 'MSO_SIN')
        cos_col: Name of the cosine encoding column (default: 'MSO_COS')
        market_open_hour: Hour when market opens in ET (default: 9)
        market_open_minute: Minute when market opens in ET (default: 30)
        market_duration_minutes: Duration of market session in minutes (default: 390)
        
    Returns:
        DataFrame: Original DataFrame with minutes since open columns added
    """
    try:
        eastern = pytz.timezone('US/Eastern')
        
        # Create a UTC tz-aware DatetimeIndex from df.index, then convert to Eastern
        idx = pd.to_datetime(df.index, utc=True)
        time_et = idx.tz_convert(eastern)
        
        # Calculate minutes since market open (9:30 AM ET)
        minutes_since_open = (time_et.hour - market_open_hour) * 60 + time_et.minute - market_open_minute
        
        # Handle times before market open (negative values)
        minutes_since_open = np.maximum(minutes_since_open, 0)
        
        # Handle times after market close (> market_duration_minutes minutes)
        minutes_since_open = np.minimum(minutes_since_open, market_duration_minutes)
        
        # Normalize to [0, 1] range
        minutes_since_open_norm = minutes_since_open / market_duration_minutes
        
        # Convert to sine and cosine representation (circular encoding)
        df[sin_col] = np.sin(2 * np.pi * minutes_since_open_norm)
        df[cos_col] = np.cos(2 * np.pi * minutes_since_open_norm)
        
        return df
    except Exception as e:
        logger.error(f"Error calculating Minutes Since Open: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Add the columns with default values in case of error
        if sin_col not in df.columns:
            df[sin_col] = 0
        if cos_col not in df.columns:
            df[cos_col] = 1  # cos(0) = 1
        return df 