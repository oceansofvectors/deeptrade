"""
Volume Weighted Average Price (VWAP) indicator module.

VWAP is the ratio of cumulative typical price * volume to cumulative volume,
reset at the start of each trading day.
"""
import pandas as pd
import numpy as np
import logging

from indicators.utils import get_ohlcv_columns, get_column

logger = logging.getLogger(__name__)


def calculate_vwap(df, target_col='VWAP'):
    """
    Calculate Volume Weighted Average Price (VWAP)

    VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    where Typical Price = (High + Low + Close) / 3

    The calculation resets at the start of each trading day.

    Args:
        df: DataFrame with OHLCV data (must have high, low, close, volume columns)
        target_col: Name of the VWAP output column (default: 'VWAP')

    Returns:
        DataFrame: Original DataFrame with VWAP column added
    """
    try:
        logger.info(f"Calculating {target_col}")

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Get column names using utility function
        cols = get_ohlcv_columns(result_df)
        high_col = cols.get('high')
        low_col = cols.get('low')
        close_col = cols.get('close')
        volume_col = cols.get('volume')

        # Validate required columns exist
        if not all([high_col, low_col, close_col]):
            logger.error(f"Missing required OHLC columns. Available: {result_df.columns.tolist()}")
            result_df[target_col] = result_df[close_col] if close_col else 0
            return result_df

        # Handle missing volume
        if volume_col is None:
            logger.warning("Volume column not found. Adding placeholder values.")
            result_df['volume'] = 1000
            volume_col = 'volume'

        # Extract date for daily reset
        if isinstance(df.index, pd.DatetimeIndex):
            # DatetimeIndex - extract date directly
            dates = df.index.normalize()
        else:
            # Try to convert index to datetime
            try:
                dates = pd.to_datetime(df.index).normalize()
            except Exception as e:
                logger.warning(f"Could not parse dates from index: {e}. Using single session.")
                dates = pd.Series([0] * len(df), index=df.index)

        # Calculate Typical Price
        typical_price = (result_df[high_col] + result_df[low_col] + result_df[close_col]) / 3

        # Calculate TPV (Typical Price * Volume)
        tpv = typical_price * result_df[volume_col]

        # Create a temporary DataFrame for groupby operations
        calc_df = pd.DataFrame({
            'date': dates,
            'tpv': tpv,
            'volume': result_df[volume_col]
        }, index=df.index)

        # Cumulative sums within each day
        calc_df['cum_tpv'] = calc_df.groupby('date')['tpv'].cumsum()
        calc_df['cum_vol'] = calc_df.groupby('date')['volume'].cumsum()

        # VWAP = cumulative TPV / cumulative Volume
        result_df[target_col] = calc_df['cum_tpv'] / calc_df['cum_vol']

        # Handle edge cases
        result_df[target_col] = result_df[target_col].replace([np.inf, -np.inf], np.nan)
        result_df[target_col] = result_df[target_col].fillna(result_df[close_col])

        return result_df

    except Exception as e:
        logger.error(f"Error calculating VWAP: {e}")
        import traceback
        traceback.print_exc()

        # Add the column with default values in case of error
        close_col = get_column(df, 'close')
        if target_col not in df.columns:
            df = df.copy()
            df[target_col] = df[close_col] if close_col else 0
        return df
