"""
On-Balance Volume (OBV) indicator module.
"""
import pandas as pd
import pandas_ta as ta
import logging

from indicators.utils import get_ohlcv_columns

logger = logging.getLogger(__name__)


def calculate_obv(df, target_col='OBV'):
    """
    Calculate On-Balance Volume (OBV) using pandas_ta library

    Args:
        df: DataFrame with price data
        target_col: Name of the OBV output column (default: 'OBV')

    Returns:
        DataFrame: Original DataFrame with OBV column added
    """
    try:
        logger.info(f"Calculating {target_col}")

        # Get column names
        cols = get_ohlcv_columns(df)
        if cols['close'] is None:
            logger.error(f"Missing close column. Available: {df.columns.tolist()}")
            df[target_col] = 0
            return df

        # Check if volume column exists, if not create with default values
        if cols['volume'] is None:
            logger.warning("Volume column not found. Adding placeholder values.")
            df['volume'] = 1000
            cols['volume'] = 'volume'

        # Calculate OBV
        df[target_col] = ta.obv(df[cols['close']], df[cols['volume']])

        # Fill NaN values with defaults
        df[target_col] = df[target_col].fillna(0)

        return df
    except Exception as e:
        logger.error(f"Error calculating OBV: {e}")
        if target_col not in df.columns:
            df[target_col] = 0
        return df 