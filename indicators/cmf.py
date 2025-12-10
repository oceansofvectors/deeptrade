"""
Chaikin Money Flow (CMF) indicator module.
"""
import pandas as pd
import pandas_ta as ta
import logging

from indicators.utils import get_ohlcv_columns

logger = logging.getLogger(__name__)


def calculate_cmf(df, length=20, target_col='CMF'):
    """
    Calculate Chaikin Money Flow (CMF) using pandas_ta library

    Args:
        df: DataFrame with price and volume data
        length: Period for CMF calculation (default: 20)
        target_col: Name of the output column (default: 'CMF')

    Returns:
        DataFrame: Original DataFrame with CMF column added
    """
    try:
        logger.info(f"Calculating {target_col} with length={length}")

        # Get column names
        cols = get_ohlcv_columns(df)
        if not all([cols['high'], cols['low'], cols['close']]):
            logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
            df[target_col] = 0
            return df

        # Check if volume column exists, if not create with default values
        if cols['volume'] is None:
            logger.warning("Volume column not found. Adding placeholder values.")
            df['volume'] = 1000
            cols['volume'] = 'volume'

        # Calculate CMF
        df[target_col] = ta.cmf(df[cols['high']], df[cols['low']], df[cols['close']], df[cols['volume']], length=length)

        # Fill NaN values with defaults
        df[target_col] = df[target_col].fillna(0)

        return df
    except Exception as e:
        logger.error(f"Error calculating CMF: {e}")
        if target_col not in df.columns:
            df[target_col] = 0
        return df 