"""
Williams %R indicator module.
"""
import pandas as pd
import pandas_ta as ta
import logging

from indicators.utils import get_ohlcv_columns

logger = logging.getLogger(__name__)


def calculate_williams_r(df, length=14, target_col='WILLIAMS_R'):
    """
    Calculate Williams %R using pandas_ta library

    Args:
        df: DataFrame with price data
        length: Period for calculation (default: 14)
        target_col: Name of the output column (default: 'WILLIAMS_R')

    Returns:
        DataFrame: Original DataFrame with Williams %R column added
    """
    try:
        logger.info(f"Calculating {target_col} with length={length}")

        # Get column names
        cols = get_ohlcv_columns(df)
        if not all([cols['high'], cols['low'], cols['close']]):
            logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
            df[target_col] = -50
            return df

        # Calculate Williams %R - willr returns negative values in the range [-100, 0]
        williams_r = ta.willr(df[cols['high']], df[cols['low']], df[cols['close']], length=length)

        # Assign to dataframe
        df[target_col] = williams_r

        # Fill NaN values with default
        df[target_col] = df[target_col].fillna(-50)

        return df
    except Exception as e:
        logger.error(f"Error calculating Williams %R: {e}")
        if target_col not in df.columns:
            df[target_col] = -50
        return df 