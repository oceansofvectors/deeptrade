"""
Exponential Moving Average (EMA) indicator module.
"""
import pandas as pd
import pandas_ta as ta
import logging

from indicators.utils import get_column

logger = logging.getLogger(__name__)


def calculate_ema(df, length=20, target_col='EMA'):
    """
    Calculate Exponential Moving Average (EMA) using pandas_ta library

    Args:
        df: DataFrame with price data
        length: Period for EMA calculation (default: 20)
        target_col: Name of the output column (default: 'EMA')

    Returns:
        DataFrame: Original DataFrame with EMA column added
    """
    try:
        logger.info(f"Calculating {target_col} with length={length}")

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Get close column name
        close_col = get_column(result_df, 'close')
        if close_col is None:
            logger.error(f"No close column found. Available: {result_df.columns.tolist()}")
            result_df[target_col] = 0
            return result_df

        # Ensure close is float for consistent comparison
        result_df[close_col] = result_df[close_col].astype(float)

        # Calculate EMA
        result_df[target_col] = ta.ema(result_df[close_col], length=length)

        # Fill NaN values with close price for early periods
        result_df[target_col] = result_df[target_col].fillna(result_df[close_col])

        return result_df
    except Exception as e:
        logger.error(f"Error calculating EMA: {e}")
        close_col = get_column(df, 'close')
        if target_col not in df.columns:
            df[target_col] = df[close_col] if close_col else 0
        return df 