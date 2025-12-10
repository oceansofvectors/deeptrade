"""
Relative Strength Index (RSI) indicator module.
"""
import pandas as pd
import pandas_ta as ta
import logging

from indicators.utils import get_column

logger = logging.getLogger(__name__)


def calculate_rsi(df, length=14, target_col='RSI'):
    """
    Calculate Relative Strength Index (RSI) using pandas_ta library

    Args:
        df: DataFrame with price data
        length: Period for RSI calculation (default: 14)
        target_col: Name of the output column (default: 'RSI')

    Returns:
        DataFrame: Original DataFrame with RSI column added
    """
    try:
        logger.info(f"Calculating {target_col} with length={length}")

        close_col = get_column(df, 'close')
        if close_col is None:
            logger.error(f"No close column found. Available: {df.columns.tolist()}")
            df[target_col] = 50
            return df

        df[target_col] = ta.rsi(df[close_col], length=length)

        # Fill NaN values with neutral RSI value
        df[target_col] = df[target_col].fillna(50)

        return df
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        if target_col not in df.columns:
            df[target_col] = 50
        return df 