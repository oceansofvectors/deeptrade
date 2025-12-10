"""
Commodity Channel Index (CCI) indicator module.
"""
import pandas as pd
import pandas_ta as ta
import logging

from indicators.utils import get_ohlcv_columns

logger = logging.getLogger(__name__)


def calculate_cci(df, length=20, target_col='CCI'):
    """
    Calculate Commodity Channel Index (CCI) using pandas_ta library

    Args:
        df: DataFrame with price data
        length: Period for CCI calculation (default: 20)
        target_col: Name of the output column (default: 'CCI')

    Returns:
        DataFrame: Original DataFrame with CCI column added
    """
    try:
        logger.info(f"Calculating {target_col} with length={length}")

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Get column names
        cols = get_ohlcv_columns(result_df)
        if not all([cols['high'], cols['low'], cols['close']]):
            logger.error(f"Missing required columns. Available: {result_df.columns.tolist()}")
            result_df[target_col] = 0.0
            return result_df

        # Calculate CCI
        cci = ta.cci(result_df[cols['high']], result_df[cols['low']], result_df[cols['close']], length=length)

        # Add to DataFrame
        result_df[target_col] = cci

        # Fill NaN values with default (zero)
        result_df[target_col] = result_df[target_col].fillna(0.0)

        return result_df
    except Exception as e:
        logger.error(f"Error calculating CCI: {e}")
        if target_col not in df.columns:
            df[target_col] = 0.0
        return df 