"""
Average Directional Index (ADX) indicator module.
"""
import pandas as pd
import pandas_ta as ta
import logging

from indicators.utils import get_ohlcv_columns

logger = logging.getLogger(__name__)


def calculate_adx(df, length=14, adx_col='ADX'):
    """
    Calculate Average Directional Index (ADX) using pandas_ta library

    Args:
        df: DataFrame with price data
        length: Period for ADX calculation (default: 14)
        adx_col: Name of the ADX output column (default: 'ADX')

    Returns:
        DataFrame: Original DataFrame with ADX column added
    """
    try:
        logger.info(f"Calculating {adx_col} with length={length}")

        # Get column names
        cols = get_ohlcv_columns(df)
        if not all([cols['high'], cols['low'], cols['close']]):
            logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
            df[adx_col] = 0.0
            return df

        # Calculate ADX
        adx_result = ta.adx(df[cols['high']], df[cols['low']], df[cols['close']], length=length)
        if adx_result is None or adx_result.empty:
            logger.info(
                "ADX warmup incomplete for %d rows; emitting zero-filled ADX column",
                len(df),
            )
            df[adx_col] = 0.0
            return df

        # Extract ADX value - handle different column naming conventions
        if f'ADX_{length}' in adx_result.columns:
            df[adx_col] = adx_result[f'ADX_{length}']
        else:
            df[adx_col] = adx_result.iloc[:, 0]

        # Fill NaN values with defaults
        df[adx_col] = df[adx_col].fillna(0.0)

        return df
    except Exception as e:
        logger.warning(f"Error calculating ADX, falling back to zeros: {e}")
        if adx_col not in df.columns:
            df[adx_col] = 0.0
        return df
