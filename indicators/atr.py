"""
Average True Range (ATR) indicator module.
"""
import pandas as pd
import pandas_ta as ta
import logging

from indicators.utils import get_ohlcv_columns

logger = logging.getLogger(__name__)


def calculate_atr(df, length=14, target_col='ATR'):
    """
    Calculate Average True Range (ATR) using pandas_ta library

    Args:
        df: DataFrame with price data
        length: Period for ATR calculation (default: 14)
        target_col: Name of the output column (default: 'ATR')

    Returns:
        DataFrame: Original DataFrame with ATR column added
    """
    try:
        logger.info(f"Calculating {target_col} with length={length}")

        # Get column names
        cols = get_ohlcv_columns(df)
        if not all([cols['high'], cols['low'], cols['close']]):
            logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
            df[target_col] = 0
            return df

        # Calculate ATR using pandas_ta
        atr = ta.atr(df[cols['high']], df[cols['low']], df[cols['close']], length=length)

        # Handle if atr is returned as a DataFrame
        if isinstance(atr, pd.DataFrame):
            if f'ATR_{length}' in atr.columns:
                atr = atr[f'ATR_{length}']
            else:
                atr = atr.iloc[:, 0]

        df[target_col] = atr

        # Fill NaN values with default
        df[target_col] = df[target_col].fillna(0)

        return df
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        if target_col not in df.columns:
            df[target_col] = 0
        return df 