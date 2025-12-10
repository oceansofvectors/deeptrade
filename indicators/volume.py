"""
Volume indicator module for volume data.
"""
import pandas as pd
import pandas_ta as ta
import logging

from indicators.utils import get_column

logger = logging.getLogger(__name__)


def calculate_volume_indicator(df, ma_length=20, target_col='VOLUME_RATIO'):
    """
    Calculate volume ratio compared to moving average

    Args:
        df: DataFrame with price and volume data
        ma_length: Period for moving average calculation (default: 20)
        target_col: Name of the volume ratio output column (default: 'VOLUME_RATIO')

    Returns:
        DataFrame: Original DataFrame with volume ratio column added
    """
    try:
        logger.info(f"Calculating {target_col} with ma_length={ma_length}")

        # Get volume column name
        volume_col = get_column(df, 'volume')
        if volume_col is None:
            logger.warning("Volume column not found. Adding placeholder values.")
            df['volume'] = 1000
            volume_col = 'volume'

        # Calculate volume moving average
        volume_ma = ta.sma(df[volume_col], length=ma_length)

        # Calculate volume ratio compared to moving average
        # volume / average => 1 means average volume, >1 is above avg, <1 is below
        df[target_col] = df[volume_col] / volume_ma

        # Fill NaN values with defaults
        df[target_col] = df[target_col].fillna(1)

        return df
    except Exception as e:
        logger.error(f"Error calculating Volume indicator: {e}")
        if target_col not in df.columns:
            df[target_col] = 1
        return df 