"""
Z-Score indicator module.
Calculates the Z-Score (standard score) for price data.
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging

from indicators.utils import get_column

logger = logging.getLogger(__name__)


def calculate_zscore(df, length=100, target_col='ZScore'):
    """
    Calculate Z-Score indicator for price data.

    The Z-Score measures how many standard deviations a data point is from the mean.
    Values typically range from -3 to +3, with:
    - Values > 2 or < -2 indicating potential outliers
    - Values around 0 indicating data near the mean

    Args:
        df: DataFrame with price data
        length: Lookback period for calculating rolling mean and std (default: 100)
        target_col: Name of the output column (default: 'ZScore')

    Returns:
        DataFrame: Original DataFrame with Z-Score column added
    """
    try:
        logger.info(f"Calculating {target_col} with length={length}")

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Get close column name
        close_col = get_column(result_df, 'close')
        if close_col is None:
            logger.error(f"No close column found. Available: {result_df.columns.tolist()}")
            result_df[target_col] = 0.0
            return result_df

        # Calculate Z-Score using pandas_ta
        zscore = ta.zscore(result_df[close_col], length=length)

        # Add to DataFrame
        result_df[target_col] = zscore

        # Fill NaN values with default (zero - indicating no deviation from mean)
        result_df[target_col] = result_df[target_col].fillna(0.0)

        # Clip extreme values to prevent outliers from affecting training
        result_df[target_col] = np.clip(result_df[target_col], -4.0, 4.0)

        logger.info(f"Z-Score calculated. Range: [{result_df[target_col].min():.4f}, {result_df[target_col].max():.4f}]")

        return result_df
    except Exception as e:
        logger.error(f"Error calculating Z-Score: {e}")
        if target_col not in df.columns:
            df[target_col] = 0.0
        return df 