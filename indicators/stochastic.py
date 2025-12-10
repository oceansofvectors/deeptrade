"""
Stochastic Oscillator indicator module.
"""
import pandas as pd
import pandas_ta as ta
import logging

from indicators.utils import get_ohlcv_columns

logger = logging.getLogger(__name__)


def calculate_stochastic(df, k_period=14, d_period=3, smooth_k=3,
                         k_col='K', d_col='D'):
    """
    Calculate Stochastic Oscillator using pandas_ta

    Args:
        df: DataFrame with price data (high, low, close)
        k_period: Period for %K calculation (default: 14)
        d_period: Period for %D calculation (default: 3)
        smooth_k: Smoothing for %K (default: 3)
        k_col: Name of the %K output column (default: 'K')
        d_col: Name of the %D output column (default: 'D')

    Returns:
        DataFrame: Original DataFrame with Stochastic columns added
    """
    try:
        logger.info(f"Calculating Stochastic with k_period={k_period}, d_period={d_period}, smooth_k={smooth_k}")

        # Create a copy of the dataframe
        result_df = df.copy()

        # Get column names
        cols = get_ohlcv_columns(result_df)
        if not all([cols['high'], cols['low'], cols['close']]):
            logger.error(f"Missing required columns. Available: {result_df.columns.tolist()}")
            result_df[k_col] = 50
            result_df[d_col] = 50
            return result_df

        # Calculate stochastic using pandas_ta
        stoch = ta.stoch(high=result_df[cols['high']], low=result_df[cols['low']], close=result_df[cols['close']],
                         k=k_period, d=d_period, smooth_k=smooth_k)

        # Get column names from the result
        k_name = stoch.columns[0]  # First column is %K
        d_name = stoch.columns[1]  # Second column is %D

        # Add to the result dataframe with user-specified column names
        result_df[k_col] = stoch[k_name]
        result_df[d_col] = stoch[d_name]

        # Fill NaN values with neutral value
        result_df[k_col] = result_df[k_col].fillna(50)
        result_df[d_col] = result_df[d_col].fillna(50)

        return result_df
    except Exception as e:
        logger.error(f"Error calculating Stochastic: {e}")
        result_df = df.copy()
        result_df[k_col] = 50
        result_df[d_col] = 50
        return result_df 