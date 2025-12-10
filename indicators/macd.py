"""
Moving Average Convergence Divergence (MACD) indicator module.
"""
import pandas as pd
import pandas_ta as ta
import logging

from indicators.utils import get_column

logger = logging.getLogger(__name__)


def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9,
                   macd_col='MACD', signal_col='Signal', histogram_col='Histogram'):
    """
    Calculate Moving Average Convergence Divergence (MACD).

    The MACD is calculated by subtracting the slow period EMA from the fast period EMA.
    The signal line is an EMA of the MACD line.
    The histogram represents the difference between the MACD and signal line.

    Args:
        df: DataFrame with price data
        fast_period: Period for fast EMA (default: 12)
        slow_period: Period for slow EMA (default: 26)
        signal_period: Period for signal line (default: 9)
        macd_col: Name of the MACD line column (default: 'MACD')
        signal_col: Name of the signal line column (default: 'Signal')
        histogram_col: Name of the histogram column (default: 'Histogram')

    Returns:
        DataFrame: Original DataFrame with MACD columns added
    """
    try:
        logger.info(f"Calculating MACD with fast={fast_period}, slow={slow_period}, signal={signal_period}")

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Get close column name
        close_col = get_column(result_df, 'close')
        if close_col is None:
            logger.error(f"No close column found. Available: {result_df.columns.tolist()}")
            result_df[macd_col] = 0
            result_df[signal_col] = 0
            result_df[histogram_col] = 0
            return result_df

        # Calculate MACD using pandas_ta
        macd_result = ta.macd(result_df[close_col], fast=fast_period, slow=slow_period, signal=signal_period)

        # Rename columns to match expected output
        result_df[macd_col] = macd_result[f'MACD_{fast_period}_{slow_period}_{signal_period}']
        result_df[signal_col] = macd_result[f'MACDs_{fast_period}_{slow_period}_{signal_period}']
        result_df[histogram_col] = macd_result[f'MACDh_{fast_period}_{slow_period}_{signal_period}']

        # Fill NaN values with zeros
        result_df[macd_col] = result_df[macd_col].fillna(0)
        result_df[signal_col] = result_df[signal_col].fillna(0)
        result_df[histogram_col] = result_df[histogram_col].fillna(0)

        return result_df

    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        if macd_col not in df.columns:
            df[macd_col] = 0
        if signal_col not in df.columns:
            df[signal_col] = 0
        if histogram_col not in df.columns:
            df[histogram_col] = 0
        return df 