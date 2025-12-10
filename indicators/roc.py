"""
Rate of Change (ROC) indicator module.
"""
import pandas as pd
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)


def _get_close_column(df):
    """
    Get the close price column name, handling different capitalizations.

    Args:
        df: DataFrame to check

    Returns:
        str: The name of the close column, or None if not found
    """
    for col_name in ['close', 'Close', 'CLOSE']:
        if col_name in df.columns:
            return col_name
    return None


def calculate_roc(df, length=10, target_col='ROC'):
    """
    Calculate the Rate of Change (ROC) for a given DataFrame.

    ROC = ((Current Price / Price n periods ago) - 1) * 100

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the price data
    length : int, optional
        The length (period) used for the ROC calculation, default is 10
    target_col : str, optional
        The name of the target column for the ROC values, default is 'ROC'

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the original data and the new ROC column
    """
    try:
        # Create a copy of the DataFrame to avoid modifying the original
        result_df = df.copy()

        # Find the close column
        close_col = _get_close_column(result_df)
        if close_col is None:
            logger.error(f"No close column found in DataFrame. Available columns: {result_df.columns.tolist()}")
            result_df[target_col] = 0
            return result_df

        # Calculate ROC using pandas_ta
        result_df[target_col] = ta.roc(result_df[close_col], length=length)

        # Fill NaN values with 0 (first 'length' periods will be NaN)
        result_df[target_col] = result_df[target_col].fillna(0)

        logger.info(f"Calculated {target_col} with length={length}")
        return result_df

    except Exception as e:
        logger.error(f"Error calculating ROC: {e}")
        if target_col not in df.columns:
            df[target_col] = 0
        return df 