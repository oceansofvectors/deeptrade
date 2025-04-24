"""
Relative Strength Index (RSI) indicator module.
"""
import pandas as pd
import pandas_ta as ta
import logging

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
        df[target_col] = ta.rsi(df['close'], length=length)
        
        # Don't automatically fill NaN values
        return df
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        # Add the column with default values in case of error
        if target_col not in df.columns:
            df[target_col] = 50
        return df 