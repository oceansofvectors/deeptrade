"""
Exponential Moving Average (EMA) indicator module.
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_ema(df, length=20, target_col='EMA'):
    """
    Calculate Exponential Moving Average (EMA) using pandas_ta library
    
    Args:
        df: DataFrame with price data
        length: Period for EMA calculation (default: 20)
        target_col: Name of the output column (default: 'EMA')
        
    Returns:
        DataFrame: Original DataFrame with EMA column added
    """
    try:
        logger.info(f"Calculating {target_col} with length={length}")
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Ensure close is float for consistent comparison
        result_df['close'] = result_df['close'].astype(float)
        
        # Calculate EMA
        result_df[target_col] = ta.ema(result_df['close'], length=length)
            
        # Fill NaN values in the main column
        result_df[target_col] = result_df[target_col].fillna(result_df['close'])
        
        return result_df
    except Exception as e:
        logger.error(f"Error calculating EMA: {e}")
        # Add the column with default values in case of error
        if target_col not in df.columns:
            df[target_col] = df['close']
        return df 