"""
Commodity Channel Index (CCI) indicator module.
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging

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
        
        # Calculate CCI
        cci = ta.cci(result_df['high'], result_df['low'], result_df['close'], length=length)
        
        # Add to DataFrame
        result_df[target_col] = cci
        
        # Fill NaN values with default (zero)
        result_df[target_col] = result_df[target_col].fillna(0.0)
        
        return result_df
    except Exception as e:
        logger.error(f"Error calculating CCI: {e}")
        # Add the column with default values in case of error
        if target_col not in df.columns:
            df[target_col] = 0.0
        return df 