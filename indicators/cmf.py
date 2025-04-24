"""
Chaikin Money Flow (CMF) indicator module.
"""
import pandas as pd
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)

def calculate_cmf(df, length=20, target_col='CMF'):
    """
    Calculate Chaikin Money Flow (CMF) using pandas_ta library
    
    Args:
        df: DataFrame with price and volume data
        length: Period for CMF calculation (default: 20)
        target_col: Name of the output column (default: 'CMF')
        
    Returns:
        DataFrame: Original DataFrame with CMF column added
    """
    try:
        # Check if volume column exists, if not create with default values
        if 'volume' not in df.columns:
            logger.warning(f"Volume column not found. Adding placeholder values.")
            df['volume'] = 1000  # Default volume
        
        logger.info(f"Calculating {target_col} with length={length}")
        
        # Calculate CMF
        df[target_col] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=length)
        
        # Fill NaN values with defaults
        df[target_col] = df[target_col].fillna(0)
        
        return df
    except Exception as e:
        logger.error(f"Error calculating CMF: {e}")
        # Add the column with default values in case of error
        if target_col not in df.columns:
            df[target_col] = 0
        return df 