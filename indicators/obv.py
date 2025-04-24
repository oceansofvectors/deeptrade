"""
On-Balance Volume (OBV) indicator module.
"""
import pandas as pd
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)

def calculate_obv(df, target_col='OBV'):
    """
    Calculate On-Balance Volume (OBV) using pandas_ta library
    
    Args:
        df: DataFrame with price data
        target_col: Name of the OBV output column (default: 'OBV')
        
    Returns:
        DataFrame: Original DataFrame with OBV column added
    """
    try:
        # Check if volume column exists, if not create with default values
        if 'volume' not in df.columns:
            logger.warning(f"Volume column not found. Adding placeholder values.")
            df['volume'] = 1000  # Default volume
        
        logger.info(f"Calculating {target_col}")
        
        # Calculate OBV
        df[target_col] = ta.obv(df['close'], df['volume'])
        
        # Fill NaN values with defaults
        df[target_col] = df[target_col].fillna(0)
        
        return df
    except Exception as e:
        logger.error(f"Error calculating OBV: {e}")
        # Add the column with default values in case of error
        if target_col not in df.columns:
            df[target_col] = 0
        return df 