"""
Day of Week indicator module.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_day_of_week(df, dow_col='DOW', sin_col='DOW_SIN', cos_col='DOW_COS'):
    """
    Calculate Day of Week indicator with circular encoding
    
    Args:
        df: DataFrame with datetime index
        dow_col: Name of the day of week column (default: 'DOW')
        sin_col: Name of the sine encoding column (default: 'DOW_SIN')
        cos_col: Name of the cosine encoding column (default: 'DOW_COS')
        
    Returns:
        DataFrame: Original DataFrame with day of week columns added
    """
    try:
        # Check if index is a datetime type
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            logger.warning("Index is not datetime. Attempting to convert.")
            try:
                temp_index = pd.to_datetime(df.index)
                # Get the day of week (0 = Monday, 6 = Sunday)
                dow = temp_index.dayofweek
            except:
                logger.error("Could not convert index to datetime. Using default values.")
                dow = np.zeros(len(df))
        else:
            # Get the day of week (0 = Monday, 6 = Sunday)
            dow = df.index.dayofweek
        
        # Add raw day of week column
        df[dow_col] = dow
        
        # Convert to sine and cosine representation (circular encoding)
        df[sin_col] = np.sin(2 * np.pi * df[dow_col] / 7)
        df[cos_col] = np.cos(2 * np.pi * df[dow_col] / 7)
        
        return df
    except Exception as e:
        logger.error(f"Error calculating Day of Week: {e}")
        # Add the columns with default values in case of error
        if dow_col not in df.columns:
            df[dow_col] = 0
        if sin_col not in df.columns:
            df[sin_col] = 0
        if cos_col not in df.columns:
            df[cos_col] = 1  # cos(0) = 1
        return df 