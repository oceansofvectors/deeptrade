"""
Price Disparity indicator module.
"""
import pandas as pd
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)

def calculate_disparity(df, length=20, column=None, sma_col='SMA', target_col='DISPARITY'):
    """
    Calculate Price Disparity indicator (percent deviation from moving average)
    
    Args:
        df: DataFrame with price data
        length: Period for SMA calculation (default: 20)
        column: Column to use for calculation (default: auto-detect 'Close' or 'close')
        sma_col: Name of the SMA column to use or create (default: 'SMA')
        target_col: Name of the output column (default: 'DISPARITY')
        
    Returns:
        DataFrame: Original DataFrame with Disparity column added
    """
    try:
        # Auto-detect column name if not specified
        if column is None:
            column = 'Close' if 'Close' in df.columns else 'close'
        
        # Check if Disparity data is already in the DataFrame
        if target_col in df.columns:
            logger.info(f"Using existing {target_col} data in DataFrame")
        else:
            logger.info(f"Calculating {target_col} with length={length}")
            
            # Check if SMA is already calculated
            if sma_col not in df.columns:
                # Calculate SMA
                df[sma_col] = ta.sma(df[column], length=length)
            
            # Calculate disparity (percent deviation from moving average)
            df[target_col] = ((df[column] / df[sma_col]) - 1)
        
        # Fill NaN values with defaults
        df[target_col] = df[target_col].fillna(0)
        
        return df
    except Exception as e:
        logger.error(f"Error calculating Disparity: {e}")
        # Add the column with default values in case of error
        if target_col not in df.columns:
            df[target_col] = 0
        return df 