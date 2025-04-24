"""
Volume Weighted Average Price (VWAP) indicator module.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_vwap(df, target_col='VWAP'):
    """
    Calculate Volume Weighted Average Price (VWAP)
    
    Args:
        df: DataFrame with price and volume data
        target_col: Name of the VWAP output column (default: 'VWAP')
        
    Returns:
        DataFrame: Original DataFrame with VWAP column added
    """
    try:
        # Check if volume column exists, if not create with default values
        if 'volume' not in df.columns:
            logger.warning(f"Volume column not found. Adding placeholder values.")
            df['volume'] = 1000  # Default volume
        
        logger.info(f"Calculating {target_col}")
        
        # Create a copy of the dataframe to avoid modifying the original
        df_calc = df.copy()
        
        # Handle date extraction in a timezone-aware safe way
        if isinstance(df.index, pd.DatetimeIndex):
            # For timezone-aware datetime index, convert to string date safely
            df_calc['date'] = df.index.strftime('%Y-%m-%d')
        else:
            # For non-datetime index or other index types, convert to datetime first
            try:
                df_calc['date'] = pd.to_datetime(df.index).strftime('%Y-%m-%d')
            except:
                # Fallback to a simple incremental counter for each day
                df_calc['date'] = pd.to_datetime(df.index).normalize()
                df_calc['date'] = df_calc['date'].dt.strftime('%Y-%m-%d')
        
        # Detect date changes
        date_change = df_calc['date'] != df_calc['date'].shift(1)
        
        # Initialize VWAP calculation
        df_calc['TP'] = (df_calc['high'] + df_calc['low'] + df_calc['close']) / 3  # Typical Price
        df_calc['TPV'] = df_calc['TP'] * df_calc['volume']  # Typical Price * Volume
        
        # Cumulative sum of TPV and Volume, reset on date change
        df_calc['cum_tpv'] = df_calc.groupby('date')['TPV'].cumsum()
        df_calc['cum_vol'] = df_calc.groupby('date')['volume'].cumsum()
        
        # VWAP calculation
        df_calc[target_col] = df_calc['cum_tpv'] / df_calc['cum_vol']
        
        # Copy the VWAP values back to the original dataframe
        df[target_col] = df_calc[target_col]
        
        # Fill NaN values with defaults
        df[target_col] = df[target_col].fillna(df['close'])
        
        return df
    except Exception as e:
        logger.error(f"Error calculating VWAP: {e}")
        # Add the column with default values in case of error
        if target_col not in df.columns:
            df[target_col] = df['close']
        return df 