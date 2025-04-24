"""
Average Directional Index (ADX) indicator module.
"""
import pandas as pd
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)

def calculate_adx(df, length=14, adx_col='ADX'):
    """
    Calculate Average Directional Index (ADX) using pandas_ta library
    
    Args:
        df: DataFrame with price data
        length: Period for ADX calculation (default: 14)
        adx_col: Name of the ADX output column (default: 'ADX')
        
    Returns:
        DataFrame: Original DataFrame with ADX column added
    """
    logger.info(f"Calculating {adx_col} with length={length}")
    
    # Calculate ADX
    adx_result = ta.adx(df['high'], df['low'], df['close'], length=length)
    
    # Extract ADX value - handle different column naming conventions
    if f'ADX_{length}' in adx_result.columns:
        df[adx_col] = adx_result[f'ADX_{length}']
    else:
        # Assume the first column is ADX
        df[adx_col] = adx_result.iloc[:, 0]
    
    # Fill NaN values with defaults
    df[adx_col] = df[adx_col].fillna(0.0)

    return df