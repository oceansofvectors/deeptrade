"""
Utility functions for indicators.
"""


def get_column(df, base_name):
    """
    Get the column name from a DataFrame, handling different capitalizations.

    Args:
        df: DataFrame to check
        base_name: Base column name to look for (e.g., 'close', 'high', 'low', 'volume')

    Returns:
        str: The actual column name found, or None if not found
    """
    # Try different capitalizations
    variants = [
        base_name.lower(),      # close
        base_name.capitalize(), # Close
        base_name.upper(),      # CLOSE
    ]

    for variant in variants:
        if variant in df.columns:
            return variant

    return None


def get_ohlcv_columns(df):
    """
    Get all OHLCV column names from a DataFrame.

    Args:
        df: DataFrame to check

    Returns:
        dict: Dictionary with keys 'open', 'high', 'low', 'close', 'volume'
              and values as the actual column names (or None if not found)
    """
    return {
        'open': get_column(df, 'open'),
        'high': get_column(df, 'high'),
        'low': get_column(df, 'low'),
        'close': get_column(df, 'close'),
        'volume': get_column(df, 'volume'),
    }
