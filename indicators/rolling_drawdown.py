"""
Rolling Drawdown indicator module.

Calculates the percentage drawdown from the rolling maximum of close prices.
Provides a regime-aware signal: values near 0 indicate the market is near highs,
deeply negative values indicate a correction or crash.
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def calculate_rolling_drawdown(df, window=50, target_col='ROLLING_DD'):
    """
    Calculate rolling drawdown percentage from the rolling high.

    Args:
        df: DataFrame with price data
        window: Lookback window for rolling max (default: 50)
        target_col: Name of the output column (default: 'ROLLING_DD')

    Returns:
        DataFrame: Original DataFrame with rolling drawdown column added
    """
    try:
        logger.info(f"Calculating {target_col} with window={window}")

        # Find close column (case-insensitive)
        close_col = None
        for col in df.columns:
            if col.lower() == 'close':
                close_col = col
                break

        if close_col is None:
            logger.error(f"No close column found. Available: {df.columns.tolist()}")
            df[target_col] = 0.0
            return df

        rolling_max = df[close_col].rolling(window=window, min_periods=1).max()
        df[target_col] = (df[close_col] - rolling_max) / rolling_max * 100

        # Fill NaN values with 0 (no drawdown)
        df[target_col] = df[target_col].fillna(0.0)

        return df
    except Exception as e:
        logger.error(f"Error calculating rolling drawdown: {e}")
        if target_col not in df.columns:
            df[target_col] = 0.0
        return df
