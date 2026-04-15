"""
Volatility Percentile indicator module.

Calculates where current realized volatility ranks relative to its own
trailing distribution. High values (near 100) signal elevated volatility
regimes; low values signal calm markets.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def calculate_vol_percentile(df, window=60, target_col='VOL_PERCENTILE'):
    """
    Calculate the percentile rank of current rolling volatility vs its trailing window.

    Args:
        df: DataFrame with price data
        window: Lookback window for both volatility and ranking (default: 60)
        target_col: Name of the output column (default: 'VOL_PERCENTILE')

    Returns:
        DataFrame: Original DataFrame with volatility percentile column added
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
            df[target_col] = 50.0
            return df

        close_series = pd.to_numeric(df[close_col], errors='coerce').astype(float)
        prior_close = close_series.shift(1)
        safe_ratio = close_series.divide(prior_close.replace(0.0, np.nan))
        safe_ratio = safe_ratio.where(safe_ratio > 0.0)
        safe_ratio = safe_ratio.replace([np.inf, -np.inf], np.nan)
        returns = np.log(safe_ratio).replace([np.inf, -np.inf], np.nan)
        rolling_std = returns.rolling(window=window, min_periods=1).std()
        # Rank current vol vs trailing distribution, output 0-100
        df[target_col] = rolling_std.rolling(window=window, min_periods=1).rank(pct=True) * 100

        # Fill NaN values with neutral 50th percentile
        df[target_col] = df[target_col].fillna(50.0)

        return df
    except Exception as e:
        logger.error(f"Error calculating volatility percentile: {e}")
        if target_col not in df.columns:
            df[target_col] = 50.0
        return df
