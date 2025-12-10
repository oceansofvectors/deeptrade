"""
Parabolic SAR (Stop and Reverse) indicator module.

Parabolic SAR is a trend-following indicator that provides potential entry and exit points.
It appears as dots above or below price:
- Dots below price = Uptrend (bullish)
- Dots above price = Downtrend (bearish)
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging

from indicators.utils import get_ohlcv_columns, get_column

logger = logging.getLogger(__name__)


def calculate_psar(df, af=0.02, max_af=0.2, psar_col='PSAR', dir_col='PSAR_DIR'):
    """
    Calculate Parabolic SAR using pandas_ta library.

    The Parabolic SAR formula:
    - SAR(tomorrow) = SAR(today) + AF * (EP - SAR(today))
    - AF starts at 0.02 and increases by 0.02 each time EP makes a new high/low
    - AF is capped at max_af (default 0.2)
    - EP (Extreme Point) is the highest high in uptrend or lowest low in downtrend

    Args:
        df: DataFrame with OHLC price data
        af: Initial acceleration factor (default: 0.02)
        max_af: Maximum acceleration factor (default: 0.2)
        psar_col: Name of the PSAR value output column (default: 'PSAR')
        dir_col: Name of the PSAR direction column (default: 'PSAR_DIR')

    Returns:
        DataFrame: Original DataFrame with PSAR columns added
            - PSAR: The SAR value (price level for stop)
            - PSAR_DIR: Direction (+1 for uptrend/long, -1 for downtrend/short)
    """
    try:
        logger.info(f"Calculating {psar_col} with af={af}, max_af={max_af}")

        # Make a copy of the dataframe to avoid modifying the original
        result_df = df.copy()

        # Get column names
        cols = get_ohlcv_columns(result_df)
        if not all([cols['high'], cols['low'], cols['close']]):
            logger.error(f"Missing required columns. Available: {result_df.columns.tolist()}")
            result_df[psar_col] = 0
            result_df[dir_col] = 0
            return result_df

        close_col = cols['close']

        # Calculate PSAR using pandas_ta
        psar = ta.psar(
            high=result_df[cols['high']],
            low=result_df[cols['low']],
            close=result_df[close_col],
            af0=af,
            af=af,
            max_af=max_af
        )

        logger.info(f"PSAR result columns: {psar.columns.tolist()}")

        # pandas_ta returns columns like:
        # - PSARl_{af}_{max_af} (long PSAR - dots below price in uptrend)
        # - PSARs_{af}_{max_af} (short PSAR - dots above price in downtrend)
        # - PSARaf_{af}_{max_af} (acceleration factor)
        # - PSARr_{af}_{max_af} (reversal signal)

        # Find the long and short PSAR columns
        psar_long_col = None
        psar_short_col = None

        for col in psar.columns:
            if col.startswith('PSARl_'):
                psar_long_col = col
            elif col.startswith('PSARs_'):
                psar_short_col = col

        # Determine direction and PSAR value
        # When PSARl has a value (not NaN), we're in an uptrend
        # When PSARs has a value (not NaN), we're in a downtrend
        if psar_long_col and psar_short_col:
            psar_long = psar[psar_long_col]
            psar_short = psar[psar_short_col]

            # Direction: +1 when long PSAR is active (uptrend), -1 when short PSAR is active
            result_df[dir_col] = np.where(
                psar_long.notna(), 1,
                np.where(psar_short.notna(), -1, 0)
            )

            # PSAR value: use long PSAR in uptrend, short PSAR in downtrend
            result_df[psar_col] = np.where(
                psar_long.notna(), psar_long,
                np.where(psar_short.notna(), psar_short, np.nan)
            )
        elif psar_long_col:
            # Fallback: only long column available
            result_df[psar_col] = psar[psar_long_col]
            result_df[dir_col] = np.where(psar[psar_long_col].notna(), 1, -1)
        else:
            # Last resort: use first available column
            result_df[psar_col] = psar.iloc[:, 0] if len(psar.columns) > 0 else result_df[close_col]
            result_df[dir_col] = np.where(result_df[close_col] > result_df[psar_col], 1, -1)

        # Fill NaN values with sensible defaults
        result_df[psar_col] = result_df[psar_col].ffill().bfill()
        result_df[psar_col] = result_df[psar_col].fillna(result_df[close_col])
        result_df[dir_col] = result_df[dir_col].fillna(0).astype(int)

        # Log distribution
        dir_counts = result_df[dir_col].value_counts().to_dict()
        logger.info(f"PSAR direction distribution: {dir_counts}")

        return result_df

    except Exception as e:
        logger.error(f"Error calculating PSAR: {e}")
        import traceback
        traceback.print_exc()

        close_col = get_column(df, 'close')
        if psar_col not in df.columns:
            df = df.copy()
            df[psar_col] = df[close_col] if close_col else 0
        if dir_col not in df.columns:
            df[dir_col] = 0
        return df
