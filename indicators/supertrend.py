"""
SuperTrend indicator module.

SuperTrend is a trend-following indicator that combines ATR with price bands.
It provides clear buy/sell signals based on price crossing the SuperTrend line.
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging

from indicators.utils import get_ohlcv_columns

logger = logging.getLogger(__name__)


def calculate_supertrend(
    df: pd.DataFrame,
    length: int = 10,
    multiplier: float = 3.0,
    target_col: str = 'supertrend'
) -> pd.DataFrame:
    """
    Calculate the standard SuperTrend indicator.

    SuperTrend uses ATR to create dynamic support/resistance bands:
    - Upper Band = (High + Low) / 2 + multiplier * ATR
    - Lower Band = (High + Low) / 2 - multiplier * ATR

    The trend direction changes when price crosses the SuperTrend line:
    - Uptrend (+1): Price is above SuperTrend (use lower band as support)
    - Downtrend (-1): Price is below SuperTrend (use upper band as resistance)

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        length: ATR period (default: 10)
        multiplier: ATR multiplier for band calculation (default: 3.0)
        target_col: Name of output column (+1/-1 signal, default: 'supertrend')

    Returns:
        DataFrame augmented with `target_col` containing +1 (uptrend) or -1 (downtrend)
    """
    try:
        # Get column names
        cols = get_ohlcv_columns(df)
        if not all([cols['high'], cols['low'], cols['close']]):
            logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
            df[target_col] = 0
            return df

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Calculate SuperTrend using pandas_ta
        st = ta.supertrend(
            high=result_df[cols['high']],
            low=result_df[cols['low']],
            close=result_df[cols['close']],
            length=length,
            multiplier=multiplier
        )

        # pandas_ta returns columns like:
        # - SUPERT_{length}_{multiplier} (the SuperTrend line value)
        # - SUPERTd_{length}_{multiplier} (the direction: 1 or -1)
        # - SUPERTl_{length}_{multiplier} (lower band, used in uptrend)
        # - SUPERTs_{length}_{multiplier} (upper band, used in downtrend)

        # Find the direction column
        trend_col = None
        for col in st.columns:
            if col.startswith('SUPERTd_'):
                trend_col = col
                break

        if trend_col is None:
            logger.error(f"Could not find SuperTrend direction column. Columns: {st.columns.tolist()}")
            result_df[target_col] = 0
            return result_df

        # Extract the trend direction (+1 for uptrend, -1 for downtrend)
        result_df[target_col] = st[trend_col].fillna(0).astype(int)

        # Log distribution
        value_counts = result_df[target_col].value_counts().to_dict()
        logger.info(f"SuperTrend distribution: {value_counts}")

        return result_df

    except Exception as e:
        logger.error(f"Error in calculate_supertrend: {e}")
        import traceback
        traceback.print_exc()

        if target_col not in df.columns:
            df = df.copy()
            df[target_col] = 0
        return df
