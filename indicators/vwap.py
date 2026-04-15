"""Volume Weighted Average Price (VWAP) indicator module."""

import logging

import numpy as np
import pandas as pd

from indicators.utils import get_ohlcv_columns, get_column
from utils.session_context import build_session_context, session_config_from_mapping

logger = logging.getLogger(__name__)


def calculate_vwap(
    df,
    target_col='VWAP',
    *,
    add_derived: bool = True,
    dist_pct_col: str = 'VWAP_DIST_PCT',
    dist_z_col: str = 'VWAP_DIST_Z',
    slope_col: str = 'VWAP_SLOPE',
    above_col: str = 'VWAP_ABOVE',
    zscore_window: int = 20,
    session_config: dict | None = None,
):
    """
    Calculate Volume Weighted Average Price (VWAP)

    VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    where Typical Price = (High + Low + Close) / 3

    The calculation resets at the start of each trading day.

    Args:
        df: DataFrame with OHLCV data (must have high, low, close, volume columns)
        target_col: Name of the VWAP output column (default: 'VWAP')

    Returns:
        DataFrame: Original DataFrame with VWAP column added
    """
    try:
        logger.info(f"Calculating {target_col}")

        # Make a copy to avoid modifying the original
        result_df = df.copy()

        # Get column names using utility function
        cols = get_ohlcv_columns(result_df)
        high_col = cols.get('high')
        low_col = cols.get('low')
        close_col = cols.get('close')
        volume_col = cols.get('volume')

        # Validate required columns exist
        if not all([high_col, low_col, close_col]):
            logger.error(f"Missing required OHLC columns. Available: {result_df.columns.tolist()}")
            result_df[target_col] = result_df[close_col] if close_col else 0
            return result_df

        # Handle missing volume
        if volume_col is None:
            logger.warning("Volume column not found. Adding placeholder values.")
            result_df['volume'] = 1000
            volume_col = 'volume'

        session_ctx = build_session_context(result_df.index, session_config_from_mapping(session_config))
        session_ids = session_ctx["session_anchor"]
        in_session = session_ctx["in_session"]

        # Calculate Typical Price
        typical_price = (result_df[high_col] + result_df[low_col] + result_df[close_col]) / 3

        # Calculate TPV (Typical Price * Volume)
        masked_volume = result_df[volume_col].where(in_session, 0.0)
        tpv = typical_price * masked_volume

        # Create a temporary DataFrame for groupby operations
        calc_df = pd.DataFrame({
            'session_id': session_ids,
            'tpv': tpv,
            'volume': masked_volume
        }, index=df.index)

        # Cumulative sums within each configured session
        calc_df['cum_tpv'] = calc_df.groupby('session_id')['tpv'].cumsum()
        calc_df['cum_vol'] = calc_df.groupby('session_id')['volume'].cumsum()

        # VWAP = cumulative TPV / cumulative Volume
        result_df[target_col] = calc_df['cum_tpv'] / calc_df['cum_vol']

        # Handle edge cases
        result_df[target_col] = result_df[target_col].replace([np.inf, -np.inf], np.nan)
        result_df[target_col] = result_df[target_col].fillna(result_df[close_col])

        if add_derived:
            vwap_dist_pct = (result_df[close_col] - result_df[target_col]) / result_df[target_col].replace(0, np.nan)
            vwap_dist_pct = vwap_dist_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            result_df[dist_pct_col] = vwap_dist_pct

            grouped_dist = result_df.groupby(session_ids)[dist_pct_col]
            rolling_mean = grouped_dist.transform(
                lambda s: s.rolling(window=zscore_window, min_periods=5).mean()
            )
            rolling_std = grouped_dist.transform(
                lambda s: s.rolling(window=zscore_window, min_periods=5).std()
            ).replace(0, np.nan)
            result_df[dist_z_col] = ((result_df[dist_pct_col] - rolling_mean) / rolling_std).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)

            calc_df[target_col] = result_df[target_col]
            result_df[slope_col] = calc_df.groupby('session_id')[target_col].diff().fillna(0.0)
            result_df[above_col] = (result_df[close_col] > result_df[target_col]).astype(float)

        return result_df

    except Exception as e:
        logger.error(f"Error calculating VWAP: {e}")
        import traceback
        traceback.print_exc()

        # Add the column with default values in case of error
        close_col = get_column(df, 'close')
        if target_col not in df.columns:
            df = df.copy()
            df[target_col] = df[close_col] if close_col else 0
        return df
