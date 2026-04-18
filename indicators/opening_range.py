"""Opening-range and overnight-gap features for the cash session."""

import logging

import numpy as np
import pandas as pd

from indicators.utils import get_ohlcv_columns
from utils.session_context import build_session_context, session_config_from_mapping

logger = logging.getLogger(__name__)


def calculate_opening_range_features(
    df: pd.DataFrame,
    *,
    opening_range_minutes: int = 30,
    or_high_col: str = "OPENING_RANGE_HIGH",
    or_low_col: str = "OPENING_RANGE_LOW",
    or_width_col: str = "OPENING_RANGE_WIDTH_PCT",
    dist_high_col: str = "DIST_TO_OR_HIGH_PCT",
    dist_low_col: str = "DIST_TO_OR_LOW_PCT",
    breakout_dir_col: str = "OR_BREAKOUT_DIR",
    breakout_active_col: str = "OR_BREAKOUT_ACTIVE",
    gap_col: str = "OVERNIGHT_GAP_PCT",
    open_prior_range_col: str = "OPEN_TO_PRIOR_RANGE_PCT",
    post_open_vol_col: str = "POST_OPEN_VOL_PCT",
    session_config: dict | None = None,
) -> pd.DataFrame:
    """Add opening-range and gap features for the configured session."""
    result_df = df.copy()
    cols = get_ohlcv_columns(result_df)
    open_col = cols.get("open")
    high_col = cols.get("high")
    low_col = cols.get("low")
    close_col = cols.get("close")

    if not all([open_col, high_col, low_col, close_col]):
        logger.error("Missing OHLC columns for opening-range features")
        for col in [
            or_high_col, or_low_col, or_width_col, dist_high_col, dist_low_col,
            breakout_dir_col, breakout_active_col, gap_col, open_prior_range_col, post_open_vol_col,
        ]:
            result_df[col] = 0.0
        return result_df

    session_ctx = build_session_context(result_df.index, session_config_from_mapping(session_config))
    session_ids = session_ctx["session_anchor"]
    minutes_since_open = session_ctx["minutes_since_open"]
    in_session = session_ctx["in_session"]
    in_opening = in_session & (minutes_since_open < opening_range_minutes)
    after_opening = in_session & (minutes_since_open >= opening_range_minutes)

    result_df[or_high_col] = 0.0
    result_df[or_low_col] = 0.0
    result_df[or_width_col] = 0.0
    result_df[dist_high_col] = 0.0
    result_df[dist_low_col] = 0.0
    result_df[breakout_dir_col] = 0.0
    result_df[breakout_active_col] = 0.0
    result_df[gap_col] = 0.0
    result_df[open_prior_range_col] = 0.0
    result_df[post_open_vol_col] = 0.0

    prev_close = None
    prev_high = None
    prev_low = None

    unique_sessions = pd.Index(session_ids.unique())
    for session in unique_sessions:
        session_mask = session_ids == session
        opening_mask = session_mask & in_opening
        if not opening_mask.any():
            continue

        session_open_idx = result_df.index[opening_mask][0]
        opening_open = float(result_df.loc[session_open_idx, open_col])
        active_session_mask = session_mask & in_session
        opening_high_series = result_df.loc[opening_mask, high_col].astype(float).cummax()
        opening_low_series = result_df.loc[opening_mask, low_col].astype(float).cummin()
        opening_mid_series = (opening_high_series + opening_low_series) / 2.0
        opening_width_series = (opening_high_series - opening_low_series).clip(lower=1e-9)
        opening_width_pct_series = opening_width_series / opening_mid_series.abs().clip(lower=1e-9)

        # During the opening window, only expose the range accumulated so far.
        opening_close_series = result_df.loc[opening_mask, close_col].astype(float)
        result_df.loc[opening_mask, or_high_col] = opening_high_series.to_numpy()
        result_df.loc[opening_mask, or_low_col] = opening_low_series.to_numpy()
        result_df.loc[opening_mask, or_width_col] = opening_width_pct_series.to_numpy()
        result_df.loc[opening_mask, dist_high_col] = (
            (opening_close_series - opening_high_series) / opening_high_series.abs().clip(lower=1e-9)
        ).to_numpy()
        result_df.loc[opening_mask, dist_low_col] = (
            (opening_close_series - opening_low_series) / opening_low_series.abs().clip(lower=1e-9)
        ).to_numpy()

        opening_high = float(opening_high_series.iloc[-1])
        opening_low = float(opening_low_series.iloc[-1])
        opening_mid = (opening_high + opening_low) / 2.0
        width = max(opening_high - opening_low, 1e-9)
        width_pct = width / max(abs(opening_mid), 1e-9)

        after_open_mask = session_mask & after_opening
        result_df.loc[after_open_mask, or_high_col] = opening_high
        result_df.loc[after_open_mask, or_low_col] = opening_low
        result_df.loc[after_open_mask, or_width_col] = width_pct

        close_series = result_df.loc[after_open_mask, close_col].astype(float)
        result_df.loc[after_open_mask, dist_high_col] = (close_series - opening_high) / max(abs(opening_high), 1e-9)
        result_df.loc[after_open_mask, dist_low_col] = (close_series - opening_low) / max(abs(opening_low), 1e-9)

        breakout_mask = after_open_mask
        breakout_dir = np.where(
            result_df.loc[breakout_mask, close_col].astype(float) > opening_high,
            1.0,
            np.where(result_df.loc[breakout_mask, close_col].astype(float) < opening_low, -1.0, 0.0),
        )
        result_df.loc[breakout_mask, breakout_dir_col] = breakout_dir
        result_df.loc[breakout_mask, breakout_active_col] = (breakout_dir != 0.0).astype(float)
        result_df.loc[breakout_mask, post_open_vol_col] = (
            (result_df.loc[breakout_mask, high_col].astype(float) - result_df.loc[breakout_mask, low_col].astype(float)) / width
        )

        if prev_close is not None:
            result_df.loc[active_session_mask, gap_col] = (opening_open - prev_close) / max(abs(prev_close), 1e-9)
        if prev_high is not None and prev_low is not None:
            prev_range = max(prev_high - prev_low, 1e-9)
            result_df.loc[active_session_mask, open_prior_range_col] = (opening_open - prev_close) / prev_range

        prev_close = float(result_df.loc[active_session_mask, close_col].iloc[-1])
        prev_high = float(result_df.loc[active_session_mask, high_col].max())
        prev_low = float(result_df.loc[active_session_mask, low_col].min())

    return result_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
