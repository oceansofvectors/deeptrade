import pandas as pd
import pandas_ta as ta
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_supertrend(
    df: pd.DataFrame,
    length: int = 21,
    multiplier: float = 4.5,
    smooth_periods: int = 5,
    threshold: float = 0.5,
    lookback_periods: int = 3,
    min_trend_duration: int = 3,
    target_col: str = 'supertrend'
) -> pd.DataFrame:
    """
    Calculate a stabilized Supertrend indicator with trend-duration guard.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        length: ATR period (default: 21)
        multiplier: ATR multiplier (default: 4.5)
        smooth_periods: Rolling window for smoothing raw trend (default: 5)
        threshold: Minimum rolling-average magnitude to register a signal (default: 0.5)
        lookback_periods: Consecutive smoothed signals required to flip (default: 3)
        min_trend_duration: Minimum bars before another flip allowed (default: 3)
        target_col: Name of output column (+1/-1 signal, default: 'supertrend')
    Returns:
        DataFrame augmented with `target_col` of +1/-1 trend signals.
    """
    try:
        # 1. Compute raw Supertrend
        st = ta.supertrend(
            high=df['high'], low=df['low'], close=df['close'],
            length=length, multiplier=multiplier
        )
        trend_col = next((c for c in st.columns if c.startswith('SUPERTd_')), None)
        df['raw_trend'] = st[trend_col].fillna(method='ffill').fillna(0) if trend_col else 0

        # 2. Smooth the raw trend (rolling average)
        if smooth_periods > 1:
            rolling = df['raw_trend'].rolling(window=smooth_periods, min_periods=1).mean()
            # map to -1, 0, +1
            df['smoothed'] = np.where(
                rolling >= threshold, 1,
                np.where(rolling <= -threshold, -1, 0)
            )
        else:
            df['smoothed'] = df['raw_trend']

        # 3. Initialize
        df[target_col] = 0
        last_flip_index = 0

        # Set first value
        if len(df) > 0:
            first_signal = df['smoothed'].iat[0]
            df[target_col].iat[0] = first_signal if first_signal != 0 else 0
            last_flip_index = 0

        # 4. Loop with lookback & min-duration
        for i in range(1, len(df)):
            prev = df[target_col].iat[i-1]
            curr = df['smoothed'].iat[i]

            # guard: enforce minimum trend duration
            if (i - last_flip_index) < min_trend_duration:
                df[target_col].iat[i] = prev
                continue

            # no new signal => hold
            if curr == 0:
                df[target_col].iat[i] = prev
                continue

            # same as prior => hold
            if curr == prev:
                df[target_col].iat[i] = prev
                continue

            # candidate flip: require lookback_periods of same smoothed
            start = max(0, i - lookback_periods + 1)
            window = df['smoothed'].iloc[start:i+1]
            if (window == curr).all():
                df[target_col].iat[i] = curr
                last_flip_index = i
            else:
                df[target_col].iat[i] = prev

        # 5. Cleanup
        df.drop(['raw_trend', 'smoothed'], axis=1, inplace=True)
        logger.info(f"Supertrend distribution: {df[target_col].value_counts().to_dict()}")
        return df

    except Exception as e:
        logger.error(f"Error in calculate_supertrend: {e}")
        if target_col not in df.columns:
            df[target_col] = 0
        return df
