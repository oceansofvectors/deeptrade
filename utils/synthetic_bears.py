"""
Synthetic bear data augmentation for training.

Generates artificial downtrend segments by inverting price movements from
real data. This teaches the model that markets don't always trend up and
helps it learn defensive behavior during corrections.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def augment_with_synthetic_bears(train_df: pd.DataFrame,
                                 oversample_ratio: float = 0.3,
                                 segment_length_pct: float = 0.15) -> pd.DataFrame:
    """
    Create synthetic bear episodes by inverting segments of the training data
    and appending them to the original training set.

    The inversion mirrors OHLC prices around the segment's starting price,
    turning uptrends into downtrends (and vice versa). Volume is amplified
    to mimic higher activity during sell-offs. Technical indicators are
    NOT recomputed here — they will be recalculated on the augmented data
    upstream, or scaled per-window after augmentation.

    Args:
        train_df: Training DataFrame with OHLC + indicator columns
        oversample_ratio: How many synthetic bars to add, as a fraction of
                          the original training data length (default: 0.3 = 30%)
        segment_length_pct: Length of each synthetic segment as a fraction of
                            the total training data (default: 0.15 = 15%)

    Returns:
        DataFrame: Original training data with synthetic bear segments appended
    """
    n = len(train_df)
    if n < 20:
        logger.warning("Training data too short for synthetic augmentation, skipping")
        return train_df

    segment_length = max(10, int(n * segment_length_pct))
    num_synthetic_bars = int(n * oversample_ratio)
    num_segments = max(1, num_synthetic_bars // segment_length)

    logger.info(f"Generating {num_segments} synthetic bear segments "
                f"({segment_length} bars each, ~{num_synthetic_bars} total bars)")

    # Identify price columns (case-insensitive)
    price_cols = {}
    for target in ['open', 'high', 'low', 'close']:
        for col in train_df.columns:
            if col.lower() == target:
                price_cols[target] = col
                break

    if 'close' not in price_cols:
        logger.error("No close column found, cannot generate synthetic bears")
        return train_df

    augmented_segments = []
    rng = np.random.default_rng(seed=42)

    for i in range(num_segments):
        # Pick a random segment from the training data
        max_start = n - segment_length
        if max_start <= 0:
            break
        start = rng.integers(0, max_start)
        segment = train_df.iloc[start:start + segment_length].copy()

        # Mirror prices around the segment's starting close price
        # This turns an uptrend into a downtrend of equal magnitude
        anchor = segment[price_cols['close']].iloc[0]

        for target, col in price_cols.items():
            segment[col] = 2 * anchor - segment[col]

        # After mirroring, high and low are swapped — fix them
        if 'high' in price_cols and 'low' in price_cols:
            h = segment[price_cols['high']].copy()
            l = segment[price_cols['low']].copy()
            segment[price_cols['high']] = h.combine(l, max)
            segment[price_cols['low']] = h.combine(l, min)

        # Amplify volume slightly to mimic panic selling (mild: 1.2-1.5x)
        for col in segment.columns:
            if col.lower() == 'volume':
                segment[col] = segment[col] * rng.uniform(1.2, 1.5)

        augmented_segments.append(segment)

    if not augmented_segments:
        return train_df

    # Concatenate original + synthetic, then shuffle to avoid the model
    # learning temporal position of synthetic data
    result = pd.concat([train_df] + augmented_segments, ignore_index=True)
    result = result.sample(frac=1.0, random_state=42).reset_index(drop=True)

    logger.info(f"Augmented training data: {n} -> {len(result)} rows "
                f"(+{len(result) - n} synthetic)")

    return result
