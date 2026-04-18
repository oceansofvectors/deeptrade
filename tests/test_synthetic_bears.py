#!/usr/bin/env python
"""Tests for synthetic bear episode augmentation."""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.synthetic_bears import REGIME_TYPES, augment_with_synthetic_bears, extract_ohlcv_frame  # noqa: E402


class TestSyntheticBearAugmentation(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2021-01-01 14:30:00", periods=120, freq="5min", tz="UTC")
        close = np.linspace(100.0, 112.0, len(idx))
        self.df = pd.DataFrame({
            "open": close - 0.15,
            "high": close + 0.35,
            "low": close - 0.45,
            "close": close,
            "volume": np.linspace(1000, 1500, len(idx)),
            "RSI": np.linspace(45, 55, len(idx)),
        }, index=idx)

    def test_extract_ohlcv_frame_keeps_only_raw_columns(self):
        raw = extract_ohlcv_frame(self.df)
        self.assertEqual(list(raw.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(len(raw), len(self.df))

    def test_augmentation_preserves_original_prefix_and_returns_metadata(self):
        raw = extract_ohlcv_frame(self.df)
        augmented, metadata = augment_with_synthetic_bears(
            raw,
            oversample_ratio=0.30,
            segment_length_pct=0.10,
            seed=7,
            return_metadata=True,
        )

        self.assertIsNotNone(metadata)
        self.assertGreater(len(augmented), len(raw))
        pd.testing.assert_frame_equal(augmented.iloc[:len(raw)].reset_index(drop=True), raw.reset_index(drop=True))
        self.assertTrue(augmented.index.is_monotonic_increasing)
        self.assertEqual(metadata["synthetic_bars"], len(augmented) - len(raw))
        self.assertEqual(sum(metadata["regime_counts"].values()), metadata["num_segments"])
        self.assertTrue(all(segment["regime_type"] in REGIME_TYPES for segment in metadata["segments"]))

    def test_synthetic_episodes_are_appended_not_row_shuffled(self):
        raw = extract_ohlcv_frame(self.df)
        augmented, metadata = augment_with_synthetic_bears(
            raw,
            oversample_ratio=0.25,
            segment_length_pct=0.10,
            seed=11,
            return_metadata=True,
        )
        first_synthetic_start = pd.Timestamp(metadata["segments"][0]["synthetic_start"])
        self.assertGreater(first_synthetic_start, raw.index[-1])
        synthetic_only = augmented.loc[augmented.index > raw.index[-1]]
        self.assertEqual(len(synthetic_only), metadata["synthetic_bars"])

    def test_tiny_oversample_ratio_does_not_force_full_segment(self):
        raw = extract_ohlcv_frame(self.df)
        augmented, metadata = augment_with_synthetic_bears(
            raw,
            oversample_ratio=0.10,
            segment_length_pct=0.20,
            seed=5,
            return_metadata=True,
        )

        self.assertEqual(metadata["target_synthetic_bars"], int(round(len(raw) * 0.10)))
        self.assertEqual(metadata["synthetic_bars"], metadata["target_synthetic_bars"])
        self.assertLess(metadata["synthetic_bars"], int(len(raw) * 0.20))
        self.assertEqual(len(augmented) - len(raw), metadata["synthetic_bars"])

    def test_sub_minimum_target_ratio_skips_augmentation(self):
        raw = extract_ohlcv_frame(self.df)
        augmented, metadata = augment_with_synthetic_bears(
            raw,
            oversample_ratio=0.01,
            segment_length_pct=0.20,
            seed=3,
            return_metadata=True,
        )

        pd.testing.assert_frame_equal(augmented, raw)
        self.assertFalse(metadata["enabled"])
        self.assertEqual(metadata["synthetic_bars"], 0)
        self.assertEqual(metadata["num_segments"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
