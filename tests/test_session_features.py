#!/usr/bin/env python
"""Tests for VWAP-derived and opening-range session features."""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config  # noqa: E402
from get_data import process_technical_indicators  # noqa: E402
from indicators.opening_range import calculate_opening_range_features  # noqa: E402
from indicators.vwap import calculate_vwap  # noqa: E402


def _sample_session_frame() -> pd.DataFrame:
    day1 = pd.date_range("2024-01-02 17:00", periods=10, freq="5min", tz="America/Chicago").tz_convert("UTC")
    day2 = pd.date_range("2024-01-03 17:00", periods=10, freq="5min", tz="America/Chicago").tz_convert("UTC")
    idx = day1.append(day2)

    closes_day1 = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=float)
    closes_day2 = np.array([120, 121, 122, 123, 124, 125, 126, 127, 128, 129], dtype=float)
    close = np.concatenate([closes_day1, closes_day2])

    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(len(close), 1000.0),
        },
        index=idx,
    )
    return df


class TestSessionFeatures(unittest.TestCase):
    def test_vwap_resets_each_cash_session(self):
        df = calculate_vwap(_sample_session_frame(), session_config=config["data"]["session"])

        self.assertAlmostEqual(df.iloc[0]["VWAP"], 100.0)
        self.assertAlmostEqual(df.iloc[10]["VWAP"], 120.0)
        self.assertAlmostEqual(df.iloc[0]["VWAP_DIST_PCT"], 0.0)
        self.assertAlmostEqual(df.iloc[10]["VWAP_DIST_PCT"], 0.0)
        self.assertAlmostEqual(df.iloc[0]["VWAP_SLOPE"], 0.0)
        self.assertAlmostEqual(df.iloc[10]["VWAP_SLOPE"], 0.0)

    def test_opening_range_and_gap_features(self):
        df = calculate_opening_range_features(
            _sample_session_frame(),
            opening_range_minutes=30,
            session_config=config["data"]["session"],
        )

        day1_or_high = 106.0
        day1_or_low = 99.0
        # The opening window should only expose the range built so far, not the
        # full future 30-minute range.
        self.assertAlmostEqual(df.iloc[0]["OPENING_RANGE_HIGH"], 101.0)
        self.assertAlmostEqual(df.iloc[0]["OPENING_RANGE_LOW"], day1_or_low)
        self.assertAlmostEqual(df.iloc[0]["OR_BREAKOUT_ACTIVE"], 0.0)
        self.assertAlmostEqual(df.iloc[6]["OPENING_RANGE_HIGH"], day1_or_high)
        self.assertAlmostEqual(df.iloc[6]["OPENING_RANGE_LOW"], day1_or_low)
        self.assertAlmostEqual(df.iloc[7]["OR_BREAKOUT_DIR"], 1.0)
        self.assertAlmostEqual(df.iloc[7]["OR_BREAKOUT_ACTIVE"], 1.0)

        expected_gap = (120.0 - 109.0) / 109.0
        self.assertAlmostEqual(df.iloc[10]["OVERNIGHT_GAP_PCT"], expected_gap)
        self.assertGreater(df.iloc[10]["OPEN_TO_PRIOR_RANGE_PCT"], 0.0)

    def test_process_technical_indicators_emits_new_columns(self):
        original_vwap = config["indicators"]["vwap"]["enabled"]
        original_or_enabled = config["indicators"].get("opening_range", {}).get("enabled", False)
        try:
            config["indicators"]["vwap"]["enabled"] = True
            config["indicators"].setdefault("opening_range", {})["enabled"] = True
            config["indicators"]["opening_range"]["minutes"] = 30

            df = process_technical_indicators(_sample_session_frame())

            for col in [
                "VWAP",
                "VWAP_DIST_PCT",
                "VWAP_DIST_Z",
                "VWAP_SLOPE",
                "VWAP_ABOVE",
                "OPENING_RANGE_HIGH",
                "OPENING_RANGE_LOW",
                "OPENING_RANGE_WIDTH_PCT",
                "DIST_TO_OR_HIGH_PCT",
                "DIST_TO_OR_LOW_PCT",
                "OR_BREAKOUT_DIR",
                "OR_BREAKOUT_ACTIVE",
                "OVERNIGHT_GAP_PCT",
                "OPEN_TO_PRIOR_RANGE_PCT",
                "POST_OPEN_VOL_PCT",
            ]:
                self.assertIn(col, df.columns)
                self.assertFalse(df[col].isna().any(), col)
        finally:
            config["indicators"]["vwap"]["enabled"] = original_vwap
            config["indicators"]["opening_range"]["enabled"] = original_or_enabled


if __name__ == "__main__":
    unittest.main(verbosity=2)
