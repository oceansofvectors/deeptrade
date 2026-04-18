#!/usr/bin/env python
"""Unit tests for derived, non-plain-TA indicator modules."""

import math
import os
import sys
import unittest

import numpy as np
import pandas as pd


PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from indicators.day_of_week import calculate_day_of_week  # noqa: E402
from indicators.minutes_since_open import calculate_minutes_since_open  # noqa: E402
from indicators.rolling_drawdown import calculate_rolling_drawdown  # noqa: E402
from indicators.vol_percentile import calculate_vol_percentile  # noqa: E402
from indicators.z_score import calculate_zscore  # noqa: E402


class TestDayOfWeekFeatures(unittest.TestCase):
    def test_day_of_week_uses_datetime_index_and_circular_encoding(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")  # Mon/Tue/Wed
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)

        result = calculate_day_of_week(df.copy())

        self.assertEqual(result["DOW"].tolist(), [0, 1, 2])
        self.assertAlmostEqual(result.iloc[0]["DOW_SIN"], 0.0, places=6)
        self.assertAlmostEqual(result.iloc[0]["DOW_COS"], 1.0, places=6)
        self.assertAlmostEqual(result.iloc[1]["DOW_SIN"], math.sin(2 * math.pi / 7), places=6)
        self.assertAlmostEqual(result.iloc[1]["DOW_COS"], math.cos(2 * math.pi / 7), places=6)

    def test_day_of_week_falls_back_to_zeroes_on_unparseable_index(self):
        df = pd.DataFrame({"close": [1.0, 2.0]}, index=["bad", "worse"])

        result = calculate_day_of_week(df.copy())

        self.assertEqual(result["DOW"].tolist(), [0.0, 0.0])
        self.assertEqual(result["DOW_SIN"].tolist(), [0.0, 0.0])
        self.assertEqual(result["DOW_COS"].tolist(), [1.0, 1.0])


class TestMinutesSinceOpenFeatures(unittest.TestCase):
    def test_minutes_since_open_respects_session_anchor_and_encodes_cyclically(self):
        local_idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-02 17:00:00", tz="America/Chicago"),
                pd.Timestamp("2024-01-02 18:30:00", tz="America/Chicago"),
                pd.Timestamp("2024-01-03 15:59:00", tz="America/Chicago"),
            ]
        )
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=local_idx.tz_convert("UTC"))

        result = calculate_minutes_since_open(
            df.copy(),
            session_config={
                "timezone": "America/Chicago",
                "open_hour": 17,
                "open_minute": 0,
                "close_hour": 16,
                "close_minute": 0,
            },
        )

        session_minutes = 23 * 60
        expected_norms = [0.0, 90.0 / session_minutes, 1379.0 / session_minutes]
        for i, norm in enumerate(expected_norms):
            self.assertAlmostEqual(result.iloc[i]["MSO_SIN"], math.sin(2 * math.pi * norm), places=6)
            self.assertAlmostEqual(result.iloc[i]["MSO_COS"], math.cos(2 * math.pi * norm), places=6)

    def test_minutes_since_open_falls_back_on_invalid_index(self):
        df = pd.DataFrame({"close": [1.0, 2.0]}, index=["bad", "worse"])

        result = calculate_minutes_since_open(df.copy())

        self.assertEqual(result["MSO_SIN"].tolist(), [0.0, 0.0])
        self.assertEqual(result["MSO_COS"].tolist(), [1.0, 1.0])


class TestRollingDrawdownFeatures(unittest.TestCase):
    def test_rolling_drawdown_matches_expected_percentages(self):
        df = pd.DataFrame({"close": [100.0, 110.0, 105.0, 120.0, 90.0]})

        result = calculate_rolling_drawdown(df.copy(), window=3)

        expected = [0.0, 0.0, (105.0 - 110.0) / 110.0 * 100.0, 0.0, (90.0 - 120.0) / 120.0 * 100.0]
        for actual, exp in zip(result["ROLLING_DD"].tolist(), expected):
            self.assertAlmostEqual(actual, exp, places=6)

    def test_rolling_drawdown_uses_case_insensitive_close_and_missing_close_fallback(self):
        upper = pd.DataFrame({"Close": [100.0, 95.0]})
        result_upper = calculate_rolling_drawdown(upper.copy(), window=2)
        self.assertLess(result_upper.iloc[1]["ROLLING_DD"], 0.0)

        missing = pd.DataFrame({"open": [1.0, 2.0]})
        result_missing = calculate_rolling_drawdown(missing.copy(), window=2)
        self.assertEqual(result_missing["ROLLING_DD"].tolist(), [0.0, 0.0])


class TestVolatilityPercentileFeatures(unittest.TestCase):
    def test_vol_percentile_is_bounded_and_rises_after_volatility_spike(self):
        df = pd.DataFrame({"close": [100.0, 100.2, 100.4, 100.6, 110.0, 109.5, 109.8]})

        result = calculate_vol_percentile(df.copy(), window=3)

        self.assertTrue(((result["VOL_PERCENTILE"] >= 0.0) & (result["VOL_PERCENTILE"] <= 100.0)).all())
        self.assertEqual(result.iloc[4]["VOL_PERCENTILE"], 100.0)
        self.assertLess(result.iloc[6]["VOL_PERCENTILE"], result.iloc[4]["VOL_PERCENTILE"])

    def test_vol_percentile_handles_bad_prices_and_missing_close(self):
        df = pd.DataFrame({"close": [100.0, 0.0, -5.0, 101.0, 102.0]})
        result = calculate_vol_percentile(df.copy(), window=3)
        self.assertTrue(result["VOL_PERCENTILE"].notna().all())
        self.assertTrue(((result["VOL_PERCENTILE"] >= 0.0) & (result["VOL_PERCENTILE"] <= 100.0)).all())

        missing = pd.DataFrame({"open": [1.0, 2.0]})
        result_missing = calculate_vol_percentile(missing.copy(), window=2)
        self.assertEqual(result_missing["VOL_PERCENTILE"].tolist(), [50.0, 50.0])


class TestZScoreFeatures(unittest.TestCase):
    def test_zscore_uses_case_insensitive_close_and_clips_extremes(self):
        df = pd.DataFrame({"CLOSE": [100.0, 100.0, 100.0, 1000.0]})

        result = calculate_zscore(df.copy(), length=3)

        self.assertIn("ZScore", result.columns)
        self.assertLessEqual(result["ZScore"].max(), 4.0)
        self.assertGreaterEqual(result["ZScore"].min(), -4.0)

    def test_zscore_returns_zeroes_when_close_missing_or_constant(self):
        constant = pd.DataFrame({"close": [100.0, 100.0, 100.0, 100.0]})
        result_constant = calculate_zscore(constant.copy(), length=3)
        self.assertTrue(np.isfinite(result_constant["ZScore"]).all())
        self.assertTrue((result_constant["ZScore"] == 0.0).all())

        missing = pd.DataFrame({"open": [1.0, 2.0]})
        result_missing = calculate_zscore(missing.copy(), length=3)
        self.assertEqual(result_missing["ZScore"].tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
