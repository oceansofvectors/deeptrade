#!/usr/bin/env python
"""Tests for data loading defaults and shared market-hours behavior."""

import copy
import os
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config  # noqa: E402
from get_data import ensure_numeric, get_data  # noqa: E402
from utils.data_utils import filter_market_hours, is_rth_bar, market_hours_only_enabled  # noqa: E402


class TestDataUtils(unittest.TestCase):
    def test_market_hours_only_enabled_defaults_true(self):
        self.assertTrue(market_hours_only_enabled({}))
        self.assertTrue(market_hours_only_enabled({"data": {}}))
        self.assertTrue(market_hours_only_enabled({"data": None}))

    def test_market_hours_only_enabled_respects_false(self):
        self.assertFalse(market_hours_only_enabled({"data": {"market_hours_only": False}}))
        self.assertTrue(market_hours_only_enabled({"data": {"market_hours_only": True}}))

    def test_is_rth_bar_handles_naive_and_weekend_values(self):
        self.assertTrue(is_rth_bar(pd.Timestamp("2026-01-05 15:30:00")))
        self.assertFalse(is_rth_bar(pd.Timestamp("2026-01-03 15:30:00", tz="UTC")))
        self.assertFalse(is_rth_bar(pd.Timestamp("2026-01-05 22:30:00", tz="UTC")))

    def test_filter_market_hours_keeps_only_weekday_nyse_rth_rows(self):
        idx = pd.to_datetime(
            [
                "2026-01-05 14:29:00+00:00",  # 9:29 ET
                "2026-01-05 14:30:00+00:00",  # 9:30 ET
                "2026-01-05 21:00:00+00:00",  # 16:00 ET
                "2026-01-05 21:01:00+00:00",  # 16:01 ET
                "2026-01-03 15:00:00+00:00",  # Saturday
            ],
            utc=True,
        )
        df = pd.DataFrame({"close": [1, 2, 3, 4, 5]}, index=idx)

        filtered = filter_market_hours(df)

        self.assertEqual(filtered["close"].tolist(), [2, 3])
        self.assertTrue((filtered.index.tz.zone if hasattr(filtered.index.tz, "zone") else str(filtered.index.tz)).upper().startswith("UTC"))


class TestGetDataLoading(unittest.TestCase):
    def setUp(self):
        self._config_backup = copy.deepcopy(config)

    def tearDown(self):
        config.clear()
        config.update(self._config_backup)

    def test_ensure_numeric_drop_invalid_drops_bad_rows(self):
        df = pd.DataFrame(
            {
                "open": ["1.0", "oops", "2.0"],
                "close": ["3.0", "4.0", "bad"],
            }
        )

        cleaned = ensure_numeric(df, ["open", "close"], drop_invalid=True)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["open"], 1.0)
        self.assertEqual(cleaned.iloc[0]["close"], 3.0)

    def test_get_data_defaults_to_market_hours_filter_when_setting_missing(self):
        raw = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05 14:30:00", periods=60, freq="1min", tz="UTC"),
                "open": range(60),
                "high": range(1, 61),
                "low": range(60),
                "close": range(1, 61),
                "volume": [10] * 60,
            }
        )
        processed = raw.set_index("timestamp").copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample.csv")
            raw.to_csv(path, index=False)
            config["data"]["csv_path"] = path
            config["data"].pop("market_hours_only", None)

            with mock.patch("get_data.process_technical_indicators", return_value=processed), \
                 mock.patch("get_data.filter_market_hours", side_effect=lambda frame: frame) as filter_mock:
                train_df, validation_df, test_df = get_data(train_ratio=0.6, validation_ratio=0.2)

        filter_mock.assert_called_once()
        self.assertIsNotNone(train_df)
        self.assertGreater(len(train_df) + len(validation_df) + len(test_df), 0)

    def test_get_data_skips_market_hours_filter_when_disabled(self):
        raw = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05 14:30:00", periods=60, freq="1min", tz="UTC"),
                "open": range(60),
                "high": range(1, 61),
                "low": range(60),
                "close": range(1, 61),
                "volume": [10] * 60,
            }
        )
        processed = raw.set_index("timestamp").copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sample.csv")
            raw.to_csv(path, index=False)
            config["data"]["csv_path"] = path
            config["data"]["market_hours_only"] = False

            with mock.patch("get_data.process_technical_indicators", return_value=processed), \
                 mock.patch("get_data.filter_market_hours") as filter_mock:
                train_df, validation_df, test_df = get_data(train_ratio=0.6, validation_ratio=0.2)

        filter_mock.assert_not_called()
        self.assertIsNotNone(train_df)
        self.assertGreater(len(train_df) + len(validation_df) + len(test_df), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
