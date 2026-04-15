#!/usr/bin/env python
"""Tests for iterative training diagnostics helpers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from train import (  # noqa: E402
    _calculate_sortino_ratio,
    _extract_metric_value,
    _infer_periods_per_year,
    _with_training_diagnostics,
)


class TestExtractMetricValue(unittest.TestCase):
    def test_sortino_metric(self):
        value, name = _extract_metric_value({"sortino_ratio": 1.5}, "sortino")
        self.assertEqual(value, 1.5)
        self.assertEqual(name, "sortino")

    def test_return_metric_default(self):
        value, name = _extract_metric_value({"total_return_pct": 7.2}, "return")
        self.assertEqual(value, 7.2)
        self.assertEqual(name, "return")


class TestTrainingDiagnostics(unittest.TestCase):
    def test_flags_dominant_action_and_metric_drop(self):
        results = {
            "trade_count": 4,
            "action_counts": {0: 95, 1: 3, 2: 2},
            "sortino_ratio": -3.0,
        }
        enriched = _with_training_diagnostics(
            results,
            metric_name="sortino",
            metric_value=-3.0,
            loss_info={"value_loss": 123.0},
            min_trades=20,
            best_metric_value=4.0,
        )

        self.assertAlmostEqual(enriched["long_action_pct"], 95.0)
        self.assertFalse(enriched["has_enough_trades"])
        self.assertTrue(enriched["warning_policy_collapse"])
        self.assertTrue(enriched["warning_metric_drop"])
        self.assertIn("dominant_action", enriched["collapse_flags"])
        self.assertIn("too_few_trades", enriched["collapse_flags"])
        self.assertIn("metric_drop", enriched["collapse_flags"])

    def test_balanced_actions_do_not_flag_collapse(self):
        results = {
            "trade_count": 120,
            "action_counts": {0: 35, 1: 30, 2: 35},
            "total_return_pct": 2.0,
        }
        enriched = _with_training_diagnostics(
            results,
            metric_name="return",
            metric_value=2.0,
            loss_info={},
            min_trades=20,
            best_metric_value=2.5,
        )

        self.assertTrue(enriched["has_enough_trades"])
        self.assertTrue(enriched["has_enough_action_mix"])
        self.assertFalse(enriched["warning_policy_collapse"])
        self.assertNotIn("dominant_action", enriched["collapse_flags"])

    def test_flat_dominance_rejects_action_mix(self):
        results = {
            "trade_count": 25,
            "action_counts": {0: 5, 1: 10, 2: 85},
            "total_return_pct": 0.5,
        }
        enriched = _with_training_diagnostics(
            results,
            metric_name="return",
            metric_value=0.5,
            loss_info={},
            min_trades=20,
            best_metric_value=1.0,
        )

        self.assertTrue(enriched["has_enough_trades"])
        self.assertFalse(enriched["has_enough_action_mix"])
        self.assertIn("flat_dominance", enriched["collapse_flags"])
        self.assertTrue(enriched["warning_policy_collapse"])

    def test_infer_periods_per_year_for_one_minute_data(self):
        idx = pd.date_range("2026-01-01", periods=10, freq="1min", tz="UTC")
        periods = _infer_periods_per_year(idx)
        self.assertGreater(periods, 500000)
        self.assertLess(periods, 530000)

    def test_sortino_is_negative_for_crash_path(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="1min", tz="UTC")
        portfolio = np.array([100000.0, 101000.0, 102000.0, 103000.0, 30000.0])
        sortino = _calculate_sortino_ratio(portfolio, idx)
        self.assertLess(sortino, 0.0)

    def test_sortino_is_positive_for_smooth_up_path(self):
        idx = pd.date_range("2026-01-01", periods=6, freq="1min", tz="UTC")
        portfolio = np.array([100000.0, 100500.0, 101000.0, 101600.0, 102300.0, 103000.0])
        sortino = _calculate_sortino_ratio(portfolio, idx)
        self.assertGreater(sortino, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
