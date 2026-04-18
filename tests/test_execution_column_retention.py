#!/usr/bin/env python
"""Tests for OHLC column retention in execution-dependent pipelines."""

import copy
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config  # noqa: E402
from train_current_model import drop_redundant_columns  # noqa: E402
from walk_forward import _drop_unused_model_columns  # noqa: E402


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [10.0, 11.0],
            "close_norm": [0.5, 0.6],
            "ATR": [1.0, 1.1],
            "ATR_RAW": [1.0, 1.1],
        }
    )


class TestExecutionColumnRetention(unittest.TestCase):
    def setUp(self):
        self._config_backup = copy.deepcopy(config)

    def tearDown(self):
        config.clear()
        config.update(self._config_backup)

    def test_walk_forward_keeps_high_low_when_fixed_take_profit_enabled(self):
        config["risk_management"]["enabled"] = False
        config["risk_management"]["dynamic_sl_tp"]["enabled"] = False
        config["risk_management"]["stop_loss"]["enabled"] = False
        config["risk_management"]["take_profit"]["enabled"] = True

        df = _sample_frame()
        _drop_unused_model_columns([df])

        self.assertIn("high", df.columns)
        self.assertIn("low", df.columns)
        self.assertIn("close", df.columns)

    def test_walk_forward_drops_high_low_when_no_execution_branch_needs_them(self):
        config["risk_management"]["enabled"] = False
        config["risk_management"]["dynamic_sl_tp"]["enabled"] = False
        config["risk_management"]["stop_loss"]["enabled"] = False
        config["risk_management"]["take_profit"]["enabled"] = False
        config["risk_management"]["trailing_stop"]["enabled"] = False

        df = _sample_frame()
        _drop_unused_model_columns([df])

        self.assertNotIn("open", df.columns)
        self.assertNotIn("high", df.columns)
        self.assertNotIn("low", df.columns)
        self.assertIn("close", df.columns)

    def test_current_model_bundle_keeps_close_even_without_stop_logic(self):
        config["risk_management"]["enabled"] = False
        config["risk_management"]["dynamic_sl_tp"]["enabled"] = False
        config["risk_management"]["stop_loss"]["enabled"] = False
        config["risk_management"]["take_profit"]["enabled"] = False
        config["risk_management"]["trailing_stop"]["enabled"] = False

        df = _sample_frame()
        drop_redundant_columns(df)

        self.assertNotIn("open", df.columns)
        self.assertNotIn("high", df.columns)
        self.assertNotIn("low", df.columns)
        self.assertIn("close", df.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
