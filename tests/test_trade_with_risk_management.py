#!/usr/bin/env python
"""Tests for the top-level trade_with_risk_management execution path."""

import os
import sys
import unittest
import copy
import importlib
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import action_space as action_space_module  # noqa: E402
from config import config  # noqa: E402
from trade import trade_with_risk_management  # noqa: E402


class _SequenceModel:
    def __init__(self, actions):
        self._actions = list(actions)
        self._idx = 0

    def predict(self, obs, deterministic=True):
        action = self._actions[min(self._idx, len(self._actions) - 1)]
        self._idx += 1
        return action, None


class TestTradeWithRiskManagement(unittest.TestCase):
    def setUp(self):
        self._action_cfg = copy.deepcopy(config.get("action_space", {}))
        config.setdefault("action_space", {})["mode"] = "fixed_contracts"
        importlib.reload(action_space_module)

    def tearDown(self):
        config["action_space"] = self._action_cfg
        importlib.reload(action_space_module)

    def _make_data(self):
        idx = pd.date_range("2026-01-02 09:30:00", periods=5, freq="1min", tz="UTC")
        closes = [100.0, 101.0, 102.0, 103.0, 104.0]
        return pd.DataFrame(
            {
                "close_norm": [0.1, 0.2, 0.3, 0.4, 0.5],
                "close": closes,
                "open": closes,
                "high": closes,
                "low": closes,
                "ATR": [1.0] * len(closes),
                "volume": [1.0] * len(closes),
            },
            index=idx,
        )

    def _make_atr_stop_data(self):
        idx = pd.date_range("2026-01-02 09:30:00", periods=4, freq="1min", tz="UTC")
        return pd.DataFrame(
            {
                "close_norm": [0.1, 0.2, 0.3, 0.4],
                "close": [100.0, 99.0, 99.0, 99.0],
                "open": [100.0, 99.0, 99.0, 99.0],
                "high": [100.5, 100.0, 99.0, 99.0],
                "low": [99.5, 97.0, 99.0, 99.0],
                "ATR": [1.0, 1.0, 1.0, 1.0],
                "volume": [1.0, 1.0, 1.0, 1.0],
            },
            index=idx,
        )

    def _make_day_boundary_data(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-01-02 20:59:00", tz="UTC"),
                pd.Timestamp("2026-01-03 14:30:00", tz="UTC"),
                pd.Timestamp("2026-01-03 14:31:00", tz="UTC"),
            ]
        )
        closes = [100.0, 101.0, 101.0]
        return pd.DataFrame(
            {
                "close_norm": [0.1, 0.2, 0.3],
                "close": closes,
                "open": closes,
                "high": closes,
                "low": closes,
                "ATR": [1.0, 1.0, 1.0],
                "volume": [1.0, 1.0, 1.0],
            },
            index=idx,
        )

    def _make_daily_risk_limit_data(self):
        idx = pd.date_range("2026-01-02 09:30:00", periods=4, freq="1min", tz="UTC")
        closes = [100.0, 95.0, 94.0, 94.0]
        return pd.DataFrame(
            {
                "close_norm": [0.1, 0.2, 0.3, 0.4],
                "close": closes,
                "open": closes,
                "high": closes,
                "low": closes,
                "ATR": [1.0, 1.0, 1.0, 1.0],
                "volume": [1.0, 1.0, 1.0, 1.0],
            },
            index=idx,
        )

    def test_model_flat_exit_records_long_close_as_sell_marker(self):
        with mock.patch("trade.PPO.load", return_value=_SequenceModel([0, 6, 6, 6])):
            results = trade_with_risk_management(
                model_path="dummy.zip",
                test_data=self._make_data(),
                initial_balance=100000.0,
                transaction_cost=0.0,
                close_at_end_of_day=False,
                deterministic=True,
            )

        self.assertEqual(results["action_counts"][0], 1)
        self.assertGreaterEqual(results["action_counts"][6], 1)
        self.assertEqual(results["exit_reasons"]["model_flat"], 1)
        self.assertEqual(len(results["buy_dates"]), 1)
        self.assertEqual(len(results["sell_dates"]), 1)
        self.assertEqual(results["final_position"], 0)

    def test_model_flat_exit_records_short_close_as_buy_marker(self):
        with mock.patch("trade.PPO.load", return_value=_SequenceModel([3, 6, 6, 6])):
            results = trade_with_risk_management(
                model_path="dummy.zip",
                test_data=self._make_data(),
                initial_balance=100000.0,
                transaction_cost=0.0,
                close_at_end_of_day=False,
                deterministic=True,
            )

        self.assertEqual(results["exit_reasons"]["model_flat"], 1)
        self.assertEqual(len(results["sell_dates"]), 1)
        self.assertEqual(len(results["buy_dates"]), 1)
        self.assertEqual(results["trade_history"][-1]["action"], "buy")

    def test_second_to_last_candle_exit_records_long_close_as_sell_marker(self):
        with mock.patch("trade.PPO.load", return_value=_SequenceModel([0, 0, 0, 0])):
            results = trade_with_risk_management(
                model_path="dummy.zip",
                test_data=self._make_data(),
                initial_balance=100000.0,
                transaction_cost=0.0,
                close_at_end_of_day=False,
                deterministic=True,
            )

        self.assertEqual(results["exit_reasons"]["second_to_last_candle"], 1)
        self.assertEqual(len(results["buy_dates"]), 1)
        self.assertEqual(len(results["sell_dates"]), 1)
        self.assertEqual(results["trade_history"][-1]["exit_reason"], "second_to_last_candle")

    def test_atr_stop_exit_uses_precomputed_stop_price(self):
        with mock.patch("trade.PPO.load", return_value=_SequenceModel([0, 6, 6])):
            results = trade_with_risk_management(
                model_path="dummy.zip",
                test_data=self._make_atr_stop_data(),
                initial_balance=100000.0,
                transaction_cost=0.0,
                close_at_end_of_day=False,
                deterministic=True,
                stop_loss_mode="atr",
                stop_loss_atr_multiplier=2.0,
            )

        self.assertEqual(results["exit_reasons"]["stop_loss"], 1)
        self.assertAlmostEqual(results["trade_history"][-1]["price"], 98.0)
        self.assertEqual(results["trade_history"][-1]["exit_reason"], "stop_loss")

    def test_end_of_day_close_happens_on_day_boundary(self):
        with mock.patch("trade.PPO.load", return_value=_SequenceModel([0, 6])):
            results = trade_with_risk_management(
                model_path="dummy.zip",
                test_data=self._make_day_boundary_data(),
                initial_balance=100000.0,
                transaction_cost=0.0,
                close_at_end_of_day=True,
                deterministic=True,
            )

        self.assertEqual(results["exit_reasons"]["end_of_day"], 1)
        self.assertEqual(results["trade_history"][-1]["exit_reason"], "end_of_day")
        self.assertEqual(results["final_position"], 0)

    def test_daily_risk_limit_forces_close_after_partial_loss(self):
        with mock.patch("trade.PPO.load", return_value=_SequenceModel([2, 0, 0])):
            results = trade_with_risk_management(
                model_path="dummy.zip",
                test_data=self._make_daily_risk_limit_data(),
                initial_balance=100000.0,
                transaction_cost=0.0,
                close_at_end_of_day=False,
                deterministic=True,
                daily_risk_limit=0.5,
            )

        self.assertEqual(results["exit_reasons"]["daily_risk_limit"], 1)
        self.assertEqual(results["trade_history"][-1]["exit_reason"], "daily_risk_limit")
        self.assertEqual(results["final_position"], 0)

    def test_normal_profitable_exit_is_not_flagged_unrealistic(self):
        with mock.patch("trade.PPO.load", return_value=_SequenceModel([2, 2, 2])):
            results = trade_with_risk_management(
                model_path="dummy.zip",
                test_data=self._make_data(),
                initial_balance=100000.0,
                transaction_cost=0.0,
                close_at_end_of_day=False,
                deterministic=True,
            )

        self.assertEqual(results["trade_history"][-1]["exit_reason"], "second_to_last_candle")
        self.assertGreater(results["trade_history"][-1]["profit"], 0.0)
        self.assertEqual(results["unrealistic_profit_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
