#!/usr/bin/env python
"""Tests for train.evaluate_agent behavior under target-allocation actions."""

import os
import sys
import unittest

import pandas as pd
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train import evaluate_agent  # noqa: E402


class _SequenceModel:
    def __init__(self, actions):
        self._actions = list(actions)
        self._idx = 0

    def predict(self, obs, deterministic=True):
        action = self._actions[min(self._idx, len(self._actions) - 1)]
        self._idx += 1
        return action, None


class TestEvaluateAgent(unittest.TestCase):
    def _make_data(self):
        idx = pd.date_range("2026-01-01 00:00:00", periods=5, freq="1min", tz="UTC")
        closes = [100.0, 110.0, 120.0, 120.0, 120.0]
        return pd.DataFrame(
            {
                "close_norm": [0.4, 0.5, 0.6, 0.6, 0.6],
                "close": closes,
                "open": closes,
                "high": closes,
                "low": closes,
                "volume": [1.0] * len(closes),
            },
            index=idx,
        )

    def test_evaluate_agent_counts_scale_in_as_trade_event(self):
        model = _SequenceModel([0, 2, 6, 6])
        results = evaluate_agent(model, self._make_data(), verbose=0, deterministic=True)

        self.assertEqual(results["trade_count"], 3)
        self.assertEqual(results["completed_trades"], 1)
        self.assertEqual(results["hit_rate"], 100.0)
        self.assertEqual(results["final_position"], 0)
        self.assertEqual(results["action_counts"][0], 1)
        self.assertEqual(results["action_counts"][2], 1)
        self.assertEqual(results["action_counts"][6], 2)
        self.assertEqual(
            [trade["trade_type"] for trade in results["trade_history"]],
            ["Long Entry", "Scale In", "Exit"],
        )
        self.assertEqual(results["trade_history"][1]["old_contracts"], 100)
        self.assertEqual(results["trade_history"][1]["new_contracts"], 454)
        self.assertFalse(results["trade_history"][1]["realized_trade"])

    def test_evaluate_agent_tracks_scale_out_and_flip_as_realized_trades(self):
        model = _SequenceModel([2, 0, 3, 6])
        results = evaluate_agent(model, self._make_data(), verbose=0, deterministic=True)

        self.assertEqual(
            [trade["trade_type"] for trade in results["trade_history"]],
            ["Long Entry", "Scale Out", "Flip", "Exit"],
        )
        self.assertEqual(results["completed_trades"], 3)
        self.assertGreaterEqual(results["hit_rate"], 66.0)
        self.assertTrue(results["trade_history"][1]["realized_trade"])
        self.assertTrue(results["trade_history"][2]["realized_trade"])

    def test_evaluate_agent_handles_non_finite_market_rows_without_crashing(self):
        data = self._make_data()
        data.loc[data.index[1], "close"] = float("nan")
        data.loc[data.index[1], "high"] = float("inf")
        data.loc[data.index[1], "low"] = float("nan")
        data.loc[data.index[1], "close_norm"] = float("nan")

        results = evaluate_agent(_SequenceModel([0, 6, 6, 6]), data, verbose=0, deterministic=True)

        self.assertTrue(math.isfinite(results["final_portfolio_value"]))
        self.assertTrue(math.isfinite(results["total_return_pct"]))
        self.assertGreaterEqual(results["final_portfolio_value"], 1000.0)
        self.assertGreaterEqual(results["completed_trades"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
