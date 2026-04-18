#!/usr/bin/env python
"""Tests for train.evaluate_agent behavior under target-contract actions."""

import os
import sys
import unittest
from types import SimpleNamespace

import pandas as pd
import math
from unittest import mock

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


class _RecurrentSequenceModel:
    def __init__(self, actions):
        self._actions = list(actions)
        self._idx = 0
        self.policy = SimpleNamespace(lstm=object())
        self.calls = []

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        action = self._actions[min(self._idx, len(self._actions) - 1)]
        self.calls.append(
            {
                "state": state,
                "episode_start": episode_start.copy() if episode_start is not None else None,
                "deterministic": deterministic,
            }
        )
        self._idx += 1
        return action, {"hidden": self._idx}


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
        self.assertEqual(results["action_history"], [0, 2, 6, 6])
        self.assertEqual(results["trade_history"][1]["old_contracts"], 1)
        self.assertEqual(results["trade_history"][1]["new_contracts"], 3)
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

    def test_evaluate_agent_synthesizes_close_norm_from_close_variants(self):
        data = self._make_data().rename(
            columns={
                "close": "Close",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "volume": "Volume",
            }
        ).drop(columns=["close_norm"])

        results = evaluate_agent(_SequenceModel([0, 6, 6, 6]), data, verbose=0, deterministic=True)

        self.assertIn("close_norm", data.columns)
        self.assertTrue(data["close_norm"].between(0.0, 1.0).all())
        self.assertEqual(data["close_norm"].iloc[0], 0.5)
        self.assertEqual(data["close_norm"].iloc[1], 1.0)
        self.assertGreaterEqual(results["completed_trades"], 1)

    def test_evaluate_agent_handles_recurrent_models_and_rendering(self):
        model = _RecurrentSequenceModel([0, 6, 6, 6])

        with mock.patch("environment.TradingEnv.render") as render_mock:
            results = evaluate_agent(model, self._make_data(), verbose=0, deterministic=True, render=True)

        self.assertEqual(model.calls[0]["state"], None)
        self.assertTrue(bool(model.calls[0]["episode_start"][0]))
        self.assertEqual(model.calls[1]["state"], {"hidden": 1})
        self.assertTrue(all(call["deterministic"] for call in model.calls))
        self.assertGreaterEqual(render_mock.call_count, 1)
        self.assertGreaterEqual(results["trade_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
