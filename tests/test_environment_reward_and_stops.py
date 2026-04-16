#!/usr/bin/env python
"""Tests for environment reward shaping and stop/target branches."""

import copy
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config  # noqa: E402
from environment import TradingEnv  # noqa: E402


class TestEnvironmentRewardAndStops(unittest.TestCase):
    def setUp(self):
        self._config_backup = copy.deepcopy(config)
        config["execution_costs"]["half_spread_points"] = 0.0
        config["execution_costs"]["base_slippage_points"] = 0.0

    def tearDown(self):
        config.clear()
        config.update(self._config_backup)

    def _make_env(self, closes, *, highs=None, lows=None):
        highs = highs or closes
        lows = lows or closes
        rows = len(closes)
        df = pd.DataFrame(
            {
                "close_norm": [0.5] * rows,
                "close": closes,
                "open": closes,
                "high": highs,
                "low": lows,
                "ATR": [1.0] * rows,
                "ROLLING_DD": [0.0] * rows,
                "VOL_PERCENTILE": [50.0] * rows,
                "volume": [1.0] * rows,
            }
        )
        return TradingEnv(df, initial_balance=100000.0, transaction_cost=0.0, position_size=1)

    def test_flat_inactivity_penalty_applies_after_grace_period(self):
        config["reward"]["flat_time_penalty"] = 0.01
        config["reward"]["flat_time_grace_steps"] = 1

        env = self._make_env([100.0, 100.0, 100.0, 100.0])
        env.reset()

        _, reward1, _, _, _ = env.step(6)
        _, reward2, _, _, _ = env.step(6)

        self.assertEqual(reward1, 0.0)
        self.assertLess(reward2, 0.0)

    def test_calm_holding_bonus_rewards_staying_long_in_calm_regime(self):
        config["reward"]["calm_holding_bonus"] = 0.01

        env = self._make_env([100.0, 100.0, 100.0])
        env.reset()

        env.step(0)
        _, reward, _, _, info = env.step(0)

        self.assertFalse(info["position_changed"])
        self.assertGreater(reward, 0.0)

    def test_dynamic_sl_tp_exit_triggers_before_next_action(self):
        config["risk_management"]["dynamic_sl_tp"]["enabled"] = True
        config["risk_management"]["dynamic_sl_tp"]["num_choices"] = 4
        config["risk_management"]["dynamic_sl_tp"]["sl_multiplier_range"] = [1.5, 3.0]
        config["risk_management"]["dynamic_sl_tp"]["tp_multiplier_range"] = [1.5, 3.0]

        env = self._make_env(
            [100.0, 100.0, 100.0],
            highs=[100.0, 100.0, 100.0],
            lows=[100.0, 98.0, 100.0],
        )
        env.reset()

        env.step([0, 0, 0])
        _, _, _, _, info = env.step([6, 0, 0])

        self.assertEqual(info["sl_tp_exit"], "stop_loss")
        self.assertEqual(info["position"], 0)
        self.assertEqual(info["current_contracts"], 0)

    def test_invalid_market_prices_are_sanitized_before_accounting(self):
        df = pd.DataFrame(
            {
                "close_norm": [0.5, float("nan"), 0.7],
                "close": [100.0, float("nan"), 102.0],
                "open": [100.0, 101.0, 102.0],
                "high": [100.0, float("inf"), 102.5],
                "low": [100.0, float("nan"), 101.5],
                "ATR": [1.0, float("nan"), 1.0],
                "ROLLING_DD": [0.0, 0.0, 0.0],
                "VOL_PERCENTILE": [50.0, 50.0, 50.0],
                "volume": [1.0, 1.0, 1.0],
            }
        )
        env = TradingEnv(df, initial_balance=100000.0, transaction_cost=0.0, position_size=1)
        env.reset()

        env.step(0)
        _, _, _, _, info = env.step(6)

        self.assertTrue(env.net_worth.is_finite())
        self.assertTrue(env.max_net_worth.is_finite())
        self.assertGreaterEqual(info["net_worth"], 1000.0)

    def test_implausible_positive_price_outliers_are_sanitized(self):
        env = self._make_env(
            [30000.0, 175.0, 30100.0, 30200.0],
            highs=[30010.0, 175.0, 30110.0, 30210.0],
            lows=[29990.0, 175.0, 30090.0, 30190.0],
        )

        self.assertGreater(env._close_array[1], 1000.0)
        self.assertEqual(env._close_array[1], env._close_array[0])

        env.reset()
        _, _, _, _, info = env.step(2)

        self.assertLessEqual(abs(info["current_contracts"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
