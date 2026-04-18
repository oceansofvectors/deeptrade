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

    def _make_env(self, closes, *, highs=None, lows=None, atr_raw=1.0, transaction_cost=0.0):
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
                "ATR_RAW": [atr_raw] * rows,
                "ROLLING_DD": [0.0] * rows,
                "ROLLING_DD_RAW": [0.0] * rows,
                "VOL_PERCENTILE": [50.0] * rows,
                "VOL_PERCENTILE_RAW": [50.0] * rows,
                "volume": [1.0] * rows,
            }
        )
        return TradingEnv(df, initial_balance=100000.0, transaction_cost=transaction_cost, position_size=1)

    def test_flat_inactivity_penalty_applies_after_grace_period(self):
        config["reward"]["flat_time_penalty"] = 0.01
        config["reward"]["flat_time_grace_steps"] = 1
        config["reward"]["flat_penalty_opportunity_vol_percentile"] = 40.0

        env = self._make_env([100.0, 100.0, 100.0, 100.0])
        env.reset()

        _, reward1, _, _, _ = env.step(6)
        _, reward2, _, _, _ = env.step(6)

        self.assertEqual(reward1, 0.0)
        self.assertLess(reward2, 0.0)

    def test_flat_inactivity_penalty_uses_current_bar_regime_not_next_bar(self):
        config["reward"]["flat_time_penalty"] = 0.01
        config["reward"]["flat_time_grace_steps"] = 0
        config["reward"]["flat_penalty_opportunity_vol_percentile"] = 50.0

        env = self._make_env([100.0, 100.0, 100.0])
        env.data.loc[:, "VOL_PERCENTILE_RAW"] = [0.0, 100.0, 100.0]
        env._vol_pct_raw_array = env.data["VOL_PERCENTILE_RAW"].to_numpy(dtype=float)
        env.reset()

        _, reward, _, _, _ = env.step(6)

        self.assertEqual(reward, 0.0)

    def test_holding_position_without_price_change_gets_no_calm_bonus(self):
        env = self._make_env([100.0, 100.0, 100.0])
        env.reset()

        env.step(0)
        _, reward, _, _, info = env.step(0)

        self.assertFalse(info["position_changed"])
        self.assertEqual(reward, 0.0)

    def test_atr_risk_actions_size_contracts_from_net_worth_and_atr(self):
        env = self._make_env([100.0, 100.0, 100.0], atr_raw=1000.0)
        env.reset()

        _, _, _, _, info = env.step(1)

        self.assertEqual(info["current_contracts"], 5)

    def test_fixed_atr_stops_use_raw_atr_units(self):
        config["risk_management"]["stop_loss"] = {"enabled": True, "mode": "atr", "atr_multiplier": 2.0}
        config["risk_management"]["take_profit"] = {"enabled": True, "mode": "atr", "atr_multiplier": 3.0}

        env = self._make_env([100.0, 100.0, 100.0], atr_raw=25.0)
        env.reset()

        env.step(0)

        self.assertEqual(float(env.fixed_sl_price), 50.0)
        self.assertEqual(float(env.fixed_tp_price), 175.0)

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
                "ATR_RAW": [1.0, float("nan"), 1.0],
                "ROLLING_DD": [0.0, 0.0, 0.0],
                "ROLLING_DD_RAW": [0.0, 0.0, 0.0],
                "VOL_PERCENTILE": [50.0, 50.0, 50.0],
                "VOL_PERCENTILE_RAW": [50.0, 50.0, 50.0],
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
            atr_raw=1000.0,
        )

        self.assertGreater(env._close_array[1], 1000.0)
        self.assertEqual(env._close_array[1], env._close_array[0])

        env.reset()
        _, _, _, _, info = env.step(2)

        self.assertEqual(abs(info["current_contracts"]), 7)

    def test_realized_trade_quality_bonus_rewards_profitable_exit(self):
        config["reward"]["realized_trade_quality_bonus"] = 0.5
        env = self._make_env([100.0, 105.0, 105.0], atr_raw=2.5)
        env.reset()

        env.step(0)
        _, reward, _, _, info = env.step(6)

        self.assertEqual(info["current_contracts"], 0)
        self.assertGreater(reward, 0.0)

    def test_fixed_take_profit_uses_next_bar_range_not_close_only(self):
        config["risk_management"]["stop_loss"] = {"enabled": False, "mode": "percentage", "percentage": 17}
        config["risk_management"]["take_profit"] = {"enabled": True, "mode": "percentage", "percentage": 5}

        env = self._make_env(
            [100.0, 100.0, 100.0],
            highs=[100.0, 106.0, 100.0],
            lows=[100.0, 99.5, 100.0],
        )
        env.reset()

        _, _, _, _, info = env.step(0)

        self.assertTrue(info["fixed_tp_triggered"])
        self.assertEqual(info["fixed_exit_reason"], "take_profit")
        self.assertEqual(info["position"], 0)
        self.assertEqual(info["current_contracts"], 0)
        self.assertEqual(info["fixed_exit_price"], 105.0)

    def test_fixed_exit_price_flows_into_realized_trade_quality_bonus(self):
        config["reward"]["realized_trade_quality_bonus"] = 0.5
        config["risk_management"]["stop_loss"] = {"enabled": False, "mode": "percentage", "percentage": 17}
        config["risk_management"]["take_profit"] = {"enabled": True, "mode": "percentage", "percentage": 5}

        env = self._make_env(
            [100.0, 100.0, 100.0],
            highs=[100.0, 106.0, 100.0],
            lows=[100.0, 99.5, 100.0],
            atr_raw=2.5,
        )
        env.reset()

        _, reward, _, _, info = env.step(0)

        self.assertTrue(info["fixed_tp_triggered"])
        self.assertGreater(reward, 0.0)

    def test_fixed_exit_charges_entry_and_exit_execution_costs(self):
        config["risk_management"]["stop_loss"] = {"enabled": False, "mode": "percentage", "percentage": 17}
        config["risk_management"]["take_profit"] = {"enabled": True, "mode": "percentage", "percentage": 5}

        env = self._make_env(
            [100.0, 100.0, 100.0],
            highs=[100.0, 106.0, 100.0],
            lows=[100.0, 99.5, 100.0],
            atr_raw=2000.0,
            transaction_cost=1.0,
        )
        env.reset()

        _, _, _, _, info = env.step(0)

        self.assertTrue(info["fixed_tp_triggered"])
        self.assertEqual(info["current_contracts"], 0)
        self.assertEqual(info["contracts_traded"], 2)
        self.assertEqual(info["transaction_cost"], 2.0)

    def test_fixed_exit_without_rebalance_still_charges_exit_cost(self):
        config["risk_management"]["stop_loss"] = {"enabled": False, "mode": "percentage", "percentage": 17}
        config["risk_management"]["take_profit"] = {"enabled": True, "mode": "percentage", "percentage": 5}

        env = self._make_env(
            [100.0, 100.0, 100.0, 100.0],
            highs=[100.0, 101.0, 106.0, 100.0],
            lows=[100.0, 99.5, 99.5, 100.0],
            atr_raw=2000.0,
            transaction_cost=1.0,
        )
        env.reset()

        env.step(0)
        _, _, _, _, info = env.step(0)

        self.assertTrue(info["fixed_tp_triggered"])
        self.assertEqual(info["current_contracts"], 0)
        self.assertEqual(info["contracts_traded"], 1)
        self.assertEqual(info["transaction_cost"], 1.0)

    def test_latent_only_observation_excludes_close_norm(self):
        config["representation"]["mode"] = "latent_only"
        rows = 4
        df = pd.DataFrame(
            {
                "close_norm": [0.2] * rows,
                "close": [100.0, 101.0, 102.0, 103.0],
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [100.0, 101.0, 102.0, 103.0],
                "low": [100.0, 101.0, 102.0, 103.0],
                "volume": [1.0] * rows,
                "ATR": [1.0] * rows,
                "ATR_RAW": [1.0] * rows,
                "ROLLING_DD": [0.0] * rows,
                "ROLLING_DD_RAW": [0.0] * rows,
                "VOL_PERCENTILE": [50.0] * rows,
                "VOL_PERCENTILE_RAW": [50.0] * rows,
                "LATENT_F0": [0.1, 0.2, 0.3, 0.4],
                "LATENT_F1": [0.4, 0.3, 0.2, 0.1],
            }
        )

        env = TradingEnv(df, initial_balance=100000.0, transaction_cost=0.0, position_size=1)
        obs, _ = env.reset()

        self.assertEqual(env.observation_space.shape[0], 6)
        self.assertAlmostEqual(obs[0], 0.1)
        self.assertAlmostEqual(obs[1], 0.4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
