import os
import sys
import unittest
import copy
import importlib
from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytz

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import action_space as action_space_module
from environment import TradingEnv
from trade import RiskManager
from config import config


class TestAllocationActionSpace(unittest.TestCase):
    def setUp(self):
        self._risk_cfg = copy.deepcopy(config.get("risk_management", {}))
        self._exec_cfg = copy.deepcopy(config.get("execution_costs", {}))
        self._action_cfg = copy.deepcopy(config.get("action_space", {}))
        config.setdefault("action_space", {})["mode"] = "fixed_contracts"
        importlib.reload(action_space_module)

    def tearDown(self):
        config["risk_management"] = self._risk_cfg
        config["execution_costs"] = self._exec_cfg
        config["action_space"] = self._action_cfg
        importlib.reload(action_space_module)

    def _make_env(self, closes, initial_balance=100000.0):
        rows = len(closes)
        df = pd.DataFrame(
            {
                "close_norm": [0.5] * rows,
                "close": closes,
                "open": closes,
                "high": closes,
                "low": closes,
                "ATR": [1.0] * rows,
                "volume": [1.0] * rows,
            }
        )
        return TradingEnv(
            df,
            initial_balance=initial_balance,
            transaction_cost=0.0,
            position_size=1,
        )

    def test_all_actions_map_to_expected_contract_targets(self):
        expected_contracts = {
            0: 1,
            1: 2,
            2: 3,
            3: -1,
            4: -2,
            5: -3,
            6: 0,
        }
        for action, expected in expected_contracts.items():
            env = self._make_env([100.0, 100.0, 100.0])
            env.reset()
            _, _, _, _, info = env.step(action)
            self.assertEqual(
                info["current_contracts"], expected,
                f"action={action} should target {expected} contracts",
            )

    def test_same_side_scale_in_recomputes_weighted_average_entry(self):
        env = self._make_env([100.0, 110.0, 120.0, 120.0])
        env.reset()

        env.step(0)  # long 1 contract @ 100
        _, _, _, _, info = env.step(2)  # long 3 contracts @ 110

        expected_avg = ((Decimal("100") * Decimal("1")) + (Decimal("110") * Decimal("2"))) / Decimal("3")
        self.assertEqual(info["current_contracts"], 3)
        self.assertAlmostEqual(info["avg_entry_price"], float(expected_avg), places=6)
        self.assertGreater(info["signed_exposure"], 0.0)

    def test_same_side_scale_down_preserves_average_entry(self):
        env = self._make_env([100.0, 110.0, 120.0, 120.0])
        env.reset()

        env.step(2)  # long 3 contracts @ 100
        _, _, _, _, info = env.step(0)  # long 1 contract @ 110

        self.assertEqual(info["current_contracts"], 1)
        self.assertEqual(info["avg_entry_price"], 100.0)
        self.assertGreater(info["net_worth"], 100000.0)

    def test_reversal_closes_and_reopens_with_new_entry(self):
        env = self._make_env([100.0, 110.0, 120.0, 120.0])
        env.reset()

        env.step(1)  # long 2 contracts @ 100
        _, _, _, _, info = env.step(4)  # short 2 contracts @ 110

        self.assertEqual(info["old_contracts"], 2)
        self.assertEqual(info["current_contracts"], -2)
        self.assertEqual(info["avg_entry_price"], 110.0)
        self.assertLess(info["signed_exposure"], 0.0)

    def test_target_contract_actions_do_not_round_down_to_zero(self):
        env = self._make_env([50000.0, 50000.0, 50000.0], initial_balance=1000.0)
        env.reset()

        _, _, _, _, info = env.step(0)

        self.assertEqual(info["current_contracts"], 1)
        self.assertNotEqual(info["signed_exposure"], 0.0)
        self.assertEqual(info["position"], 1)

    def test_rebalanced_position_drives_next_bar_pnl(self):
        env = self._make_env([100.0, 110.0, 110.0])
        env.reset()

        _, _, _, _, info = env.step(2)  # long 3 contracts before 100->110 move

        self.assertEqual(info["current_contracts"], 3)
        self.assertEqual(info["net_worth"], 100002.25)

    def test_fixed_take_profit_percentage_exits_without_double_counting_pnl(self):
        config["risk_management"]["stop_loss"] = {"enabled": False, "mode": "percentage", "percentage": 17}
        config["risk_management"]["take_profit"] = {"enabled": True, "mode": "percentage", "percentage": 5}
        config["execution_costs"]["half_spread_points"] = 0.0
        config["execution_costs"]["base_slippage_points"] = 0.0

        env = self._make_env([100.0, 110.0, 110.0])
        env.reset()

        _, _, _, _, info = env.step(2)

        self.assertTrue(info["fixed_tp_triggered"])
        self.assertEqual(info["position"], 0)
        self.assertEqual(info["current_contracts"], 0)
        self.assertEqual(info["net_worth"], 100001.5)


class TestRiskManagerAllocationRebalancing(unittest.TestCase):
    def setUp(self):
        self.eastern = pytz.timezone("US/Eastern")
        self.date = pd.Timestamp(datetime(2024, 1, 2, 10, 0, 0), tz=self.eastern)

    def test_rebalance_scale_in_updates_weighted_average(self):
        rm = RiskManager(initial_balance=100000.0, transaction_cost=0.0)
        changed = rm.rebalance_position(1, 100.0, self.date)
        self.assertTrue(changed)
        changed = rm.rebalance_position(3, 110.0, self.date)
        self.assertTrue(changed)

        expected_avg = ((Decimal("100") * Decimal("1")) + (Decimal("110") * Decimal("2"))) / Decimal("3")
        self.assertEqual(rm.position, 1)
        self.assertEqual(rm.current_contracts, 3)
        self.assertAlmostEqual(float(rm.entry_price), float(expected_avg), places=6)

    def test_rebalance_scale_down_realizes_partial_pnl(self):
        rm = RiskManager(initial_balance=100000.0, transaction_cost=0.0)
        rm.rebalance_position(3, 100.0, self.date)
        rm.rebalance_position(1, 110.0, self.date)

        self.assertEqual(rm.position, 1)
        self.assertEqual(rm.current_contracts, 1)
        self.assertEqual(rm.entry_price, Decimal("100"))
        self.assertEqual(rm.net_worth, Decimal("100002.0"))

    def test_rebalance_reversal_sets_new_entry_and_side(self):
        rm = RiskManager(initial_balance=100000.0, transaction_cost=0.0)
        rm.rebalance_position(2, 100.0, self.date)
        rm.rebalance_position(-2, 110.0, self.date)

        self.assertEqual(rm.position, -1)
        self.assertEqual(rm.current_contracts, 2)
        self.assertEqual(rm.entry_price, Decimal("110"))
        self.assertEqual(rm.net_worth, Decimal("100002.0"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
