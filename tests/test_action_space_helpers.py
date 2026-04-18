#!/usr/bin/env python
"""Tests for shared action-space helper semantics."""

import copy
import importlib
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import action_space as action_space_module  # noqa: E402
from config import config  # noqa: E402
from action_space import (  # noqa: E402
    ACTION_COUNT,
    ACTION_MODE,
    FLAT_ACTION,
    LONG_ACTIONS,
    RISK_BUCKETS,
    SHORT_ACTIONS,
    TARGET_CONTRACTS,
    action_direction,
    action_from_target_contracts,
    action_label,
    action_mode,
    action_value,
    action_size_contracts,
    target_contracts_for_action,
)


class TestActionSpaceHelpers(unittest.TestCase):
    def setUp(self):
        self._config_backup = copy.deepcopy(config)

    def tearDown(self):
        config.clear()
        config.update(self._config_backup)
        importlib.reload(action_space_module)

    def test_default_atr_risk_mode_falls_back_to_fixed_targets_without_context(self):
        expected = [1, 2, 3, -1, -2, -3, 0]
        self.assertEqual(ACTION_MODE, "atr_risk")
        self.assertEqual([target_contracts_for_action(i) for i in range(ACTION_COUNT)], expected)

    def test_atr_risk_actions_map_to_contracts_with_context(self):
        contracts = [
            target_contracts_for_action(i, net_worth=100000.0, atr=1000.0)
            for i in range(ACTION_COUNT)
        ]
        self.assertEqual(contracts, [2, 5, 7, -2, -5, -7, 0])

    def test_labels_and_directions_match_default_risk_bucket_semantics(self):
        self.assertEqual(action_label(0), "LONG_0.5R")
        self.assertEqual(action_label(5), "SHORT_1.5R")
        self.assertEqual(action_label(6), "FLAT")
        self.assertEqual(action_direction(1), 1)
        self.assertEqual(action_direction(4), -1)
        self.assertEqual(action_direction(6), 0)
        self.assertAlmostEqual(action_value(0), 0.5)
        self.assertAlmostEqual(action_value(4), -1.0)

    def test_size_and_reverse_lookup(self):
        self.assertEqual(action_size_contracts(2), 1.5)
        self.assertEqual(action_size_contracts(4), 1.0)
        self.assertEqual(action_from_target_contracts(3), 2)
        self.assertEqual(action_from_target_contracts(-2), 4)
        self.assertEqual(action_from_target_contracts(0), 6)
        self.assertIsNone(action_from_target_contracts(7))

    def test_unknown_label_and_exact_reverse_lookup(self):
        self.assertEqual(action_label(99), "UNKNOWN_99")
        self.assertEqual(action_from_target_contracts(1), 0)

    def test_module_constants_capture_long_short_and_flat_groups(self):
        self.assertEqual(TARGET_CONTRACTS, (1, 2, 3, -1, -2, -3, 0))
        self.assertEqual(RISK_BUCKETS, (0.5, 1.0, 1.5, -0.5, -1.0, -1.5, 0.0))
        self.assertEqual(FLAT_ACTION, 6)
        self.assertEqual(LONG_ACTIONS, (0, 1, 2))
        self.assertEqual(SHORT_ACTIONS, (3, 4, 5))

    def test_invalid_configured_contract_grid_falls_back_to_defaults(self):
        config["action_space"] = {"target_contracts": [1, "bad", 0]}

        reloaded = importlib.reload(action_space_module)

        self.assertEqual(reloaded.TARGET_CONTRACTS, reloaded.DEFAULT_TARGET_CONTRACTS)
        self.assertEqual(reloaded.ACTION_COUNT, 7)
        self.assertEqual(reloaded.FLAT_ACTION, 6)

    def test_custom_fixed_contract_mode_rebuilds_labels_and_groups(self):
        config["action_space"] = {"mode": "fixed_contracts", "target_contracts": [4, 2, 1, -1, -2, -4, 0]}

        reloaded = importlib.reload(action_space_module)

        self.assertEqual(reloaded.action_mode(), "fixed_contracts")
        self.assertEqual(reloaded.TARGET_CONTRACTS, (4, 2, 1, -1, -2, -4, 0))
        self.assertEqual(reloaded.action_label(0), "LONG_4")
        self.assertEqual(reloaded.action_label(5), "SHORT_4")
        self.assertEqual(reloaded.LONG_ACTIONS, (0, 1, 2))
        self.assertEqual(reloaded.SHORT_ACTIONS, (3, 4, 5))
        self.assertEqual(reloaded.action_from_target_contracts(-4), 5)

    def test_invalid_action_mode_falls_back_to_atr_risk(self):
        config["action_space"] = {"mode": "weird", "risk_buckets": [1, 2, 3, -1, -2, -3, 0]}

        reloaded = importlib.reload(action_space_module)

        self.assertEqual(reloaded.action_mode(), "atr_risk")
        self.assertEqual(reloaded.RISK_BUCKETS, (1.0, 2.0, 3.0, -1.0, -2.0, -3.0, 0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
