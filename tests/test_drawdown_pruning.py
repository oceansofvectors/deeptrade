import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tuning_prune import should_hard_prune_trial, tuning_prune_diagnostics


class TestDrawdownPruning(unittest.TestCase):
    def test_tuning_prune_diagnostics_reports_absolute_drawdown(self):
        results = {"action_counts": {6: 10}, "trade_count": 1, "max_drawdown": -41.5}
        diag = tuning_prune_diagnostics(results)
        self.assertEqual(diag["max_drawdown_pct"], 41.5)

    def test_hard_prunes_trial_above_max_drawdown_threshold(self):
        results = {"action_counts": {0: 5, 6: 1}, "trade_count": 3, "max_drawdown": -41.0}
        should_prune, reasons = should_hard_prune_trial(
            results,
            {"early_prune_max_drawdown_pct": 40.0},
        )
        self.assertTrue(should_prune)
        self.assertIn("excessive_drawdown", reasons)

    def test_does_not_prune_trial_at_or_below_threshold(self):
        results = {"action_counts": {0: 3, 5: 3}, "trade_count": 4, "max_drawdown": -39.9}
        should_prune, reasons = should_hard_prune_trial(
            results,
            {"early_prune_max_drawdown_pct": 40.0},
        )
        self.assertFalse(should_prune)
        self.assertNotIn("excessive_drawdown", reasons)

    def test_zero_trade_prune_waits_until_min_step_and_collapsed_actions(self):
        results = {"action_counts": {0: 40, 1: 30, 6: 30}, "trade_count": 0, "max_drawdown": -5.0}
        should_prune, reasons = should_hard_prune_trial(
            results,
            {"early_prune_zero_trade": True, "early_prune_zero_trade_min_step": 3},
            prune_step=1,
        )
        self.assertFalse(should_prune)
        self.assertNotIn("zero_trade", reasons)

        should_prune_late, reasons_late = should_hard_prune_trial(
            results,
            {"early_prune_zero_trade": True, "early_prune_zero_trade_min_step": 3},
            prune_step=3,
        )
        self.assertFalse(should_prune_late)
        self.assertNotIn("zero_trade", reasons_late)

        collapsed_results = {"action_counts": {6: 100}, "trade_count": 0, "max_drawdown": -1.0}
        should_prune_collapsed, reasons_collapsed = should_hard_prune_trial(
            collapsed_results,
            {"early_prune_zero_trade": True, "early_prune_zero_trade_min_step": 3, "early_prune_flat_action_pct": 99.0},
            prune_step=3,
        )
        self.assertTrue(should_prune_collapsed)
        self.assertIn("zero_trade", reasons_collapsed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
