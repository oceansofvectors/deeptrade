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


if __name__ == "__main__":
    unittest.main(verbosity=2)
