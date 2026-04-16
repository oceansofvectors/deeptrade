#!/usr/bin/env python
"""Tests for walk-forward hyperparameter tuning safeguards."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from walk_forward import (  # noqa: E402
    _narrow_hp_config,
    _normalize_action_counts,
    _sample_stage_params,
    _score_tuning_trial,
    _should_hard_prune_trial,
    _summarize_trial_scores,
)


class TestWalkForwardTuningScore(unittest.TestCase):
    def test_normalize_action_counts_handles_string_keys(self):
        normalized = _normalize_action_counts({"0": 3, "1": 5, "2": 7})
        self.assertEqual(normalized, {0: 3, 1: 5, 2: 7, 3: 0, 4: 0, 5: 0, 6: 0})

    def test_always_short_policy_is_penalized(self):
        results = {
            "total_return_pct": 16.01,
            "sortino_ratio": 5.98,
            "calmar_ratio": 2.10,
            "max_drawdown": -7.61,
            "trade_count": 1,
            "action_counts": {5: 1377},
        }
        score, diagnostics = _score_tuning_trial(results, validation_bars=1377)

        self.assertLess(score, 0.0)
        self.assertGreater(diagnostics["total_penalty"], diagnostics["base_score"])
        self.assertIn("too_few_trades", diagnostics["collapse_flags"])
        self.assertIn("dominant_action", diagnostics["collapse_flags"])
        self.assertIn("single_action_policy", diagnostics["collapse_flags"])
        self.assertIn("always_short", diagnostics["collapse_flags"])

    def test_balanced_policy_keeps_positive_score(self):
        results = {
            "total_return_pct": 45.0,
            "sortino_ratio": 10.0,
            "calmar_ratio": 5.0,
            "max_drawdown": -3.81,
            "trade_count": 32,
            "action_counts": {0: 363, 3: 320, 6: 694},
        }
        score, diagnostics = _score_tuning_trial(results, validation_bars=1377)

        self.assertGreater(score, 0.0)
        self.assertEqual(diagnostics["collapse_flags"], [])
        self.assertEqual(diagnostics["min_trades_required"], 5)

    def test_baseline_penalty_applies_when_trial_barely_beats_baseline(self):
        final_score, diagnostics = _summarize_trial_scores(
            evaluation_scores=[1.05, 0.95],
            baseline_scores=[{0: 0.9, 1: 1.0, 2: 0.0}, {0: 0.7, 1: 0.9, 2: 0.0}],
            baseline_margin=0.25,
        )

        self.assertLess(final_score, diagnostics["aggregate_score"])
        self.assertGreater(diagnostics["baseline_penalty"], 0.0)

    def test_holdout_stability_affects_aggregate_score(self):
        final_score, diagnostics = _summarize_trial_scores(
            evaluation_scores=[4.0, 3.0, 1.0],
            baseline_scores=[{0: 0.0, 1: 0.5, 2: 0.0}] * 3,
            baseline_margin=0.25,
        )

        self.assertAlmostEqual(diagnostics["primary_score"], 4.0)
        self.assertAlmostEqual(diagnostics["holdout_mean"], 2.0)
        self.assertGreater(diagnostics["holdout_std"], 0.0)
        self.assertAlmostEqual(final_score, diagnostics["aggregate_score"])

    def test_hard_prune_detects_all_flat_zero_trade_policy(self):
        should_prune, reasons = _should_hard_prune_trial(
            {"trade_count": 0, "action_counts": {6: 100}},
            {
                "early_prune_zero_trade": True,
                "early_prune_flat_action_pct": 99.0,
                "early_prune_single_action": True,
            },
        )

        self.assertTrue(should_prune)
        self.assertIn("zero_trade", reasons)
        self.assertIn("all_flat", reasons)
        self.assertIn("single_action", reasons)

    def test_hard_prune_keeps_active_policy(self):
        should_prune, reasons = _should_hard_prune_trial(
            {"trade_count": 12, "action_counts": {0: 35, 3: 28, 6: 37}},
            {
                "early_prune_zero_trade": True,
                "early_prune_flat_action_pct": 99.0,
                "early_prune_single_action": True,
            },
        )

        self.assertFalse(should_prune)
        self.assertEqual(reasons, [])

    def test_excessive_drawdown_is_penalized(self):
        results = {
            "total_return_pct": 25.0,
            "sortino_ratio": 3.0,
            "calmar_ratio": 0.8,
            "max_drawdown": -55.0,
            "trade_count": 40,
            "action_counts": {0: 300, 3: 250, 6: 100},
        }
        score, diagnostics = _score_tuning_trial(results, validation_bars=1000)

        self.assertIn("excessive_drawdown", diagnostics["collapse_flags"])
        self.assertGreater(diagnostics["total_penalty"], 0.0)
        self.assertLess(score, diagnostics["base_score"])

    def test_narrow_hp_config_shrinks_ranges_around_best_params(self):
        base = {
            "learning_rate": {"min": 1e-5, "max": 1e-3, "log": True},
            "n_steps": {"min": 256, "max": 1024, "log": True},
            "ent_coef": {"min": 0.05, "max": 0.25, "log": False},
        }
        narrowed = _narrow_hp_config(
            base,
            {"learning_rate": 1e-4, "n_steps": 512, "ent_coef": 0.12},
            0.35,
        )

        self.assertGreaterEqual(narrowed["learning_rate"]["min"], base["learning_rate"]["min"])
        self.assertLessEqual(narrowed["learning_rate"]["max"], base["learning_rate"]["max"])
        self.assertLess(narrowed["learning_rate"]["max"] - narrowed["learning_rate"]["min"],
                        base["learning_rate"]["max"] - base["learning_rate"]["min"])
        self.assertLess(narrowed["n_steps"]["max"] - narrowed["n_steps"]["min"],
                        base["n_steps"]["max"] - base["n_steps"]["min"])
        self.assertLess(narrowed["ent_coef"]["max"] - narrowed["ent_coef"]["min"],
                        base["ent_coef"]["max"] - base["ent_coef"]["min"])

    def test_sample_stage_params_only_uses_retained_fields(self):
        class DummyTrial:
            def __init__(self):
                self.calls = []

            def suggest_float(self, name, low, high, log=False):
                self.calls.append(("float", name))
                return low

            def suggest_int(self, name, low, high, log=False):
                self.calls.append(("int", name))
                return low

        trial = DummyTrial()
        params = _sample_stage_params(
            trial,
            {
                "learning_rate": {"min": 1e-5, "max": 1e-4, "log": True},
                "n_steps": {"min": 256, "max": 512, "log": True},
                "ent_coef": {"min": 0.05, "max": 0.25, "log": False},
                "reward_turnover_penalty": {"min": 0.01, "max": 0.02},
                "reward_calm_holding_bonus": {"min": 0.001, "max": 0.002},
                "gamma": {"min": 0.99, "max": 0.999},
                "lstm_hidden_size": {"choices": [64, 128]},
            },
        )

        self.assertEqual(
            set(params.keys()),
            {
                "learning_rate",
                "n_steps",
                "ent_coef",
                "reward_turnover_penalty",
                "reward_calm_holding_bonus",
            },
        )
        self.assertEqual(
            [name for _, name in trial.calls],
            [
                "learning_rate",
                "n_steps",
                "ent_coef",
                "reward_turnover_penalty",
                "reward_calm_holding_bonus",
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
