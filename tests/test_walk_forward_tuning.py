#!/usr/bin/env python
"""Tests for walk-forward hyperparameter tuning safeguards."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from walk_forward import (  # noqa: E402
    _narrow_hp_config,
    _normalize_action_counts,
    _resolve_qrdqn_hp_config,
    _sample_qrdqn_params,
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
            "max_drawdown": -65.0,
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

            def suggest_categorical(self, name, choices):
                self.calls.append(("categorical", name))
                return choices[0]

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
                "gamma": {"min": 0.99, "max": 0.999},
                "gae_lambda": {"min": 0.9, "max": 0.99},
                "reward_turnover_penalty": {"min": 0.01, "max": 0.02},
                "reward_loss_multiplier": {"min": 0.25, "max": 1.25},
                "reward_drawdown_penalty": {"min": 0.5, "max": 4.0},
                "reward_drawdown_penalty_threshold": {"min": 0.01, "max": 0.05},
                "reward_flat_time_penalty": {"min": 0.0, "max": 0.002},
                "reward_flat_time_grace_steps": {"min": 15, "max": 60},
                "synthetic_oversample_ratio": {"min": 0.0, "max": 0.2},
            },
        )

        self.assertEqual(
            set(params.keys()),
            {
                "learning_rate",
                "n_steps",
                "ent_coef",
                "gamma",
                "gae_lambda",
                "reward_turnover_penalty",
                "reward_loss_multiplier",
                "reward_drawdown_penalty",
                "reward_drawdown_penalty_threshold",
                "reward_flat_time_penalty",
                "reward_flat_time_grace_steps",
                "synthetic_oversample_ratio",
            },
        )
        self.assertEqual(
            [name for _, name in trial.calls],
            [
                "learning_rate",
                "n_steps",
                "ent_coef",
                "gamma",
                "gae_lambda",
                "reward_turnover_penalty",
                "reward_loss_multiplier",
                "reward_drawdown_penalty",
                "reward_drawdown_penalty_threshold",
                "reward_flat_time_penalty",
                "reward_flat_time_grace_steps",
                "synthetic_oversample_ratio",
            ],
        )

    def test_sample_qrdqn_params_uses_minimal_search_fields(self):
        class DummyTrial:
            def __init__(self):
                self.calls = []

            def suggest_categorical(self, name, choices):
                self.calls.append(("categorical", name))
                return choices[0]

            def suggest_float(self, name, low, high, log=False):
                self.calls.append(("float", name))
                return low

            def suggest_int(self, name, low, high, log=False):
                self.calls.append(("int", name))
                return low

        trial = DummyTrial()
        params = _sample_qrdqn_params(
            trial,
            {
                "learning_rate": {"min": 1e-5, "max": 5e-4, "log": True},
                "batch_size": {"choices": [32, 64, 128]},
                "gamma": {"min": 0.985, "max": 0.999},
                "buffer_size": {"choices": [50000, 100000]},
                "learning_starts": {"choices": [2000, 5000]},
                "train_freq": {"choices": [4, 16]},
                "gradient_steps": {"choices": [1, 4]},
                "target_update_interval": {"choices": [1000, 2000]},
                "exploration_fraction": {"min": 0.05, "max": 0.2},
                "exploration_final_eps": {"min": 0.02, "max": 0.1},
                "reward_turnover_penalty": {"min": 0.002, "max": 0.02},
                "reward_loss_multiplier": {"min": 0.25, "max": 1.25},
                "reward_drawdown_penalty": {"min": 0.5, "max": 4.0},
                "reward_drawdown_penalty_threshold": {"min": 0.01, "max": 0.05},
                "reward_flat_time_penalty": {"min": 0.0, "max": 0.002},
                "reward_flat_time_grace_steps": {"min": 15, "max": 60},
                "synthetic_oversample_ratio": {"min": 0.0, "max": 0.2},
            },
        )

        self.assertEqual(
            params,
            {
                "learning_rate": 1e-5,
                "batch_size": 32,
                "gamma": 0.985,
                "buffer_size": 50000,
                "learning_starts": 2000,
                "train_freq": 4,
                "gradient_steps": 1,
                "target_update_interval": 1000,
                "exploration_fraction": 0.05,
                "exploration_final_eps": 0.02,
                "reward_turnover_penalty": 0.002,
                "reward_loss_multiplier": 0.25,
                "reward_drawdown_penalty": 0.5,
                "reward_drawdown_penalty_threshold": 0.01,
                "reward_flat_time_penalty": 0.0,
                "reward_flat_time_grace_steps": 15,
                "synthetic_oversample_ratio": 0.0,
            },
        )
        self.assertEqual(
            [name for _, name in trial.calls],
            [
                "learning_rate",
                "batch_size",
                "gamma",
                "buffer_size",
                "learning_starts",
                "train_freq",
                "gradient_steps",
                "target_update_interval",
                "exploration_fraction",
                "exploration_final_eps",
                "reward_turnover_penalty",
                "reward_loss_multiplier",
                "reward_drawdown_penalty",
                "reward_drawdown_penalty_threshold",
                "reward_flat_time_penalty",
                "reward_flat_time_grace_steps",
                "synthetic_oversample_ratio",
            ],
        )

    def test_resolve_qrdqn_hp_config_inherits_shared_reward_fields(self):
        resolved = _resolve_qrdqn_hp_config(
            {
                "parameters": {
                    "learning_rate": {"min": 1e-5, "max": 5e-4, "log": True},
                    "reward_turnover_penalty": {"min": 0.002, "max": 0.02},
                    "reward_loss_multiplier": {"min": 0.25, "max": 1.25},
                    "reward_flat_time_grace_steps": {"min": 15, "max": 60},
                    "synthetic_oversample_ratio": {"min": 0.0, "max": 0.2},
                },
                "qrdqn_parameters": {
                    "batch_size": {"choices": [64, 128]},
                },
            }
        )

        self.assertIn("reward_turnover_penalty", resolved)
        self.assertIn("reward_loss_multiplier", resolved)
        self.assertIn("reward_flat_time_grace_steps", resolved)
        self.assertIn("synthetic_oversample_ratio", resolved)
        self.assertEqual(resolved["batch_size"]["choices"], [64, 128])


if __name__ == "__main__":
    unittest.main(verbosity=2)
