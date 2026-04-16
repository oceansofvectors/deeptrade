#!/usr/bin/env python
"""Tests for walk_forward._run_tuning_stage orchestration."""

import copy
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import optuna

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config  # noqa: E402
from walk_forward import _run_tuning_stage  # noqa: E402


class _FakeEnv:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeModel:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.learn_calls = []

    def learn(self, total_timesteps, reset_num_timesteps=False):
        self.learn_calls.append((total_timesteps, reset_num_timesteps))
        return self


class _FakeTrial:
    def __init__(self, number=0):
        self.number = number
        self.reports = []
        self.user_attrs = {}

    def report(self, value, step):
        self.reports.append((value, step))

    def should_prune(self):
        return False

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class _FakeStudy:
    def __init__(self, trial):
        self.trial = trial
        self.best_params = {}
        self.best_value = float("-inf")

    def optimize(self, objective, n_trials, n_jobs=None, show_progress_bar=False):
        try:
            value = objective(self.trial)
            self.best_value = value
            self.best_params = {
                "learning_rate": 1e-4,
                "n_steps": 256,
                "ent_coef": 0.1,
                "reward_turnover_penalty": 0.01,
                "reward_calm_holding_bonus": 0.001,
            }
        except optuna.TrialPruned:
            self.best_value = float("-inf")
            self.best_params = {}


class TestRunTuningStage(unittest.TestCase):
    def setUp(self):
        self._config_backup = copy.deepcopy(config)
        config["seed"] = 42
        config["model"]["gamma"] = 0.99
        config["model"]["gae_lambda"] = 0.95
        config["model"]["max_grad_norm"] = 0.5
        config["reward"]["turnover_penalty"] = 0.01
        config["reward"]["calm_holding_bonus"] = 0.001
        config["training"]["random_start_pct"] = 0.2
        config["hyperparameter_tuning"]["pruning_eval_steps"] = 1000
        config["sequence_model"]["enabled"] = False

    def tearDown(self):
        config.clear()
        config.update(self._config_backup)

    def _base_kwargs(self):
        df = pd.DataFrame({"close": [1.0], "close_norm": [0.1]})
        return dict(
            stage_name="stage1",
            hp_config={
                "learning_rate": {"min": 1e-5, "max": 1e-3, "log": True},
                "n_steps": {"min": 128, "max": 512, "log": True},
                "ent_coef": {"min": 0.01, "max": 0.2},
                "reward_turnover_penalty": {"min": 0.0, "max": 0.02},
                "reward_calm_holding_bonus": {"min": 0.0, "max": 0.01},
            },
            train_data=df,
            validation_data=df,
            evaluation_sets=[df, df],
            reference_columns=list(df.columns),
            baseline_scores=[{0: 0.0}, {0: 0.0}],
            baseline_margin=0.25,
            n_trials=1,
            tuning_timesteps=1000,
            window_folder=None,
            use_parallel=False,
            n_jobs=1,
        )

    def test_run_tuning_stage_returns_empty_result_when_no_trials(self):
        result = _run_tuning_stage(**{**self._base_kwargs(), "n_trials": 0})
        self.assertEqual(result["best_params"], {})
        self.assertEqual(result["best_value"], float("-inf"))
        self.assertIsNone(result["study"])

    def test_run_tuning_stage_executes_successful_objective(self):
        trial = _FakeTrial()
        study = _FakeStudy(trial)
        train_env = _FakeEnv()

        with mock.patch("walk_forward.optuna.samplers.TPESampler"), \
             mock.patch("walk_forward.optuna.pruners.MedianPruner"), \
             mock.patch("walk_forward.optuna.create_study", return_value=study), \
             mock.patch("walk_forward._sample_stage_params", return_value={
                 "learning_rate": 1e-4,
                 "n_steps": 256,
                 "ent_coef": 0.1,
                 "reward_turnover_penalty": 0.01,
                 "reward_calm_holding_bonus": 0.001,
             }), \
             mock.patch("walk_forward._build_tuning_env", return_value=(train_env, 1)), \
             mock.patch("walk_forward.get_device", return_value="cpu"), \
             mock.patch("walk_forward.PPO", _FakeModel), \
             mock.patch("walk_forward.evaluate_agent", side_effect=[
                 {"total_return_pct": 1.0, "sortino_ratio": 2.0, "calmar_ratio": 1.0, "max_drawdown": -5.0, "trade_count": 12, "action_counts": {0: 5, 3: 4, 6: 3}},
                 {"total_return_pct": 2.0, "sortino_ratio": 3.0, "calmar_ratio": 1.5, "max_drawdown": -4.0, "trade_count": 14, "action_counts": {0: 6, 3: 5, 6: 3}},
                 {"total_return_pct": 1.5, "sortino_ratio": 2.5, "calmar_ratio": 1.2, "max_drawdown": -4.5, "trade_count": 13, "action_counts": {0: 5, 3: 5, 6: 3}},
             ]), \
             mock.patch("walk_forward._score_tuning_trial", side_effect=[
                 (0.5, {}),
                 (1.0, {"base_score": 1.0, "total_penalty": 0.0, "trade_count": 14, "min_trades_required": 5, "dominant_action_pct": 45.0, "flat_action_pct": 21.4, "collapse_flags": [], "action_counts": {0: 6, 3: 5, 6: 3}}),
                 (0.8, {"base_score": 0.8, "total_penalty": 0.0, "trade_count": 13, "min_trades_required": 5, "dominant_action_pct": 40.0, "flat_action_pct": 23.0, "collapse_flags": [], "action_counts": {0: 5, 3: 5, 6: 3}}),
             ]), \
             mock.patch("walk_forward._summarize_trial_scores", return_value=(1.25, {
                 "primary_score": 1.0,
                 "holdout_mean": 0.8,
                 "holdout_std": 0.0,
                 "aggregate_score": 1.25,
                 "best_baseline_score": 0.1,
                 "baseline_gap": 1.15,
                 "baseline_penalty": 0.0,
             })), \
             mock.patch("walk_forward._should_hard_prune_trial", return_value=(False, [])):
            result = _run_tuning_stage(**self._base_kwargs())

        self.assertEqual(result["best_params"]["n_steps"], 256)
        self.assertEqual(result["best_value"], 1.25)
        self.assertIs(result["study"], study)
        self.assertEqual(trial.reports, [(0.5, 1)])
        self.assertIn("tuning_diagnostics", trial.user_attrs)
        self.assertTrue(train_env.closed)

    def test_run_tuning_stage_hard_prunes_trial(self):
        trial = _FakeTrial()
        study = _FakeStudy(trial)
        train_env = _FakeEnv()

        with mock.patch("walk_forward.optuna.samplers.TPESampler"), \
             mock.patch("walk_forward.optuna.pruners.MedianPruner"), \
             mock.patch("walk_forward.optuna.create_study", return_value=study), \
             mock.patch("walk_forward._sample_stage_params", return_value={
                 "learning_rate": 1e-4,
                 "n_steps": 256,
                 "ent_coef": 0.1,
             }), \
             mock.patch("walk_forward._build_tuning_env", return_value=(train_env, 1)), \
             mock.patch("walk_forward.get_device", return_value="cpu"), \
             mock.patch("walk_forward.PPO", _FakeModel), \
             mock.patch("walk_forward.evaluate_agent", return_value={"total_return_pct": -10.0, "trade_count": 0, "action_counts": {6: 100}}), \
             mock.patch("walk_forward._score_tuning_trial", return_value=(-5.0, {})), \
             mock.patch("walk_forward._should_hard_prune_trial", return_value=(True, ["zero_trade"])):
            result = _run_tuning_stage(**{**self._base_kwargs(), "evaluation_sets": [pd.DataFrame({"close": [1.0], "close_norm": [0.1]})], "baseline_scores": [{0: 0.0}]})

        self.assertEqual(result["best_params"], {})
        self.assertEqual(result["best_value"], float("-inf"))
        self.assertEqual(trial.user_attrs["hard_prune_reasons"], ["zero_trade"])
        self.assertTrue(train_env.closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
