#!/usr/bin/env python
"""Tests for the staged walk-forward hyperparameter tuning pipeline."""

import copy
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import walk_forward  # noqa: E402
from environment import TradingEnv  # noqa: E402
from utils.synthetic_bears import augment_with_synthetic_bears  # noqa: E402


def make_price_frame(rows=20, freq="D"):
    index = pd.date_range("2024-01-01", periods=rows, freq=freq)
    values = np.linspace(100.0, 120.0, rows)
    return pd.DataFrame(
        {
            "open": values,
            "high": values + 1.0,
            "low": values - 1.0,
            "close": values,
            "volume": np.linspace(1000, 2000, rows),
            "close_norm": np.linspace(0.1, 0.9, rows),
        },
        index=index,
    )


class DummyEnv:
    last_reward_overrides = None

    def __init__(self, data, **kwargs):
        DummyEnv.last_reward_overrides = kwargs.get("reward_overrides")
        self.data = data

    def close(self):
        return None


class DummyModel:
    def __init__(self, *args, **kwargs):
        self.learn_calls = 0

    def learn(self, total_timesteps, reset_num_timesteps=False, progress_bar=False):
        self.learn_calls += 1
        return self


class FakeTrial:
    def __init__(self, number, prune=False):
        self.number = number
        self._prune = prune
        self.report_calls = 0

    def suggest_float(self, name, low, high, log=False):
        return float(low)

    def suggest_int(self, name, low, high, log=False):
        return int(low)

    def suggest_categorical(self, name, choices):
        return choices[0]

    def report(self, value, step):
        self.report_calls += 1

    def should_prune(self):
        return self._prune and self.report_calls >= 1


class FakeStudy:
    def __init__(self, trials):
        self._trials = trials
        self.best_params = None
        self.best_value = None

    def optimize(self, objective, n_trials, n_jobs, show_progress_bar):
        best_value = None
        best_params = None
        for trial in self._trials[:n_trials]:
            try:
                value = objective(trial)
            except walk_forward.optuna.TrialPruned:
                continue
            params = {}
            if best_value is None or value > best_value:
                best_value = value
                best_params = params
        self.best_value = best_value if best_value is not None else -1e12
        self.best_params = best_params or {}


class HyperparameterTuningPipelineTests(unittest.TestCase):
    def setUp(self):
        self.original_config = copy.deepcopy(walk_forward.config)
        walk_forward.config.setdefault("data", {})["market_hours_only"] = False
        walk_forward.config.setdefault("hyperparameter_tuning", {}).setdefault("parallel_processing", {})
        walk_forward.config["hyperparameter_tuning"]["parallel_processing"]["enabled"] = False
        walk_forward.config["sequence_model"]["enabled"] = False

    def tearDown(self):
        walk_forward.config.clear()
        walk_forward.config.update(self.original_config)

    def test_select_tuning_datasets_honors_dates_and_falls_back(self):
        frame = make_price_frame(rows=60)
        walk_forward.config["hyperparameter_tuning"]["scope"] = {
            "mode": "dedicated_global",
            "global_period": {"start_date": "2024-01-03", "end_date": "2024-02-20"},
            "train_ratio": 0.5,
            "validation_ratio": 0.5,
        }

        train_data, val_data, metadata = walk_forward.select_tuning_datasets(frame, [])
        self.assertEqual(str(train_data.index[0].date()), "2024-01-03")
        self.assertEqual(str(val_data.index[-1].date()), "2024-02-20")
        self.assertEqual(metadata["mode"], "dedicated_global")

        walk_forward.config["hyperparameter_tuning"]["scope"]["global_period"] = {"start_date": None, "end_date": None}
        train_data, val_data, metadata = walk_forward.select_tuning_datasets(frame, [])
        self.assertEqual(str(train_data.index[0].date()), "2024-01-01")
        self.assertEqual(str(val_data.index[-1].date()), "2024-02-29")
        self.assertEqual(metadata["train_rows"] + metadata["validation_rows"], len(frame))

    def test_trading_env_reward_overrides_take_precedence(self):
        frame = make_price_frame(rows=8)
        walk_forward.config["reward"]["loss_multiplier"] = 0.1
        walk_forward.config["reward"]["turnover_penalty"] = 0.2
        walk_forward.config["reward"]["calm_holding_bonus"] = 0.3

        env = TradingEnv(
            frame,
            reward_overrides={
                "loss_multiplier": 0.8,
                "turnover_penalty": 0.01,
                "calm_holding_bonus": 0.004,
            },
        )

        self.assertAlmostEqual(env._reward_loss_multiplier, 0.8)
        self.assertAlmostEqual(env._reward_turnover_pen, 0.01)
        self.assertAlmostEqual(env._reward_calm_bonus, 0.004)

    def test_augmentation_seed_is_deterministic(self):
        frame = make_price_frame(rows=30)
        first = augment_with_synthetic_bears(frame, oversample_ratio=0.3, segment_length_pct=0.2, seed=7)
        second = augment_with_synthetic_bears(frame, oversample_ratio=0.3, segment_length_pct=0.2, seed=7)
        third = augment_with_synthetic_bears(frame, oversample_ratio=0.3, segment_length_pct=0.2, seed=11)

        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(first.equals(third))

    def test_hyperparameter_tuning_uses_exact_metric_and_does_not_mutate_config(self):
        original_loss_multiplier = walk_forward.config["reward"]["loss_multiplier"]

        with mock.patch.object(walk_forward, "TradingEnv", DummyEnv), \
             mock.patch.object(walk_forward, "PPO", DummyModel), \
             mock.patch.object(walk_forward, "prepare_model_datasets", side_effect=lambda **kwargs: (kwargs["train_data"], kwargs["validation_data"], None)), \
             mock.patch.object(walk_forward, "evaluate_agent", return_value={
                 "total_return_pct": 9.0,
                 "sortino_ratio": 1.23,
                 "calmar_ratio": 2.34,
                 "max_drawdown": -4.5,
                 "trade_count": 25,
                 "action_counts": {0: 8, 1: 8, 2: 9},
             }):
            sortino_result = walk_forward.hyperparameter_tuning(
                train_data=make_price_frame(rows=40),
                validation_data=make_price_frame(rows=30),
                n_trials=1,
                eval_metric="sortino",
                stage_name="reward_stage_test",
                parameter_names=["reward_loss_multiplier"],
                tuning_timesteps=5000,
                base_seed=101,
            )
            self.assertAlmostEqual(sortino_result["best_value"], 1.23)
            self.assertEqual(walk_forward.config["reward"]["loss_multiplier"], original_loss_multiplier)
            self.assertIn("loss_multiplier", DummyEnv.last_reward_overrides)

            calmar_result = walk_forward.hyperparameter_tuning(
                train_data=make_price_frame(rows=40),
                validation_data=make_price_frame(rows=30),
                n_trials=1,
                eval_metric="calmar",
                stage_name="ppo_stage_test",
                parameter_names=["learning_rate"],
                tuning_timesteps=5000,
                base_seed=202,
            )
            self.assertAlmostEqual(calmar_result["best_value"], 2.34)

    def test_failed_and_pruned_trials_are_recorded(self):
        frame = make_price_frame(rows=12)
        fake_trials = [FakeTrial(0, prune=True), FakeTrial(1, prune=False), FakeTrial(2, prune=False)]
        original_prepare = walk_forward.prepare_model_datasets
        original_optuna = walk_forward.optuna

        def prepare_side_effect(**kwargs):
            if kwargs.get("augmentation_seed") == 1002:
                raise ValueError("boom")
            return original_prepare(**kwargs)

        with mock.patch.object(walk_forward, "optuna") as fake_optuna, \
             mock.patch.object(walk_forward, "TradingEnv", DummyEnv), \
             mock.patch.object(walk_forward, "PPO", DummyModel), \
             mock.patch.object(walk_forward, "prepare_model_datasets", side_effect=prepare_side_effect), \
             mock.patch.object(walk_forward, "evaluate_agent", return_value={
                 "total_return_pct": 5.0,
                 "sortino_ratio": 0.5,
                 "calmar_ratio": 0.25,
                 "max_drawdown": -1.0,
                 "trade_count": 25,
                 "action_counts": {0: 10, 1: 8, 2: 7},
             }):
            fake_optuna.TrialPruned = original_optuna.TrialPruned
            fake_optuna.samplers = original_optuna.samplers
            fake_optuna.pruners = original_optuna.pruners
            fake_optuna.create_study.return_value = FakeStudy(fake_trials)

            result = walk_forward.hyperparameter_tuning(
                train_data=frame,
                validation_data=frame.iloc[:6],
                n_trials=3,
                eval_metric="sortino",
                stage_name="record_status_test",
                parameter_names=["learning_rate"],
                tuning_timesteps=5000,
                base_seed=1000,
            )

        statuses = {record["status"] for record in result["trial_records"]}
        self.assertIn("pruned", statuses)
        self.assertIn("failed", statuses)
        self.assertIn("completed", statuses)

    def test_tuning_rejects_too_few_trades(self):
        with mock.patch.object(walk_forward, "TradingEnv", DummyEnv), \
             mock.patch.object(walk_forward, "PPO", DummyModel), \
             mock.patch.object(walk_forward, "prepare_model_datasets", side_effect=lambda **kwargs: (kwargs["train_data"], kwargs["validation_data"], None)), \
             mock.patch.object(walk_forward, "evaluate_agent", return_value={
                 "total_return_pct": 12.0,
                 "sortino_ratio": 3.0,
                 "calmar_ratio": 1.5,
                 "max_drawdown": -4.0,
                 "trade_count": 1,
                 "action_counts": {0: 1, 1: 0, 2: 29},
             }):
            result = walk_forward.hyperparameter_tuning(
                train_data=make_price_frame(rows=40),
                validation_data=make_price_frame(rows=30),
                n_trials=1,
                eval_metric="sortino",
                stage_name="reject_low_trades_test",
                parameter_names=["learning_rate"],
                tuning_timesteps=5000,
                base_seed=303,
            )

        self.assertEqual(result["best_value"], -1e12)
        self.assertEqual(result["trial_records"][0]["status"], "rejected")
        self.assertIn("too_few_trades", result["trial_records"][0]["reason"])

    def test_tuning_rejects_non_positive_return_for_sortino(self):
        with mock.patch.object(walk_forward, "TradingEnv", DummyEnv), \
             mock.patch.object(walk_forward, "PPO", DummyModel), \
             mock.patch.object(walk_forward, "prepare_model_datasets", side_effect=lambda **kwargs: (kwargs["train_data"], kwargs["validation_data"], None)), \
             mock.patch.object(walk_forward, "evaluate_agent", return_value={
                 "total_return_pct": -25.0,
                 "sortino_ratio": 6.5,
                 "calmar_ratio": -0.8,
                 "max_drawdown": -30.0,
                 "trade_count": 50,
                 "action_counts": {0: 20, 1: 15, 2: 15},
             }):
            result = walk_forward.hyperparameter_tuning(
                train_data=make_price_frame(rows=40),
                validation_data=make_price_frame(rows=30),
                n_trials=1,
                eval_metric="sortino",
                stage_name="reject_negative_return_test",
                parameter_names=["learning_rate"],
                tuning_timesteps=5000,
                base_seed=404,
            )

        self.assertEqual(result["best_value"], -1e12)
        self.assertEqual(result["trial_records"][0]["status"], "rejected")
        self.assertIn("non_positive_return", result["trial_records"][0]["reason"])

    def test_save_best_hyperparameters_updates_all_sections(self):
        config_data = {
            "model": {},
            "sequence_model": {},
            "reward": {},
            "augmentation": {"synthetic_bears": {}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            with open(config_path, "w") as handle:
                yaml.safe_dump(config_data, handle)

            walk_forward.save_best_hyperparameters_to_config(
                {
                    "learning_rate": 0.001,
                    "batch_size": 64,
                    "gamma": 0.991,
                    "gae_lambda": 0.95,
                    "ent_coef": 0.11,
                    "n_steps": 256,
                    "lstm_hidden_size": 128,
                    "n_lstm_layers": 2,
                    "reward_loss_multiplier": 0.8,
                    "reward_turnover_penalty": 0.02,
                    "reward_calm_holding_bonus": 0.003,
                    "synthetic_oversample_ratio": 0.25,
                },
                config_path=config_path,
            )

            with open(config_path, "r") as handle:
                saved = yaml.safe_load(handle)

            self.assertEqual(saved["model"]["batch_size"], 64.0)
            self.assertEqual(saved["sequence_model"]["lstm_hidden_size"], 128)
            self.assertEqual(saved["reward"]["loss_multiplier"], 0.8)
            self.assertEqual(saved["augmentation"]["synthetic_bears"]["oversample_ratio"], 0.25)

    def test_run_staged_tuning_uses_stage_specific_parameters_and_writes_artifacts(self):
        frame = make_price_frame(rows=40)
        walk_forward.config["hyperparameter_tuning"]["scope"] = {
            "mode": "dedicated_global",
            "global_period": {"start_date": None, "end_date": None},
            "train_ratio": 0.5,
            "validation_ratio": 0.5,
        }
        walk_forward.config["hyperparameter_tuning"]["stages"] = {
            "reward_and_augmentation": {"enabled": True, "n_trials": 1, "tuning_timesteps": 5000},
            "ppo_and_sequence": {"enabled": True, "n_trials": 1, "tuning_timesteps": 5000},
        }

        captured_parameter_names = []

        def fake_hyperparameter_tuning(**kwargs):
            captured_parameter_names.append((kwargs["stage_name"], kwargs["parameter_names"]))
            if kwargs["stage_name"] == "reward_and_augmentation":
                return {
                    "best_params": {"reward_loss_multiplier": 0.7, "synthetic_oversample_ratio": 0.22},
                    "best_value": 1.0,
                    "study": object(),
                    "trial_records": [{"stage": kwargs["stage_name"], "trial_number": 0, "params": {}, "status": "completed"}],
                }
            return {
                "best_params": {"learning_rate": 0.0002, "gamma": 0.995},
                "best_value": 2.0,
                "study": object(),
                "trial_records": [{"stage": kwargs["stage_name"], "trial_number": 0, "params": {}, "status": "completed"}],
            }

        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch.object(walk_forward, "hyperparameter_tuning", side_effect=fake_hyperparameter_tuning), \
             mock.patch.object(walk_forward, "get_lstm_feature_params", return_value={"lookback": 20, "hidden_size": 16, "num_layers": 1, "output_size": 4, "pretrain_lr": 0.001, "pretrain_epochs": 2, "pretrain_batch_size": 4, "pretrain_patience": 1}), \
             mock.patch.object(walk_forward, "apply_lstm_features_to_datasets", side_effect=lambda **kwargs: (kwargs["train_data"], kwargs["validation_data"], None, kwargs["lstm_params"])), \
             mock.patch.object(walk_forward, "save_best_hyperparameters_to_config"), \
             mock.patch.object(walk_forward, "apply_best_hyperparameters_to_runtime_config"):
            os.makedirs(os.path.join(tmpdir, "reports"), exist_ok=True)
            result = walk_forward.run_staged_hyperparameter_tuning(
                data=frame,
                window_data_list=[{"train_data": frame.iloc[:8], "validation_data": frame.iloc[8:12]}],
                session_folder=tmpdir,
                eval_metric="sortino",
                fallback_trials=1,
            )

            self.assertEqual(
                captured_parameter_names,
                [
                    ("reward_and_augmentation", ["reward_loss_multiplier", "reward_turnover_penalty", "reward_calm_holding_bonus", "synthetic_oversample_ratio"]),
                    ("ppo_and_sequence", ["learning_rate", "n_steps", "batch_size", "ent_coef", "gamma", "gae_lambda", "lstm_hidden_size", "n_lstm_layers"]),
                ],
            )
            self.assertIn("reward_loss_multiplier", result["best_hyperparameters"])
            self.assertIn("learning_rate", result["best_hyperparameters"])
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "reports", "global_tuning_stage1_best_params.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "reports", "global_tuning_stage2_best_params.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "reports", "global_tuning_trials.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "reports", "global_tuning_trials.csv")))

    def test_walk_forward_runs_tuning_once_and_reuses_results(self):
        frame = make_price_frame(rows=24)
        captured_calls = []

        def fake_process_single_window(*args, **kwargs):
            captured_calls.append({"best_hyperparameters": args[14], "lstm_feature_params": args[15]})
            return {
                "window": args[0],
                "return": 1.0,
                "portfolio_value": 101000.0,
                "trade_count": 2,
                "sortino_ratio": 0.5,
            }

        with mock.patch.object(walk_forward, "run_staged_hyperparameter_tuning", return_value={
            "scope": "dedicated_global",
            "tuning_period": {"actual_start": "2024-01-01", "actual_end": "2024-01-10"},
            "stage_winners": {"reward_and_augmentation": {"reward_loss_multiplier": 0.6}, "ppo_and_sequence": {"learning_rate": 0.0003}},
            "best_hyperparameters": {"reward_loss_multiplier": 0.6, "learning_rate": 0.0003},
            "lstm_feature_params": {"hidden_size": 16},
        }) as tuning_mock, \
             mock.patch.object(walk_forward, "process_single_window", side_effect=fake_process_single_window), \
             mock.patch.object(walk_forward, "plot_walk_forward_results"), \
             mock.patch.object(walk_forward, "export_consolidated_trade_history"):
            walk_forward.config["walk_forward"]["parallel_processing"]["enabled"] = False
            with tempfile.TemporaryDirectory() as tmpdir:
                cwd = os.getcwd()
                os.chdir(tmpdir)
                try:
                    results = walk_forward.walk_forward_testing(
                        data=frame,
                        window_size=6,
                        step_size=3,
                        train_ratio=0.5,
                        validation_ratio=0.25,
                        initial_timesteps=10,
                        additional_timesteps=5,
                        max_iterations=1,
                        n_stagnant_loops=1,
                        improvement_threshold=0.01,
                        run_hyperparameter_tuning=True,
                        tuning_trials=1,
                        max_windows=2,
                    )
                finally:
                    os.chdir(cwd)

            self.assertEqual(tuning_mock.call_count, 1)
            self.assertEqual(len(captured_calls), 2)
            for call in captured_calls:
                self.assertEqual(call["best_hyperparameters"], {"reward_loss_multiplier": 0.6, "learning_rate": 0.0003})
                self.assertEqual(call["lstm_feature_params"], {"hidden_size": 16})
            self.assertEqual(results["tuning_scope"], "dedicated_global")


if __name__ == "__main__":
    unittest.main(verbosity=2)
