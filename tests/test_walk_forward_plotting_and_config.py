#!/usr/bin/env python
"""Tests for walk_forward plotting and config persistence helpers."""

import copy
import os
import sys
import tempfile
import unittest
from unittest import mock

import yaml
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config  # noqa: E402
from walk_forward import (  # noqa: E402
    _current_lstm_params_from_config,
    plot_training_progress,
    plot_walk_forward_results,
    plot_window_performance,
    run_augmentation_sweep,
    save_best_hyperparameters_to_config,
)


class TestWalkForwardPlottingAndConfig(unittest.TestCase):
    def setUp(self):
        self._config_backup = copy.deepcopy(config)

    def tearDown(self):
        config.clear()
        config.update(self._config_backup)

    def test_current_lstm_params_reflects_config(self):
        config["indicators"]["lstm_features"]["enabled"] = True
        config["indicators"]["lstm_features"]["lookback"] = 12
        config["indicators"]["lstm_features"]["hidden_size"] = 48

        params = _current_lstm_params_from_config()

        self.assertEqual(params["lookback"], 12)
        self.assertEqual(params["hidden_size"], 48)

    def test_current_lstm_params_returns_none_when_disabled(self):
        config["indicators"]["lstm_features"]["enabled"] = False
        self.assertIsNone(_current_lstm_params_from_config())

    def test_save_best_hyperparameters_to_config_updates_yaml(self):
        payload = {
            "model": {"learning_rate": 0.001},
            "sequence_model": {"lstm_hidden_size": 64},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(payload, f)

            save_best_hyperparameters_to_config(
                {
                    "learning_rate": 0.0002,
                    "n_steps": 512,
                    "ent_coef": 0.15,
                    "lstm_hidden_size": 128,
                    "n_lstm_layers": 2,
                },
                config_path=path,
            )

            with open(path, "r") as f:
                updated = yaml.safe_load(f)

        self.assertEqual(updated["model"]["learning_rate"], 0.0002)
        self.assertEqual(updated["model"]["n_steps"], 512.0)
        self.assertEqual(updated["sequence_model"]["lstm_hidden_size"], 128)
        self.assertEqual(updated["sequence_model"]["n_lstm_layers"], 2)

    def test_run_augmentation_sweep_restores_ratio_and_saves_summary(self):
        config["augmentation"]["tuning"] = {
            "candidate_oversample_ratios": [0.0, 0.2],
            "max_windows": 1,
            "timesteps": 10,
        }
        config["augmentation"]["synthetic_bears"]["oversample_ratio"] = 0.1
        config["indicators"]["lstm_features"]["enabled"] = False

        window_data_list = [
            {
                "window_idx": 1,
                "train_data": pd.DataFrame({"close": [1.0]}),
                "validation_data": pd.DataFrame({"close": [1.0]}),
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch("walk_forward._prepare_tuning_inputs", return_value=(pd.DataFrame({"close": [1.0]}), pd.DataFrame({"close": [1.0]}), {"synthetic_bars": 5})), \
             mock.patch("walk_forward.train_walk_forward_model", return_value=(object(), {})), \
             mock.patch("walk_forward.evaluate_agent", return_value={"total_return_pct": 2.0, "sortino_ratio": 1.0, "calmar_ratio": 0.5, "max_drawdown": -1.0, "trade_count": 3}), \
             mock.patch("walk_forward._score_tuning_trial", side_effect=[(1.0, {"action_counts": {0: 1}, "collapse_flags": []}), (2.0, {"action_counts": {0: 1}, "collapse_flags": []})]):
            summary = run_augmentation_sweep(
                window_data_list,
                session_folder=tmpdir,
                eval_metric="sortino",
            )

            saved_path = os.path.join(tmpdir, "reports", "augmentation_tuning_results.json")
            self.assertTrue(os.path.exists(saved_path))

        self.assertEqual(summary["best_ratio"], 0.2)
        self.assertEqual(config["augmentation"]["synthetic_bears"]["oversample_ratio"], 0.1)

    def test_plot_training_progress_writes_metric_and_portfolio_plots(self):
        training_stats = [
            {"iteration": 0, "return_pct": 1.0, "portfolio_value": 101000.0, "is_best": False, "metric_used": "return"},
            {"iteration": 1, "return_pct": 2.0, "portfolio_value": 102000.0, "is_best": True, "metric_used": "return"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_training_progress(training_stats, tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "training_progress.png")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "portfolio_progress.png")))

    def test_plot_window_performance_writes_png(self):
        idx = pd.date_range("2026-01-01 00:00:00", periods=4, freq="1min", tz="UTC")
        test_data = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]}, index=idx)
        results = {"portfolio_history": [100000.0, 100050.0, 100025.0], "action_history": [0, 1, 0]}
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_window_performance(test_data, results, tmpdir, window_num=1)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "test_performance.png")))

    def test_plot_walk_forward_results_writes_summary_charts(self):
        all_window_results = [
            {"window": 1, "return": 1.0, "portfolio_value": 101000.0, "trade_count": 10, "prediction_accuracy": 60.0, "correct_predictions": 6, "total_predictions": 10},
            {"window": 2, "return": 2.0, "portfolio_value": 102000.0, "trade_count": 12, "prediction_accuracy": 75.0, "correct_predictions": 9, "total_predictions": 12},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "plots"), exist_ok=True)
            plot_walk_forward_results(all_window_results, tmpdir, eval_metric="prediction_accuracy")
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "plots", "walk_forward_results_prediction_accuracy.png")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "plots", "cumulative_performance_prediction_accuracy.png")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
