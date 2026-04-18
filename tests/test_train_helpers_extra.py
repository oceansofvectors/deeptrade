#!/usr/bin/env python
"""Tests for train.py helper utilities and plotting callbacks."""

import os
import sys
import tempfile
import types
import unittest
import copy
from unittest import mock
import json

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train import (  # noqa: E402
    EntropyDecayCallback,
    LossTrackingCallback,
    _checkpoint_prefix,
    _serialize_validation_results_for_report,
    get_configured_algorithm,
    plot_results,
    plot_training_progress,
    save_trade_history,
    train_walk_forward_model,
)


class _FakeFig:
    def __init__(self):
        self.traces = []
        self.annotations = []
        self.layouts = []
        self.yaxes = []
        self.shown = False

    def add_trace(self, trace, **kwargs):
        self.traces.append((trace, kwargs))

    def add_annotation(self, **kwargs):
        self.annotations.append(kwargs)

    def update_layout(self, **kwargs):
        self.layouts.append(kwargs)

    def update_yaxes(self, **kwargs):
        self.yaxes.append(kwargs)

    def show(self):
        self.shown = True


class TestTrainHelpersExtra(unittest.TestCase):
    def setUp(self):
        from config import config  # noqa: WPS433

        self._config_backup = copy.deepcopy(config)

    def tearDown(self):
        from config import config  # noqa: WPS433

        config.clear()
        config.update(self._config_backup)

    def test_checkpoint_prefix_uses_window_folder_when_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = _checkpoint_prefix(tmpdir)
            self.assertTrue(prefix.startswith(tmpdir))
            self.assertTrue(prefix.endswith("best_model"))

    def test_get_configured_algorithm_reads_explicit_benchmark_mode(self):
        from config import config  # noqa: WPS433

        config.setdefault("model", {})["algorithm"] = "qrdqn"
        self.assertEqual(get_configured_algorithm(), "qrdqn")

    def test_checkpoint_prefix_creates_temp_prefix_when_folder_missing(self):
        prefix = _checkpoint_prefix(None)
        self.assertIn("deeptrade_best_model_", prefix)
        self.assertTrue(prefix.endswith("best_model"))
        self.assertTrue(os.path.isdir(os.path.dirname(prefix)))

    def test_loss_tracking_callback_reads_and_resets_losses(self):
        callback = LossTrackingCallback()
        callback.model = types.SimpleNamespace(
            logger=types.SimpleNamespace(
                name_to_value={
                    "train/policy_gradient_loss": 1.5,
                    "train/value_loss": 2.5,
                    "train/entropy_loss": 0.5,
                }
            )
        )

        callback._on_training_end()

        self.assertEqual(callback.get_latest_losses()["value_loss"], 2.5)
        self.assertEqual(callback.get_avg_loss(), 2.5)
        self.assertEqual(callback.n_updates, 1)

        callback.reset()
        self.assertIsNone(callback.get_avg_loss())
        self.assertEqual(callback.n_updates, 0)

    def test_entropy_decay_callback_updates_model_coef(self):
        callback = EntropyDecayCallback(initial_ent=0.20, final_ent=0.05, total_timesteps=100)
        callback.model = types.SimpleNamespace(ent_coef=0.20)
        callback.num_timesteps = 50

        callback._on_step()

        self.assertAlmostEqual(callback.model.ent_coef, 0.125, places=6)

    def test_plot_results_handles_leading_portfolio_point(self):
        fig = _FakeFig()
        results = {
            "dates": ["2026-01-01 00:00:00", "2026-01-01 00:01:00"],
            "price_history": [100.0, 101.0],
            "portfolio_history": [100000.0, 100050.0, 100075.0],
            "buy_dates": ["2026-01-01 00:00:00"],
            "buy_prices": [100.0],
            "sell_dates": ["2026-01-01 00:01:00"],
            "sell_prices": [101.0],
            "final_portfolio_value": 100075.0,
            "total_return_pct": 0.08,
        }

        with mock.patch("train.make_subplots", return_value=fig), \
             mock.patch("train.go.Scatter", side_effect=lambda **kwargs: kwargs):
            plot_results(results)

        self.assertEqual(len(fig.traces), 4)
        self.assertEqual(len(fig.annotations), 1)
        self.assertTrue(fig.shown)

    def test_plot_training_progress_renders_both_series(self):
        fig = _FakeFig()
        all_results = [
            {"total_return_pct": 1.0, "trade_count": 10},
            {"total_return_pct": 2.0, "trade_count": 12},
        ]

        with mock.patch("train.make_subplots", return_value=fig), \
             mock.patch("train.go.Scatter", side_effect=lambda **kwargs: kwargs):
            plot_training_progress(all_results)

        self.assertEqual(len(fig.traces), 2)
        self.assertEqual(len(fig.yaxes), 2)
        self.assertTrue(fig.shown)

    def test_save_trade_history_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "train_trade_history.csv")
            save_trade_history(
                [{"date": "2026-01-01", "trade_type": "Long Entry", "price": 100.0}],
                filename=path,
            )
            self.assertTrue(os.path.exists(path))

    def test_serialize_validation_results_for_report_keeps_reporting_fields(self):
        payload = _serialize_validation_results_for_report(
            {
                "final_portfolio_value": 101000.0,
                "total_return_pct": 1.0,
                "trade_count": 25,
                "rebalance_count": 25,
                "completed_trades": 10,
                "economic_trade_count": 10,
                "sortino_ratio": 1.5,
                "calmar_ratio": 0.6,
                "max_drawdown": -4.0,
                "selected_via_fallback": True,
                "collapse_flags": ["too_few_trades"],
            },
            evaluation_metric="sortino",
        )

        self.assertEqual(payload["evaluation_metric_used"], "sortino")
        self.assertEqual(payload["economic_trade_count"], 10)
        self.assertEqual(payload["completed_trades"], 10)
        self.assertEqual(payload["sortino_ratio"], 1.5)
        self.assertEqual(payload["calmar_ratio"], 0.6)
        self.assertEqual(payload["max_drawdown"], -4.0)
        self.assertTrue(payload["selected_via_fallback"])
        self.assertEqual(payload["collapse_flags"], ["too_few_trades"])

    def test_train_walk_forward_model_uses_defaults_and_writes_artifacts(self):
        train_df = pd.DataFrame({"close": [1.0]})
        val_df = pd.DataFrame({"close": [1.0]})
        fake_model = mock.MagicMock()
        validation_results = {
            "final_portfolio_value": 101000.0,
            "total_return_pct": 1.0,
            "trade_count": 5,
            "completed_trades": 2,
            "economic_trade_count": 2,
            "sortino_ratio": 1.4,
            "calmar_ratio": 0.5,
            "max_drawdown": -2.5,
            "selected_via_fallback": True,
            "collapse_flags": ["too_few_trades"],
            "loss_history": [1.0, 0.5],
        }
        all_results = [
            {
                "total_return_pct": 1.0,
                "final_portfolio_value": 101000.0,
                "trade_count": 5,
                "is_best": True,
                "metric_used": "return",
                "action_counts": {0: 3, 6: 2},
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch("train.train_agent_iteratively", return_value=(fake_model, validation_results, all_results)):
            model, result = train_walk_forward_model(
                train_df,
                val_df,
                initial_timesteps=1,
                additional_timesteps=1,
                max_iterations=1,
                window_folder=tmpdir,
                run_hyperparameter_tuning=False,
            )

            self.assertIs(model, fake_model)
            self.assertEqual(result["loss_history"], [1.0, 0.5])
            self.assertEqual(len(result["iterations"]), 1)
            fake_model.save.assert_called_once_with(os.path.join(tmpdir, "model"))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "validation_results.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "training_stats.json")))
            with open(os.path.join(tmpdir, "validation_results.json"), "r") as f:
                saved_validation = json.load(f)
            self.assertEqual(saved_validation["economic_trade_count"], 2)
            self.assertEqual(saved_validation["sortino_ratio"], 1.4)
            self.assertTrue(saved_validation["selected_via_fallback"])

    def test_train_walk_forward_model_can_use_tuning_results(self):
        train_df = pd.DataFrame({"close": [1.0]})
        val_df = pd.DataFrame({"close": [1.0]})
        fake_model = mock.MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir, \
             mock.patch("walk_forward.hyperparameter_tuning", return_value={"best_params": {"learning_rate": 0.0002, "n_steps": 256, "ent_coef": 0.12}}), \
             mock.patch("train.train_agent_iteratively", return_value=(fake_model, {"loss_history": []}, [])) as iterative_mock:
            train_walk_forward_model(
                train_df,
                val_df,
                initial_timesteps=1,
                additional_timesteps=1,
                max_iterations=1,
                tuning_folder=tmpdir,
                run_hyperparameter_tuning=True,
                tuning_trials=2,
            )

            passed_params = iterative_mock.call_args.kwargs["model_params"]
            self.assertEqual(passed_params["learning_rate"], 0.0002)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "best_params.json")))
            with open(os.path.join(tmpdir, "best_params.json"), "r") as f:
                saved = json.load(f)
            self.assertEqual(saved["n_steps"], 256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
