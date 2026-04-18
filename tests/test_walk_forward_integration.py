#!/usr/bin/env python
"""Integration-style tests for walk-forward plumbing."""

import copy
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config  # noqa: E402
from environment import TradingEnv  # noqa: E402
from get_data import process_technical_indicators  # noqa: E402
from walk_forward import (  # noqa: E402
    hyperparameter_tuning,
    load_tradingview_data,
    process_single_window,
    walk_forward_testing,
)


def _raw_ohlcv(rows=120):
    idx = pd.date_range("2026-01-01 00:00:00", periods=rows, freq="1min", tz="UTC")
    close = np.linspace(95000.0, 95500.0, rows)
    frame = pd.DataFrame(
        {
            "open": close - 2.0,
            "high": close + 3.0,
            "low": close - 3.0,
            "close": close,
            "volume": np.linspace(10.0, 20.0, rows),
        },
        index=idx,
    )
    return frame


class TestEnvironmentFeatureInclusion(unittest.TestCase):
    def test_environment_includes_new_numeric_features_by_default(self):
        df = pd.DataFrame(
            {
                "close_norm": [0.4, 0.5, 0.6],
                "close": [95000.0, 95010.0, 95020.0],
                "open": [94995.0, 95005.0, 95015.0],
                "high": [95005.0, 95015.0, 95025.0],
                "low": [94990.0, 95000.0, 95010.0],
                "volume": [10.0, 11.0, 12.0],
                "VWAP_DIST_PCT": [0.0, 0.01, 0.02],
                "VWAP_DIST_Z": [0.0, 0.5, 1.0],
                "OPENING_RANGE_WIDTH_PCT": [0.0, 0.1, 0.1],
                "DIST_TO_OR_HIGH_PCT": [0.0, -0.05, 0.02],
                "MSO_SIN": [0.0, 0.1, 0.2],
                "rtype": [1, 1, 1],
                "publisher_id": [44, 44, 44],
            }
        )

        env = TradingEnv(df, initial_balance=100000.0, transaction_cost=0.0, position_size=1)
        obs, _ = env.reset(seed=42)

        self.assertIn("VWAP_DIST_PCT", env.technical_indicators)
        self.assertIn("VWAP_DIST_Z", env.technical_indicators)
        self.assertIn("OPENING_RANGE_WIDTH_PCT", env.technical_indicators)
        self.assertIn("DIST_TO_OR_HIGH_PCT", env.technical_indicators)
        self.assertNotIn("rtype", env.technical_indicators)
        self.assertNotIn("publisher_id", env.technical_indicators)
        self.assertEqual(obs.shape[0], 1 + len(env.technical_indicators) + 4)


class TestTradingViewLoader(unittest.TestCase):
    def test_load_tradingview_data_accepts_ts_event_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mbt.csv")
            rows = [
                {"ts_event": "2026-01-01T00:00:00Z", "open": 95000, "high": 95010, "low": 94990, "close": 95005, "volume": 10},
                {"ts_event": "2026-01-01T00:00:00Z", "open": 95001, "high": 95011, "low": 94991, "close": 95006, "volume": 11},
                {"ts_event": "2026-01-01T00:01:00Z", "open": 95010, "high": 95020, "low": 95000, "close": 95015, "volume": 12},
            ]
            pd.DataFrame(rows).to_csv(path, index=False)

            loaded = load_tradingview_data(path)

        self.assertEqual(len(loaded), 2)
        self.assertTrue(loaded.index.is_monotonic_increasing)
        self.assertFalse(loaded.index.has_duplicates)
        self.assertAlmostEqual(float(loaded.iloc[0]["close"]), 95006.0)

    def test_load_tradingview_data_filters_to_session_dominant_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mbt_multi_contract.csv")
            rows = [
                # Session 1: MBTX4 should dominate overall volume despite overlap.
                {"ts_event": "2026-01-02T00:00:00Z", "open": 95000, "high": 95010, "low": 94990, "close": 95005, "volume": 500, "symbol": "MBTX4", "instrument_id": 1},
                {"ts_event": "2026-01-02T00:00:00Z", "open": 96000, "high": 96010, "low": 95990, "close": 96005, "volume": 10, "symbol": "MBTH5", "instrument_id": 2},
                {"ts_event": "2026-01-02T00:01:00Z", "open": 95010, "high": 95020, "low": 95000, "close": 95015, "volume": 500, "symbol": "MBTX4", "instrument_id": 1},
                {"ts_event": "2026-01-02T00:01:00Z", "open": 96010, "high": 96020, "low": 96000, "close": 96015, "volume": 200, "symbol": "MBTH5", "instrument_id": 2},
                # Session 2: MBTH5 dominates and should be selected after the roll.
                {"ts_event": "2026-01-02T23:00:00Z", "open": 97000, "high": 97010, "low": 96990, "close": 97005, "volume": 20, "symbol": "MBTX4", "instrument_id": 1},
                {"ts_event": "2026-01-02T23:00:00Z", "open": 98000, "high": 98010, "low": 97990, "close": 98005, "volume": 600, "symbol": "MBTH5", "instrument_id": 2},
                {"ts_event": "2026-01-02T23:01:00Z", "open": 97010, "high": 97020, "low": 97000, "close": 97015, "volume": 20, "symbol": "MBTX4", "instrument_id": 1},
                {"ts_event": "2026-01-02T23:01:00Z", "open": 98010, "high": 98020, "low": 98000, "close": 98015, "volume": 600, "symbol": "MBTH5", "instrument_id": 2},
            ]
            pd.DataFrame(rows).to_csv(path, index=False)

            loaded = load_tradingview_data(path)

        self.assertEqual(len(loaded), 4)
        self.assertTrue(loaded.index.is_monotonic_increasing)
        self.assertFalse(loaded.index.has_duplicates)
        self.assertEqual(loaded.iloc[0]["symbol"], "MBTX4")
        self.assertEqual(loaded.iloc[1]["symbol"], "MBTX4")
        self.assertEqual(loaded.iloc[2]["symbol"], "MBTH5")
        self.assertEqual(loaded.iloc[3]["symbol"], "MBTH5")
        self.assertAlmostEqual(float(loaded.iloc[0]["close"]), 95005.0)
        self.assertAlmostEqual(float(loaded.iloc[2]["close"]), 98005.0)

    def test_load_tradingview_data_drops_rows_with_non_numeric_ohlcv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mbt_bad_row.csv")
            rows = [
                {"ts_event": "2026-01-01T00:00:00Z", "open": "95000", "high": "95010", "low": "94990", "close": "95005", "volume": "10"},
                {"ts_event": "2026-01-01T00:01:00Z", "open": "oops", "high": "95020", "low": "95000", "close": "95015", "volume": "12"},
                {"ts_event": "2026-01-01T00:02:00Z", "open": "95020", "high": "95030", "low": "95010", "close": "95025", "volume": "11"},
            ]
            pd.DataFrame(rows).to_csv(path, index=False)

            loaded = load_tradingview_data(path)

        self.assertEqual(len(loaded), 2)
        self.assertListEqual(list(loaded["close"].astype(float)), [95005.0, 95025.0])


class TestHyperparameterTuningStages(unittest.TestCase):
    def setUp(self):
        self._config_backup = copy.deepcopy(config)
        config["model"]["algorithm"] = "ppo"
        config["sequence_model"]["enabled"] = False

    def tearDown(self):
        config.clear()
        config.update(self._config_backup)

    def test_hyperparameter_tuning_runs_stage1_and_stage2(self):
        config["hyperparameter_tuning"]["stage1_trials"] = 2
        config["hyperparameter_tuning"]["stage2_trials"] = 1
        config["hyperparameter_tuning"]["stage1_timesteps"] = 1000
        config["hyperparameter_tuning"]["stage2_timesteps"] = 2000
        config["hyperparameter_tuning"]["stage2_window_shrink"] = 0.2

        train_df = pd.DataFrame({"close_norm": [0.1, 0.2, 0.3], "feat": [1.0, 2.0, 3.0]})
        val_df = pd.DataFrame({"close_norm": [0.2, 0.3], "feat": [2.0, 3.0]})
        holdout_df = pd.DataFrame({"close_norm": [0.25, 0.35], "feat": [2.5, 3.5]})

        stage_calls = []

        def fake_run_stage(**kwargs):
            stage_calls.append(kwargs)
            if kwargs["stage_name"] == "stage1":
                return {
                    "best_params": {
                        "learning_rate": 1e-4,
                        "n_steps": 512,
                        "ent_coef": 0.15,
                        "reward_turnover_penalty": 0.01,
                    },
                    "best_value": 1.0,
                    "study": None,
                }
            return {
                "best_params": {
                    "learning_rate": 8e-5,
                    "n_steps": 480,
                    "ent_coef": 0.14,
                    "reward_turnover_penalty": 0.011,
                },
                "best_value": 1.4,
                "study": None,
            }

        with mock.patch("walk_forward._score_constant_action_baselines", return_value=[{0: 0.0, 1: 0.0, 2: 0.0}] * 3), \
             mock.patch("walk_forward._run_tuning_stage", side_effect=fake_run_stage):
            result = hyperparameter_tuning(
                train_data=train_df,
                validation_data=val_df,
                n_trials=3,
                holdout_validation_sets=[holdout_df],
            )

        self.assertEqual(result["selected_stage"], "stage2")
        self.assertEqual(len(stage_calls), 2)
        self.assertEqual(len(stage_calls[0]["evaluation_sets"]), 1)
        self.assertEqual(len(stage_calls[1]["evaluation_sets"]), 2)
        self.assertIn("learning_rate", result["stage2_search_space"])
        self.assertLess(
            result["stage2_search_space"]["learning_rate"]["max"] - result["stage2_search_space"]["learning_rate"]["min"],
            config["hyperparameter_tuning"]["parameters"]["learning_rate"]["max"]
            - config["hyperparameter_tuning"]["parameters"]["learning_rate"]["min"],
        )

    def test_hyperparameter_tuning_dispatches_to_qrdqn_tuner(self):
        train_df = pd.DataFrame({"close_norm": [0.1, 0.2], "feat": [1.0, 2.0]})
        val_df = pd.DataFrame({"close_norm": [0.3, 0.4], "feat": [3.0, 4.0]})

        with mock.patch("walk_forward.get_configured_algorithm", return_value="qrdqn"), \
             mock.patch(
                 "walk_forward._run_qrdqn_tuning",
                 return_value={
                     "best_params": {"learning_rate": 2e-4, "batch_size": 64, "gamma": 0.99},
                     "best_value": 1.25,
                     "selected_stage": "qrdqn",
                     "stage1_best_params": {"learning_rate": 2e-4, "batch_size": 64, "gamma": 0.99},
                     "stage1_best_value": 1.25,
                     "stage2_best_params": {},
                     "stage2_best_value": float("-inf"),
                     "stage2_search_space": {},
                     "study": None,
                 },
             ) as qrdqn_mock:
            result = hyperparameter_tuning(train_df, val_df, n_trials=2)

        qrdqn_mock.assert_called_once()
        self.assertEqual(result["selected_stage"], "qrdqn")
        self.assertEqual(result["best_params"]["batch_size"], 64)


class TestWalkForwardTestingWindows(unittest.TestCase):
    def test_walk_forward_testing_respects_max_windows(self):
        config_backup = copy.deepcopy(config)
        try:
            config["indicators"]["lstm_features"]["tuning"]["enabled"] = False
            idx = pd.date_range("2026-01-01", periods=24 * 8, freq="1h", tz="UTC")
            data = pd.DataFrame({"close": np.linspace(1.0, 2.0, len(idx))}, index=idx)

            with tempfile.TemporaryDirectory() as tmpdir:
                old_cwd = os.getcwd()
                os.chdir(tmpdir)
                try:
                    with mock.patch("walk_forward.process_single_window") as process_mock, \
                         mock.patch("walk_forward.plot_walk_forward_results"), \
                         mock.patch("walk_forward.export_consolidated_trade_history"), \
                         mock.patch("walk_forward.generate_walk_forward_report", return_value="reports/test.html"):
                        process_mock.side_effect = [
                            {"window": 1, "return": 1.0, "portfolio_value": 101000.0, "trade_count": 5, "sortino_ratio": 1.2},
                            {"window": 2, "return": 2.0, "portfolio_value": 102000.0, "trade_count": 6, "sortino_ratio": 1.4},
                        ]
                        result = walk_forward_testing(
                            data=data,
                            window_size=2,
                            step_size=1,
                            train_ratio=0.5,
                            validation_ratio=0.25,
                            initial_timesteps=1,
                            additional_timesteps=1,
                            max_iterations=1,
                            max_windows=2,
                        )
                finally:
                    os.chdir(old_cwd)
        finally:
            config.clear()
            config.update(config_backup)

        self.assertEqual(process_mock.call_count, 2)
        self.assertEqual(result["num_windows"], 2)
        self.assertEqual(len(result["all_window_results"]), 2)
        self.assertEqual(result["html_report_path"], "reports/test.html")

    def test_walk_forward_testing_defaults_to_market_hours_filter_when_missing_setting(self):
        config_backup = copy.deepcopy(config)
        try:
            config["indicators"]["lstm_features"]["tuning"]["enabled"] = False
            config["data"].pop("market_hours_only", None)
            idx = pd.date_range("2026-01-05 14:30:00", periods=24 * 4, freq="1h", tz="UTC")
            data = pd.DataFrame({"close": np.linspace(1.0, 2.0, len(idx))}, index=idx)

            with tempfile.TemporaryDirectory() as tmpdir:
                old_cwd = os.getcwd()
                os.chdir(tmpdir)
                try:
                    with mock.patch("walk_forward.filter_market_hours", side_effect=lambda frame: frame) as filter_mock, \
                         mock.patch("walk_forward.process_single_window", return_value={"window": 1, "return": 1.0, "portfolio_value": 101000.0, "trade_count": 1, "sortino_ratio": 1.0}), \
                         mock.patch("walk_forward.plot_walk_forward_results"), \
                         mock.patch("walk_forward.export_consolidated_trade_history"), \
                         mock.patch("walk_forward.generate_walk_forward_report", return_value="reports/test.html"):
                        walk_forward_testing(
                            data=data,
                            window_size=2,
                            step_size=1,
                            train_ratio=0.5,
                            validation_ratio=0.25,
                            initial_timesteps=1,
                            additional_timesteps=1,
                            max_iterations=1,
                            max_windows=1,
                        )
                finally:
                    os.chdir(old_cwd)
        finally:
            config.clear()
            config.update(config_backup)

        filter_mock.assert_called_once()

    def test_walk_forward_testing_retunes_on_configured_window_cadence(self):
        config_backup = copy.deepcopy(config)
        try:
            config["indicators"]["lstm_features"]["tuning"]["enabled"] = False
            config["hyperparameter_tuning"]["retune_every_windows"] = 2
            idx = pd.date_range("2026-01-05 14:30:00", periods=24 * 10, freq="1h", tz="UTC")
            data = pd.DataFrame({"close": np.linspace(1.0, 2.0, len(idx))}, index=idx)

            with tempfile.TemporaryDirectory() as tmpdir:
                old_cwd = os.getcwd()
                os.chdir(tmpdir)
                try:
                    with mock.patch("walk_forward._run_window_hyperparameter_tuning") as tune_mock, \
                         mock.patch("walk_forward.process_single_window") as process_mock, \
                         mock.patch("walk_forward.plot_walk_forward_results"), \
                         mock.patch("walk_forward.export_consolidated_trade_history"), \
                         mock.patch("walk_forward.generate_walk_forward_report", return_value="reports/test.html"):
                        tune_mock.side_effect = [
                            ({"learning_rate": 1e-4}, {"selected_stage": "stage1"}),
                            ({"learning_rate": 2e-4}, {"selected_stage": "stage1"}),
                        ]
                        process_mock.side_effect = [
                            {"window": 1, "return": 1.0, "portfolio_value": 101000.0, "trade_count": 5, "sortino_ratio": 1.2},
                            {"window": 2, "return": 2.0, "portfolio_value": 102000.0, "trade_count": 6, "sortino_ratio": 1.4},
                            {"window": 3, "return": 3.0, "portfolio_value": 103000.0, "trade_count": 7, "sortino_ratio": 1.6},
                        ]
                        walk_forward_testing(
                            data=data,
                            window_size=3,
                            step_size=2,
                            train_ratio=0.5,
                            validation_ratio=0.25,
                            initial_timesteps=1,
                            additional_timesteps=1,
                            max_iterations=1,
                            max_windows=3,
                            run_hyperparameter_tuning=True,
                            tuning_trials=2,
                        )
                finally:
                    os.chdir(old_cwd)
        finally:
            config.clear()
            config.update(config_backup)

        self.assertEqual(tune_mock.call_count, 2)
        self.assertEqual(process_mock.call_count, 3)


class TestProcessSingleWindow(unittest.TestCase):
    def setUp(self):
        self._config_backup = copy.deepcopy(config)
        config["model"]["algorithm"] = "ppo"
        config["representation"]["mode"] = "hybrid"
        config["sequence_model"]["enabled"] = False
        config["indicators"]["lstm_features"]["enabled"] = True
        config["indicators"]["lstm_features"]["tuning"]["enabled"] = False

    def tearDown(self):
        config.clear()
        config.update(self._config_backup)

    def test_process_single_window_passes_vae_config_and_writes_report_data(self):
        config["augmentation"]["synthetic_bears"]["enabled"] = False
        config["indicators"]["lstm_features"]["enabled"] = True
        config["indicators"]["lstm_features"]["tuning"]["enabled"] = False
        config["indicators"]["lstm_features"]["lookback"] = 10
        config["indicators"]["lstm_features"]["hidden_size"] = 32
        config["indicators"]["lstm_features"]["num_layers"] = 2
        config["indicators"]["lstm_features"]["output_size"] = 4
        config["indicators"]["lstm_features"]["beta"] = 0.0008266502663726915
        config["indicators"]["lstm_features"]["kl_warmup_epochs"] = 10
        config["sequence_model"]["enabled"] = False
        config["risk_management"]["enabled"] = False

        raw = _raw_ohlcv(90)
        train = process_technical_indicators(raw.iloc[:50].copy())
        val = process_technical_indicators(raw.iloc[50:70].copy())
        test = process_technical_indicators(raw.iloc[70:].copy())
        window_data = pd.concat([train, val, test])

        class FakeLSTMGenerator:
            last_init = None

            def __init__(self, **kwargs):
                FakeLSTMGenerator.last_init = kwargs
                self.output_size = kwargs["output_size"]

            def fit(self, train_df, validation_df, checkpoint_path=None):
                return None

            def transform(self, df):
                result = df.copy()
                for i in range(self.output_size):
                    result[f"LATENT_F{i}"] = float(i)
                return result

            def save(self, path):
                with open(path, "wb") as handle:
                    handle.write(b"fake")

        fake_training_stats = {
            "iterations": [
                {"iteration": 0, "return_pct": 1.0, "sortino_ratio": 1.1, "trade_count": 5, "is_best": True}
            ],
            "loss_history": [],
        }
        fake_results = {
            "total_return_pct": 1.5,
            "final_portfolio_value": 101500.0,
            "trade_count": 5,
            "final_position": 0,
            "hit_rate": 60.0,
            "profitable_trades": 3,
            "max_drawdown": -2.5,
            "calmar_ratio": 0.6,
            "sortino_ratio": 1.4,
            "dates": list(test.index),
            "price_history": list(test["close"]),
            "portfolio_history": [100000.0] + [100100.0 + i for i in range(len(test.index))],
            "position_history": [0] * (len(test.index) + 1),
            "drawdown_history": [0.0] * (len(test.index) + 1),
            "action_history": [0] * len(test.index),
            "action_counts": {0: len(test.index), 1: 0, 2: 0},
            "trade_history": [{"date": test.index[0], "trade_type": "Long", "price": float(test["close"].iloc[0])}],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            window_folder = os.path.join(tmpdir, "window_1")
            os.makedirs(window_folder, exist_ok=True)
            with mock.patch("walk_forward.LSTMFeatureGenerator", FakeLSTMGenerator), \
                 mock.patch("walk_forward.train_walk_forward_model", return_value=(object(), fake_training_stats)), \
                 mock.patch("walk_forward.evaluate_agent", return_value=fake_results), \
                 mock.patch("walk_forward.plot_training_progress"), \
                 mock.patch("walk_forward.plot_window_performance"), \
                 mock.patch("walk_forward.save_trade_history"):
                result = process_single_window(
                    window_idx=1,
                    num_windows=2,
                    window_data=window_data,
                    train_data=train,
                    validation_data=val,
                    test_data=test,
                    window_folder=window_folder,
                    initial_timesteps=100,
                    additional_timesteps=10,
                    max_iterations=1,
                    n_stagnant_loops=1,
                    improvement_threshold=0.1,
                    run_hyperparameter_tuning=False,
                    tuning_trials=1,
                    best_hyperparameters=None,
                )

            report_path = os.path.join(window_folder, "report_data.json")
            self.assertTrue(os.path.exists(report_path))
            with open(report_path, "r") as handle:
                payload = json.load(handle)

        self.assertEqual(FakeLSTMGenerator.last_init["beta"], config["indicators"]["lstm_features"]["beta"])
        self.assertEqual(
            FakeLSTMGenerator.last_init["kl_warmup_epochs"],
            config["indicators"]["lstm_features"]["kl_warmup_epochs"],
        )
        self.assertEqual(payload["metrics"]["trade_count"], 5)
        self.assertEqual(result["report_data_path"].endswith("report_data.json"), True)
        self.assertEqual(result["action_history"], fake_results["action_history"])
        self.assertEqual(result["trade_history"], fake_results["trade_history"])

    def test_process_single_window_uses_session_lstm_params_without_retuning(self):
        config["augmentation"]["synthetic_bears"]["enabled"] = False
        config["indicators"]["lstm_features"]["enabled"] = True
        config["indicators"]["lstm_features"]["tuning"]["enabled"] = True
        config["sequence_model"]["enabled"] = False
        config["risk_management"]["enabled"] = False

        raw = _raw_ohlcv(90)
        train = process_technical_indicators(raw.iloc[:50].copy())
        val = process_technical_indicators(raw.iloc[50:70].copy())
        test = process_technical_indicators(raw.iloc[70:].copy())
        window_data = pd.concat([train, val, test])

        resolved_params = {
            "lookback": 9,
            "hidden_size": 24,
            "num_layers": 1,
            "output_size": 3,
            "beta": 0.001,
            "kl_warmup_epochs": 2,
            "pretrain_epochs": 2,
            "pretrain_lr": 0.001,
            "pretrain_batch_size": 8,
            "pretrain_patience": 2,
            "pretrain_min_delta": 0.0001,
        }

        class FakeLSTMGenerator:
            last_init = None

            def __init__(self, **kwargs):
                FakeLSTMGenerator.last_init = kwargs
                self.output_size = kwargs["output_size"]

            def fit(self, train_df, validation_df, checkpoint_path=None):
                return None

            def transform(self, df):
                result = df.copy()
                for i in range(self.output_size):
                    result[f"LATENT_F{i}"] = float(i)
                return result

            def save(self, path):
                with open(path, "wb") as handle:
                    handle.write(b"fake")

        fake_results = {
            "total_return_pct": 1.5,
            "final_portfolio_value": 101500.0,
            "trade_count": 5,
            "final_position": 0,
            "hit_rate": 60.0,
            "profitable_trades": 3,
            "max_drawdown": -2.5,
            "calmar_ratio": 0.6,
            "sortino_ratio": 1.4,
            "dates": list(test.index),
            "price_history": list(test["close"]),
            "portfolio_history": [100000.0] + [100100.0 + i for i in range(len(test.index))],
            "position_history": [0] * (len(test.index) + 1),
            "drawdown_history": [0.0] * (len(test.index) + 1),
            "action_history": [0] * len(test.index),
            "action_counts": {0: len(test.index)},
            "trade_history": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            window_folder = os.path.join(tmpdir, "window_2")
            os.makedirs(window_folder, exist_ok=True)
            with mock.patch("walk_forward.LSTMFeatureGenerator", FakeLSTMGenerator), \
                 mock.patch("walk_forward.tune_lstm_hyperparameters") as tune_mock, \
                 mock.patch("walk_forward.train_walk_forward_model", return_value=(object(), {"iterations": [], "loss_history": []})), \
                 mock.patch("walk_forward.evaluate_agent", return_value=fake_results), \
                 mock.patch("walk_forward.plot_training_progress"), \
                 mock.patch("walk_forward.plot_window_performance"), \
                 mock.patch("walk_forward.save_trade_history"):
                process_single_window(
                    window_idx=2,
                    num_windows=2,
                    window_data=window_data,
                    train_data=train,
                    validation_data=val,
                    test_data=test,
                    window_folder=window_folder,
                    initial_timesteps=10,
                    additional_timesteps=5,
                    max_iterations=1,
                    n_stagnant_loops=1,
                    improvement_threshold=0.1,
                    run_hyperparameter_tuning=False,
                    tuning_trials=1,
                    best_hyperparameters=None,
                    resolved_lstm_params=resolved_params,
                    lstm_tuning_scope="session",
                )

        tune_mock.assert_not_called()
        self.assertEqual(FakeLSTMGenerator.last_init["hidden_size"], 24)
        self.assertEqual(FakeLSTMGenerator.last_init["output_size"], 3)

    def test_process_single_window_retunes_lstm_when_scope_is_per_window(self):
        config["augmentation"]["synthetic_bears"]["enabled"] = False
        config["indicators"]["lstm_features"]["enabled"] = True
        config["indicators"]["lstm_features"]["tuning"]["enabled"] = True
        config["sequence_model"]["enabled"] = False
        config["risk_management"]["enabled"] = False

        raw = _raw_ohlcv(90)
        train = process_technical_indicators(raw.iloc[:50].copy())
        val = process_technical_indicators(raw.iloc[50:70].copy())
        test = process_technical_indicators(raw.iloc[70:].copy())
        window_data = pd.concat([train, val, test])

        tuned_params = {
            "lookback": 12,
            "hidden_size": 40,
            "num_layers": 2,
            "output_size": 5,
            "beta": 0.002,
            "kl_warmup_epochs": 3,
            "pretrain_epochs": 2,
            "pretrain_lr": 0.0008,
            "pretrain_batch_size": 8,
            "pretrain_patience": 2,
            "pretrain_min_delta": 0.0001,
        }

        class FakeLSTMGenerator:
            last_init = None

            def __init__(self, **kwargs):
                FakeLSTMGenerator.last_init = kwargs
                self.output_size = kwargs["output_size"]

            def fit(self, train_df, validation_df, checkpoint_path=None):
                return None

            def transform(self, df):
                result = df.copy()
                for i in range(self.output_size):
                    result[f"LATENT_F{i}"] = float(i)
                return result

            def save(self, path):
                with open(path, "wb") as handle:
                    handle.write(b"fake")

        fake_results = {
            "total_return_pct": 1.5,
            "final_portfolio_value": 101500.0,
            "trade_count": 5,
            "final_position": 0,
            "hit_rate": 60.0,
            "profitable_trades": 3,
            "max_drawdown": -2.5,
            "calmar_ratio": 0.6,
            "sortino_ratio": 1.4,
            "dates": list(test.index),
            "price_history": list(test["close"]),
            "portfolio_history": [100000.0] + [100100.0 + i for i in range(len(test.index))],
            "position_history": [0] * (len(test.index) + 1),
            "drawdown_history": [0.0] * (len(test.index) + 1),
            "action_history": [0] * len(test.index),
            "action_counts": {0: len(test.index)},
            "trade_history": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            window_folder = os.path.join(tmpdir, "window_3")
            os.makedirs(window_folder, exist_ok=True)
            with mock.patch("walk_forward.LSTMFeatureGenerator", FakeLSTMGenerator), \
                 mock.patch("walk_forward.tune_lstm_hyperparameters", return_value=tuned_params) as tune_mock, \
                 mock.patch("walk_forward.train_walk_forward_model", return_value=(object(), {"iterations": [], "loss_history": []})), \
                 mock.patch("walk_forward.evaluate_agent", return_value=fake_results), \
                 mock.patch("walk_forward.plot_training_progress"), \
                 mock.patch("walk_forward.plot_window_performance"), \
                 mock.patch("walk_forward.save_trade_history"):
                process_single_window(
                    window_idx=3,
                    num_windows=3,
                    window_data=window_data,
                    train_data=train,
                    validation_data=val,
                    test_data=test,
                    window_folder=window_folder,
                    initial_timesteps=10,
                    additional_timesteps=5,
                    max_iterations=1,
                    n_stagnant_loops=1,
                    improvement_threshold=0.1,
                    run_hyperparameter_tuning=False,
                    tuning_trials=1,
                    best_hyperparameters=None,
                    resolved_lstm_params=None,
                    lstm_tuning_scope="per_window",
                )

        tune_mock.assert_called_once()
        self.assertEqual(FakeLSTMGenerator.last_init["hidden_size"], 40)
        self.assertEqual(FakeLSTMGenerator.last_init["output_size"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
