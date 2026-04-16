#!/usr/bin/env python
"""Tests for walk-forward helper logic outside the main orchestration loop."""

import os
import sys
import unittest
import copy
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config  # noqa: E402
from walk_forward import (  # noqa: E402
    _apply_trial_config_overrides,
    _align_feature_columns,
    _build_tuning_env,
    _deterministic_mode_enabled,
    _drop_unused_model_columns,
    _narrow_hp_config,
    _prepare_tuning_inputs,
    _resolve_tuning_n_jobs,
    _restore_trial_config_overrides,
    _resolve_close_column,
    _sanitize_ohlc_outliers,
    _sanitize_cuda_library_path,
    _sample_stage_params,
    _score_constant_action_baselines,
    _summarize_trial_scores,
    _valid_batch_size_choices,
    build_window_report_payload,
    calculate_hit_rate_from_trade_results,
    evaluate_agent_prediction_accuracy,
    save_json,
)


class TestWalkForwardHelpers(unittest.TestCase):
    def setUp(self):
        self._config_backup = copy.deepcopy(config)

    def tearDown(self):
        config.clear()
        config.update(self._config_backup)

    class _SequenceModel:
        def __init__(self, actions):
            self.actions = list(actions)
            self.idx = 0

        def predict(self, obs, deterministic=True):
            action = self.actions[min(self.idx, len(self.actions) - 1)]
            self.idx += 1
            return action, None

    def test_calculate_hit_rate_counts_only_completed_trades(self):
        results = {
            "trade_count": 4,
            "trade_history": [
                {"action": "buy"},
                {"profit": 10.0},
                {"profit": -5.0},
                {"profit": 2.5},
            ],
        }

        enriched = calculate_hit_rate_from_trade_results(results)

        self.assertEqual(enriched["profitable_trades"], 2)
        self.assertEqual(enriched["completed_trades"], 3)
        self.assertAlmostEqual(enriched["hit_rate"], 66.66666666666666)

    def test_build_window_report_payload_trims_series_lengths_consistently(self):
        idx = pd.date_range("2026-01-01 00:00:00", periods=3, freq="1min", tz="UTC")
        test_data = pd.DataFrame({"close": [100.0, 101.0, 102.0]}, index=idx)
        test_results = {
            "dates": list(idx),
            "price_history": [100.0, 101.0, 102.0],
            "portfolio_history": [100000.0, 100100.0, 100200.0, 100300.0],
            "position_history": [0, 1, 0, 0],
            "drawdown_history": [0.0, -0.1, -0.2, -0.3],
            "action_history": [0, 6, 6],
            "trade_history": [],
            "action_counts": {0: 1, 6: 2},
            "total_return_pct": 0.3,
            "final_portfolio_value": 100300.0,
            "trade_count": 1,
            "hit_rate": 100.0,
            "profitable_trades": 1,
            "prediction_accuracy": 0.0,
            "correct_predictions": 0,
            "total_predictions": 0,
            "max_drawdown": -0.3,
            "calmar_ratio": 1.0,
            "sortino_ratio": 2.0,
            "final_position": 0,
        }

        payload = build_window_report_payload(
            window_idx=1,
            test_data=test_data,
            test_results=test_results,
            training_stats={"iterations": []},
            validation_results={},
            window_periods={"train_start": str(idx[0]), "test_end": str(idx[-1])},
            evaluation_metric="sortino",
        )

        series = payload["series"]
        self.assertEqual(len(series["portfolio_timestamps"]), 4)
        self.assertEqual(len(series["portfolio_values"]), 4)
        self.assertEqual(len(series["position_timestamps"]), 4)
        self.assertEqual(len(series["position_values"]), 4)
        self.assertEqual(len(series["drawdown_timestamps"]), 4)
        self.assertEqual(len(series["drawdown_values"]), 4)
        self.assertEqual(payload["metrics"]["final_portfolio_value"], 100300.0)

    def test_align_feature_columns_adds_missing_columns_with_zeroes(self):
        df = pd.DataFrame({"close": [100.0], "foo": [2.0]})
        aligned = _align_feature_columns(["close", "bar"], df)

        self.assertEqual(list(aligned.columns), ["close", "bar"])
        self.assertEqual(aligned.iloc[0]["close"], 100.0)
        self.assertEqual(aligned.iloc[0]["bar"], 0.0)

    def test_sanitize_ohlc_outliers_replaces_implausible_positive_bars(self):
        df = pd.DataFrame(
            {
                "open": [30000.0, 175.0, 30100.0],
                "high": [30010.0, 175.0, 30110.0],
                "low": [29990.0, 175.0, 30090.0],
                "close": [30000.0, 175.0, 30100.0],
                "volume": [1.0, 1.0, 1.0],
            }
        )

        cleaned = _sanitize_ohlc_outliers(df)

        self.assertGreater(cleaned.loc[1, "close"], 1000.0)
        self.assertEqual(cleaned.loc[1, "close"], cleaned.loc[0, "close"])
        self.assertEqual(cleaned.loc[1, "open"], cleaned.loc[0, "open"])
        self.assertEqual(cleaned.loc[1, "high"], cleaned.loc[0, "high"])
        self.assertEqual(cleaned.loc[1, "low"], cleaned.loc[0, "low"])

    def test_resolve_close_column_supports_common_variants(self):
        self.assertEqual(_resolve_close_column(pd.DataFrame({"Close": [1.0]})), "Close")
        self.assertEqual(_resolve_close_column(pd.DataFrame({"CLOSE": [1.0]})), "CLOSE")
        self.assertEqual(_resolve_close_column(pd.DataFrame({"close": [1.0]})), "close")

    def test_resolve_close_column_raises_when_missing(self):
        with self.assertRaises(KeyError):
            _resolve_close_column(pd.DataFrame({"open": [1.0]}))

    def test_valid_batch_size_choices_prefers_even_divisors(self):
        self.assertEqual(_valid_batch_size_choices(960), [16, 32, 64])
        self.assertEqual(_valid_batch_size_choices(750), [16, 32, 64, 128, 256, 512])

    def test_narrow_hp_config_shrinks_linear_and_log_ranges_around_best(self):
        narrowed = _narrow_hp_config(
            {
                "learning_rate": {"min": 1e-5, "max": 1e-2, "log": True},
                "ent_coef": {"min": 0.01, "max": 0.30},
            },
            {"learning_rate": 1e-3, "ent_coef": 0.1},
            shrink=0.5,
        )

        self.assertGreaterEqual(narrowed["learning_rate"]["min"], 1e-5)
        self.assertLessEqual(narrowed["learning_rate"]["max"], 1e-2)
        self.assertLess(narrowed["learning_rate"]["max"] - narrowed["learning_rate"]["min"], 1e-2)
        self.assertAlmostEqual(narrowed["ent_coef"]["min"], 0.0275)
        self.assertAlmostEqual(narrowed["ent_coef"]["max"], 0.1725)

    def test_sample_stage_params_uses_trial_samplers(self):
        class _Trial:
            def __init__(self):
                self.float_args = []

            def suggest_int(self, name, low, high, log=False):
                self.int_args = (name, low, high, log)
                return 512

            def suggest_float(self, name, low, high, log=False):
                self.float_args.append((name, low, high, log))
                return {"learning_rate": 1e-4, "ent_coef": 0.1, "reward_turnover_penalty": 0.01, "reward_calm_holding_bonus": 0.002}[name]

        trial = _Trial()
        params = _sample_stage_params(
            trial,
            {
                "learning_rate": {"min": 1e-5, "max": 1e-3, "log": True},
                "n_steps": {"min": 128, "max": 1024, "log": True},
                "ent_coef": {"min": 0.01, "max": 0.2},
                "reward_turnover_penalty": {"min": 0.0, "max": 0.02},
                "reward_calm_holding_bonus": {"min": 0.0, "max": 0.01},
            },
        )

        self.assertEqual(params["n_steps"], 512)
        self.assertEqual(trial.int_args, ("n_steps", 128, 1024, True))
        self.assertAlmostEqual(params["learning_rate"], 1e-4)
        self.assertAlmostEqual(params["ent_coef"], 0.1)
        self.assertAlmostEqual(params["reward_turnover_penalty"], 0.01)
        self.assertAlmostEqual(params["reward_calm_holding_bonus"], 0.002)

    def test_summarize_trial_scores_applies_baseline_penalty(self):
        score, diagnostics = _summarize_trial_scores(
            evaluation_scores=[5.0, 3.0],
            baseline_scores=[{0: 4.8, 1: 4.6}, {0: 4.0, 1: 3.5}],
            baseline_margin=1.0,
        )

        self.assertLess(score, diagnostics["aggregate_score"])
        self.assertGreater(diagnostics["baseline_penalty"], 0.0)
        self.assertEqual(diagnostics["best_baseline_score"], 4.4)

    def test_constant_action_baselines_cover_full_action_space(self):
        dataset = pd.DataFrame({"close": [100.0], "close_norm": [0.5]})
        with mock.patch("walk_forward.evaluate_agent", return_value={"total_return_pct": 1.0, "trade_count": 1, "action_counts": {6: 1}}) as eval_mock, \
             mock.patch("walk_forward._score_tuning_trial", side_effect=lambda results, bars: (float(results["total_return_pct"]), {})):
            scores = _score_constant_action_baselines([dataset], ["close", "close_norm"])

        self.assertEqual(set(scores[0].keys()), set(range(7)))
        self.assertEqual(eval_mock.call_count, 7)

    def test_constant_action_baselines_degrade_failed_evaluations(self):
        dataset = pd.DataFrame({"close": [100.0], "close_norm": [0.5]})

        def fake_eval(model, data, verbose=0, deterministic=True):
            if model.action == 3:
                raise RuntimeError("boom")
            return {"total_return_pct": 1.0, "trade_count": 1, "action_counts": {6: 1}}

        with mock.patch("walk_forward.evaluate_agent", side_effect=fake_eval), \
             mock.patch("walk_forward._score_tuning_trial", side_effect=lambda results, bars: (float(results["total_return_pct"]), {})):
            scores = _score_constant_action_baselines([dataset], ["close", "close_norm"])

        self.assertEqual(scores[0][3], -100.0)
        self.assertEqual(scores[0][0], 1.0)

    def test_trial_config_overrides_apply_and_restore(self):
        config["reward"]["loss_multiplier"] = 0.7
        config["reward"]["turnover_penalty"] = 0.01
        config["reward"]["calm_holding_bonus"] = 0.001
        config["augmentation"]["synthetic_bears"]["oversample_ratio"] = 0.3

        originals = _apply_trial_config_overrides(
            {
                "reward_loss_multiplier": 0.9,
                "reward_turnover_penalty": 0.02,
                "reward_calm_holding_bonus": 0.003,
                "synthetic_oversample_ratio": 0.5,
            }
        )

        self.assertEqual(config["reward"]["loss_multiplier"], 0.9)
        self.assertEqual(config["reward"]["turnover_penalty"], 0.02)
        self.assertEqual(config["reward"]["calm_holding_bonus"], 0.003)
        self.assertEqual(config["augmentation"]["synthetic_bears"]["oversample_ratio"], 0.5)

        _restore_trial_config_overrides(originals)

        self.assertEqual(config["reward"]["loss_multiplier"], 0.7)
        self.assertEqual(config["reward"]["turnover_penalty"], 0.01)
        self.assertEqual(config["reward"]["calm_holding_bonus"], 0.001)
        self.assertEqual(config["augmentation"]["synthetic_bears"]["oversample_ratio"], 0.3)

    def test_resolve_tuning_n_jobs_obeys_deterministic_mode(self):
        config["reproducibility"]["deterministic_mode"] = True
        self.assertTrue(_deterministic_mode_enabled())
        self.assertEqual(_resolve_tuning_n_jobs(), 1)

    def test_resolve_tuning_n_jobs_uses_parallel_limits(self):
        config["reproducibility"]["deterministic_mode"] = False
        config["hyperparameter_tuning"]["parallel_processing"] = {
            "enabled": True,
            "n_jobs": 0,
            "reserve_cores": 2,
            "max_auto_jobs": 3,
        }
        config["hyperparameter_tuning"]["n_envs"] = 2

        with mock.patch("walk_forward.multiprocessing.cpu_count", return_value=10):
            self.assertEqual(_resolve_tuning_n_jobs(), 3)

    def test_drop_unused_model_columns_respects_risk_management_needs(self):
        raw = pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
                "position": [0],
                "VWAP": [1.0],
            }
        )
        config["risk_management"]["enabled"] = False
        config["risk_management"]["dynamic_sl_tp"]["enabled"] = False
        df = raw.copy()
        _drop_unused_model_columns([df])
        self.assertNotIn("open", df.columns)
        self.assertNotIn("high", df.columns)
        self.assertNotIn("low", df.columns)
        self.assertNotIn("volume", df.columns)

        config["risk_management"]["enabled"] = True
        df = raw.copy()
        _drop_unused_model_columns([df])
        self.assertIn("open", df.columns)
        self.assertIn("high", df.columns)
        self.assertIn("low", df.columns)
        self.assertNotIn("volume", df.columns)

    def test_prepare_tuning_inputs_runs_scaling_and_column_drop(self):
        train_df = pd.DataFrame(
            {"close": [1.0], "open": [1.0], "high": [1.0], "low": [1.0], "volume": [1.0], "position": [0]}
        )
        val_df = train_df.copy()
        config["augmentation"]["synthetic_bears"]["enabled"] = False
        config["risk_management"]["enabled"] = False
        config["risk_management"]["dynamic_sl_tp"]["enabled"] = False

        with mock.patch("walk_forward.process_technical_indicators", side_effect=lambda df: df.copy()), \
             mock.patch("walk_forward.get_standardized_column_names", return_value=["close"]), \
             mock.patch(
                 "walk_forward.scale_window",
                 return_value=(None, train_df.copy(), val_df.copy(), None),
             ):
            prepared_train, prepared_val, synth = _prepare_tuning_inputs(train_df, val_df)

        self.assertIsNone(synth)
        self.assertIn("close", prepared_train.columns)
        self.assertNotIn("open", prepared_train.columns)
        self.assertNotIn("position", prepared_train.columns)
        self.assertIn("close", prepared_val.columns)

    def test_prepare_tuning_inputs_sanitizes_price_outliers_before_scaling(self):
        train_df = pd.DataFrame(
            {
                "close": [30000.0, 30100.0],
                "open": [30000.0, 30100.0],
                "high": [30010.0, 30110.0],
                "low": [29990.0, 30090.0],
                "volume": [1.0, 1.0],
                "position": [0, 0],
            }
        )
        val_df = pd.DataFrame(
            {
                "close": [30050.0, 175.0, 30150.0],
                "open": [30050.0, 175.0, 30150.0],
                "high": [30060.0, 175.0, 30160.0],
                "low": [30040.0, 175.0, 30140.0],
                "volume": [1.0, 1.0, 1.0],
                "position": [0, 0, 0],
            }
        )
        config["augmentation"]["synthetic_bears"]["enabled"] = False

        def _fake_scale_window(train_data, val_data, test_data, **kwargs):
            self.assertGreater(float(val_data.iloc[1]["close"]), 1000.0)
            return None, train_data.copy(), val_data.copy(), test_data.copy()

        with mock.patch("walk_forward.process_technical_indicators", side_effect=lambda df: df.copy()), \
             mock.patch("walk_forward.get_standardized_column_names", return_value=[]), \
             mock.patch("walk_forward.scale_window", side_effect=_fake_scale_window):
            _, prepared_val, _ = _prepare_tuning_inputs(train_df, val_df)

        self.assertGreater(float(prepared_val.iloc[1]["close"]), 1000.0)

    def test_build_tuning_env_uses_dummy_or_subproc_based_on_count(self):
        data = pd.DataFrame({"close": [1.0], "close_norm": [0.0]})
        config["seed"] = 7
        config["training"]["n_envs"] = 4
        config["hyperparameter_tuning"]["n_envs"] = 1

        class _EnvStub:
            def __init__(self):
                self.seed_calls = []

            def seed(self, value):
                self.seed_calls.append(value)

        dummy_env = _EnvStub()
        with mock.patch("walk_forward.DummyVecEnv", return_value=dummy_env):
            env, count = _build_tuning_env(data)
        self.assertIs(env, dummy_env)
        self.assertEqual(count, 1)
        self.assertEqual(dummy_env.seed_calls, [1007])

        subproc_env = _EnvStub()
        config["hyperparameter_tuning"]["n_envs"] = 3
        with mock.patch("walk_forward.SubprocVecEnv", return_value=subproc_env):
            env, count = _build_tuning_env(data)
        self.assertIs(env, subproc_env)
        self.assertEqual(count, 3)
        self.assertEqual(subproc_env.seed_calls, [1007])

    def test_sanitize_cuda_library_path_filters_foreign_venv_and_dedupes(self):
        original = os.environ.get("LD_LIBRARY_PATH")
        try:
            os.environ["LD_LIBRARY_PATH"] = ":".join(
                [
                    "/home/orion/Documents/loki/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib",
                    "/tmp/custom",
                    "/tmp/custom",
                ]
            )
            with mock.patch("walk_forward.os.path.isdir", return_value=True):
                _sanitize_cuda_library_path()
            sanitized = os.environ["LD_LIBRARY_PATH"].split(":")
        finally:
            if original is None:
                os.environ.pop("LD_LIBRARY_PATH", None)
            else:
                os.environ["LD_LIBRARY_PATH"] = original

        self.assertNotIn(
            "/home/orion/Documents/loki/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib",
            sanitized,
        )
        self.assertEqual(sanitized.count("/tmp/custom"), 1)
        self.assertTrue(any("cudnn/lib" in path for path in sanitized))

    def test_save_json_serializes_timestamps_and_numpy_scalars(self):
        import json
        import tempfile
        import numpy as np
        from decimal import Decimal

        payload = {
            "timestamp": pd.Timestamp("2026-01-01 12:00:00"),
            "arr": np.array([1, 2]),
            "flag": np.bool_(True),
            "value": Decimal("1.5"),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "payload.json")
            save_json(payload, path)
            with open(path, "r") as fh:
                loaded = json.load(fh)

        self.assertEqual(loaded["timestamp"], "2026-01-01 12:00:00")
        self.assertEqual(loaded["arr"], [1, 2])
        self.assertTrue(loaded["flag"])
        self.assertEqual(loaded["value"], 1.5)

    def test_prediction_accuracy_eval_tracks_actions_and_portfolio(self):
        idx = pd.date_range("2026-01-01 00:00:00", periods=4, freq="1min", tz="UTC")
        test_data = pd.DataFrame(
            {
                "close_norm": [0.1, 0.2, 0.3, 0.4],
                "close": [100.0, 101.0, 102.0, 103.0],
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [100.0, 101.0, 102.0, 103.0],
                "low": [100.0, 101.0, 102.0, 103.0],
                "volume": [1.0, 1.0, 1.0, 1.0],
            },
            index=idx,
        )

        results = evaluate_agent_prediction_accuracy(
            self._SequenceModel([0, 0, 6]),
            test_data,
            verbose=0,
            deterministic=True,
        )

        self.assertGreaterEqual(results["prediction_accuracy"], 66.0)
        self.assertEqual(results["action_history"], [0, 0, 6])
        self.assertGreaterEqual(results["trade_count"], 2)
        self.assertLess(results["completed_trades"], results["trade_count"])
        self.assertGreaterEqual(results["completed_trades"], 1)
        self.assertEqual(results["hit_rate"], 100.0)
        self.assertGreater(len(results["portfolio_history"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
