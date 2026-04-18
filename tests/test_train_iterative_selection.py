#!/usr/bin/env python
"""Tests for train_agent_iteratively checkpoint selection behavior."""

import copy
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config  # noqa: E402
from train import train_agent_iteratively  # noqa: E402


class _FakeEnv:
    def __init__(self, *args, **kwargs):
        self.observation_space = SimpleNamespace(shape=(4,))
        self.technical_indicators = []
        self.net_worth = 100000.0

    def reset(self, seed=None):
        return [0.0, 0.0, 0.0, 0.0], {}


class _FakeVecEnv:
    def __init__(self, env_fns):
        self.env_fns = env_fns
        self.seed_value = None

    def seed(self, seed):
        self.seed_value = seed


class _FakeModel:
    saved_paths = []

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def learn(self, total_timesteps, progress_bar=False, callback=None):
        return self

    def save(self, path):
        self.__class__.saved_paths.append(path)

    @classmethod
    def load(cls, path, env=None, device=None):
        inst = cls()
        inst.loaded_path = path
        inst.loaded_env = env
        inst.loaded_device = device
        return inst


class TestTrainIterativeSelection(unittest.TestCase):
    def setUp(self):
        self._config_backup = copy.deepcopy(config)
        _FakeModel.saved_paths = []
        config["training"]["n_envs"] = 1
        config["training"]["verbose"] = 0
        config["training"]["random_start_pct"] = 0.0
        config["model"]["algorithm"] = "ppo"
        config["model"]["ent_coef_decay"] = False
        config["model"]["use_lr_decay"] = False
        config["sequence_model"]["enabled"] = False
        config["environment"]["initial_balance"] = 100000.0

    def tearDown(self):
        config.clear()
        config.update(self._config_backup)

    def _patch_train_stack(self):
        return mock.patch.multiple(
            "train",
            TradingEnv=_FakeEnv,
            DummyVecEnv=_FakeVecEnv,
            PPO=_FakeModel,
            check_env=mock.DEFAULT,
            get_device=mock.DEFAULT,
        )

    def test_iterative_training_selects_improved_checkpoint(self):
        train_df = pd.DataFrame({"close": [1.0]})
        val_df = pd.DataFrame({"close": [1.0] * 1000})
        eval_results = [
            {
                "total_return_pct": 1.0,
                "final_portfolio_value": 101000.0,
                "trade_count": 30,
                "action_counts": {0: 15, 3: 10, 6: 5},
                "max_drawdown": -5.0,
            },
            {
                "total_return_pct": 2.5,
                "final_portfolio_value": 102500.0,
                "trade_count": 35,
                "action_counts": {0: 12, 3: 12, 6: 11},
                "max_drawdown": -6.0,
            },
        ]

        with self._patch_train_stack() as patched, \
             mock.patch("train.evaluate_agent", side_effect=eval_results):
            patched["check_env"].return_value = None
            patched["get_device"].return_value = "cpu"
            best_model, best_results, all_results = train_agent_iteratively(
                train_df,
                val_df,
                initial_timesteps=1,
                max_iterations=1,
                additional_timesteps=1,
                n_stagnant_loops=1,
                evaluation_metric="return",
                window_folder=None,
            )

        self.assertFalse(best_results["selected_via_fallback"])
        self.assertEqual(best_results["total_return_pct"], 2.5)
        self.assertEqual(len(all_results), 2)
        self.assertTrue(best_model.loaded_path.endswith("best_model"))

    def test_iterative_training_falls_back_when_all_candidates_fail_gates(self):
        train_df = pd.DataFrame({"close": [1.0]})
        val_df = pd.DataFrame({"close": [1.0] * 1000})
        eval_results = [
            {
                "total_return_pct": 1.0,
                "final_portfolio_value": 101000.0,
                "trade_count": 0,
                "action_counts": {6: 100},
                "max_drawdown": -5.0,
            },
            {
                "total_return_pct": 5.0,
                "final_portfolio_value": 105000.0,
                "trade_count": 0,
                "action_counts": {6: 100},
                "max_drawdown": -5.0,
            },
        ]

        with self._patch_train_stack() as patched, \
             mock.patch("train.evaluate_agent", side_effect=eval_results):
            patched["check_env"].return_value = None
            patched["get_device"].return_value = "cpu"
            best_model, best_results, all_results = train_agent_iteratively(
                train_df,
                val_df,
                initial_timesteps=1,
                max_iterations=1,
                additional_timesteps=1,
                n_stagnant_loops=1,
                evaluation_metric="return",
                window_folder=None,
            )

        self.assertTrue(best_results["selected_via_fallback"])
        self.assertIn("fallback_score", best_results)
        self.assertEqual(len(all_results), 2)
        self.assertTrue(best_model.loaded_path.endswith("best_model_fallback"))

    def test_iterative_training_ignores_tiny_metric_gains_below_threshold(self):
        train_df = pd.DataFrame({"close": [1.0]})
        val_df = pd.DataFrame({"close": [1.0] * 1000})
        eval_results = [
            {
                "total_return_pct": 1.0,
                "final_portfolio_value": 101000.0,
                "trade_count": 30,
                "action_counts": {0: 15, 3: 10, 6: 5},
                "max_drawdown": -5.0,
            },
            {
                "total_return_pct": 1.005,
                "final_portfolio_value": 101005.0,
                "trade_count": 32,
                "action_counts": {0: 12, 3: 12, 6: 8},
                "max_drawdown": -5.0,
            },
        ]

        with self._patch_train_stack() as patched, \
             mock.patch("train.evaluate_agent", side_effect=eval_results):
            patched["check_env"].return_value = None
            patched["get_device"].return_value = "cpu"
            best_model, best_results, all_results = train_agent_iteratively(
                train_df,
                val_df,
                initial_timesteps=1,
                max_iterations=1,
                additional_timesteps=1,
                n_stagnant_loops=1,
                evaluation_metric="return",
                improvement_threshold=0.01,
                window_folder=None,
            )

        self.assertFalse(best_results["selected_via_fallback"])
        self.assertEqual(best_results["total_return_pct"], 1.0)
        self.assertEqual(len(all_results), 2)
        self.assertTrue(best_model.loaded_path.endswith("best_model"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
