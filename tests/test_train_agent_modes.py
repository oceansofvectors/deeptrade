#!/usr/bin/env python
"""Tests for train.train_agent mode selection and scheduler wiring."""

import copy
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import config  # noqa: E402
from train import EntropyDecayCallback, train_agent  # noqa: E402


class _FakeEnv:
    last_kwargs = None

    def __init__(self, *args, **kwargs):
        self.__class__.last_kwargs = kwargs
        self.observation_space = SimpleNamespace(shape=(4,))
        self.technical_indicators = []
        self.seed_value = None

    def reset(self, seed=None):
        return [0.0, 0.0, 0.0, 0.0], {}


class _FakeVecEnv:
    def __init__(self, env_fns):
        self.env_fns = env_fns
        self.created_envs = [fn() for fn in env_fns]
        self.seed_value = None

    def seed(self, seed):
        self.seed_value = seed


class _FakeModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.learn_calls = []

    def learn(self, total_timesteps, progress_bar=False, callback=None):
        self.learn_calls.append((total_timesteps, progress_bar, callback))
        return self


class TestTrainAgentModes(unittest.TestCase):
    def setUp(self):
        self._config_backup = copy.deepcopy(config)
        _FakeEnv.last_kwargs = None
        config["seed"] = 42
        config["training"]["n_envs"] = 1
        config["training"]["random_start_pct"] = 0.2
        config["model"]["learning_rate"] = 0.0003
        config["model"]["ent_coef"] = 0.01
        config["model"]["use_lr_decay"] = False
        config["model"]["ent_coef_decay"] = False
        config["model"]["n_steps"] = 128
        config["sequence_model"]["enabled"] = False
        config["sequence_model"]["device"] = "cpu"

    def tearDown(self):
        config.clear()
        config.update(self._config_backup)

    def _patch_stack(self):
        return mock.patch.multiple(
            "train",
            TradingEnv=_FakeEnv,
            DummyVecEnv=_FakeVecEnv,
            SubprocVecEnv=_FakeVecEnv,
            PPO=_FakeModel,
            RecurrentPPO=_FakeModel,
            check_env=mock.DEFAULT,
            get_device=mock.DEFAULT,
        )

    def test_train_agent_uses_subproc_and_decay_schedules(self):
        config["training"]["n_envs"] = 2
        config["model"]["use_lr_decay"] = True
        config["model"]["final_learning_rate"] = 1e-5
        config["model"]["ent_coef_decay"] = True
        config["model"]["final_ent_coef"] = 0.001

        train_df = pd.DataFrame({"close": [1.0]})

        with self._patch_stack() as patched, \
             mock.patch("stable_baselines3.common.utils.LinearSchedule", side_effect=lambda start, end, frac: ("schedule", start, end, frac)):
            patched["check_env"].return_value = None
            patched["get_device"].return_value = "cpu"
            model = train_agent(train_df, total_timesteps=10)

        self.assertEqual(model.kwargs["learning_rate"], ("schedule", 0.0003, 1e-05, 1.0))
        self.assertEqual(model.kwargs["ent_coef"], ("schedule", 0.01, 0.001, 1.0))
        self.assertEqual(model.learn_calls[0][0], 10)

    def test_train_agent_recurrent_uses_entropy_callback(self):
        config["sequence_model"]["enabled"] = True
        config["sequence_model"]["lstm_hidden_size"] = 64
        config["sequence_model"]["n_lstm_layers"] = 2
        config["model"]["ent_coef_decay"] = True
        config["model"]["final_ent_coef"] = 0.001
        config["model"]["n_steps"] = 128

        train_df = pd.DataFrame({"close": [1.0]})

        with self._patch_stack() as patched, \
             mock.patch("train.RECURRENT_PPO_AVAILABLE", True):
            patched["check_env"].return_value = None
            patched["get_device"].return_value = "cpu"
            model = train_agent(train_df, total_timesteps=20)

        callback = model.learn_calls[0][2][0]
        self.assertIsInstance(callback, EntropyDecayCallback)
        self.assertEqual(model.kwargs["policy_kwargs"]["lstm_hidden_size"], 64)
        self.assertEqual(model.kwargs["policy_kwargs"]["n_lstm_layers"], 2)
        self.assertEqual(model.kwargs["n_steps"], 128)
        self.assertEqual(model.kwargs["batch_size"], 128)
        self.assertEqual(_FakeEnv.last_kwargs["min_episode_steps"], 128)

    def test_train_agent_warns_and_falls_back_when_recurrent_unavailable(self):
        config["sequence_model"]["enabled"] = True
        train_df = pd.DataFrame({"close": [1.0]})

        with self._patch_stack() as patched, \
             mock.patch("train.RECURRENT_PPO_AVAILABLE", False), \
             mock.patch("train.logger.warning") as warn_mock:
            patched["check_env"].return_value = None
            patched["get_device"].return_value = "cpu"
            model = train_agent(train_df, total_timesteps=5)

        self.assertIsInstance(model, _FakeModel)
        warn_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
