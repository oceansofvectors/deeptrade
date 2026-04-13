#!/usr/bin/env python
"""Regression tests for vectorized env selection in training."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import train  # noqa: E402


class TrainVecEnvTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [2.0, 3.0, 4.0],
                "low": [0.5, 1.5, 2.5],
                "close": [1.5, 2.5, 3.5],
                "close_norm": [0.1, 0.5, 0.9],
                "volume": [100, 120, 140],
            }
        )

    def test_build_vectorized_env_uses_dummy_inside_worker_process(self):
        dummy_env = object()

        with mock.patch.object(train.mp, "current_process", return_value=SimpleNamespace(name="ForkProcess-1")), \
             mock.patch.object(train, "DummyVecEnv", return_value=dummy_env) as dummy_cls, \
             mock.patch.object(train, "SubprocVecEnv") as subproc_cls:
            result = train._build_vectorized_env(self.frame, {}, 4, window_label="[W1] ")

        self.assertIs(result, dummy_env)
        self.assertTrue(dummy_cls.called)
        subproc_cls.assert_not_called()

    def test_build_vectorized_env_falls_back_when_subproc_startup_fails(self):
        dummy_env = object()

        with mock.patch.object(train.mp, "current_process", return_value=SimpleNamespace(name="MainProcess")), \
             mock.patch.object(train, "SubprocVecEnv", side_effect=RuntimeError("boom")) as subproc_cls, \
             mock.patch.object(train, "DummyVecEnv", return_value=dummy_env) as dummy_cls:
            result = train._build_vectorized_env(self.frame, {}, 4)

        self.assertIs(result, dummy_env)
        self.assertTrue(subproc_cls.called)
        self.assertTrue(dummy_cls.called)


if __name__ == "__main__":
    unittest.main()
