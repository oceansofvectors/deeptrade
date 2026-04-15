#!/usr/bin/env python
"""Tests for deterministic seeding helpers and tuning job resolution."""

import os
import unittest
from unittest import mock

from utils import seeding
from walk_forward import _deterministic_mode_enabled, _resolve_tuning_n_jobs


class TestSeedingHelpers(unittest.TestCase):
    def test_enable_full_determinism_sets_env(self):
        seeding.enable_full_determinism(123)
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), "123")
        self.assertEqual(os.environ.get("CUBLAS_WORKSPACE_CONFIG"), ":16:8")

    def test_seed_worker_uses_process_identity(self):
        with mock.patch("utils.seeding.set_global_seed") as mock_set_seed:
            fake_process = mock.Mock()
            fake_process._identity = (3,)
            with mock.patch("utils.seeding.multiprocessing.current_process", return_value=fake_process):
                seeding.seed_worker(10)
        expected_seed = seeding.seed_value + 10 + 3
        mock_set_seed.assert_called_once_with(expected_seed)


class TestDeterministicMode(unittest.TestCase):
    def test_deterministic_mode_enabled_from_config(self):
        self.assertTrue(_deterministic_mode_enabled())

    def test_resolve_tuning_n_jobs_forces_one_in_deterministic_mode(self):
        self.assertEqual(_resolve_tuning_n_jobs(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
