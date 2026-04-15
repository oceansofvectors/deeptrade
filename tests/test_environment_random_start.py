#!/usr/bin/env python
"""Tests for environment random-start safety guards."""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment import TradingEnv  # noqa: E402


class TestEnvironmentRandomStart(unittest.TestCase):
    def test_random_start_respects_min_episode_steps(self):
        rows = 100
        df = pd.DataFrame(
            {
                "close_norm": np.linspace(0.1, 0.9, rows),
                "close": np.linspace(95000.0, 96000.0, rows),
                "open": np.linspace(94999.0, 95999.0, rows),
                "high": np.linspace(95001.0, 96001.0, rows),
                "low": np.linspace(94998.0, 95998.0, rows),
                "volume": np.linspace(10.0, 20.0, rows),
            }
        )
        env = TradingEnv(
            df,
            initial_balance=100000.0,
            transaction_cost=0.0,
            position_size=1,
            random_start_pct=1.0,
            min_episode_steps=25,
        )

        max_seen_step = 0
        for seed in range(20):
            env.reset(seed=seed)
            max_seen_step = max(max_seen_step, env.current_step)

        self.assertLessEqual(max_seen_step, env.total_steps - 25)


if __name__ == "__main__":
    unittest.main(verbosity=2)
