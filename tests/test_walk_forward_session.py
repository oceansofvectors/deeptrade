#!/usr/bin/env python
"""Tests for walk-forward session/day handling."""

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from walk_forward import get_trading_days  # noqa: E402


class TestWalkForwardSessionDays(unittest.TestCase):
    def test_get_trading_days_uses_configured_cross_midnight_session(self):
        idx = pd.to_datetime(
            [
                "2026-01-05 22:10:00+00:00",  # 16:10 Chicago, still prior session
                "2026-01-05 23:10:00+00:00",  # 17:10 Chicago, new session
                "2026-01-06 15:00:00+00:00",  # 09:00 Chicago, same session as 17:10
            ]
        )
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)

        trading_days = get_trading_days(df)

        self.assertEqual(trading_days, ["2026-01-04", "2026-01-05"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
