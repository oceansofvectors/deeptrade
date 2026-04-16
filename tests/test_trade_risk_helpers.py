#!/usr/bin/env python
"""Additional tests for trade.RiskManager helpers and persistence."""

import os
import sys
import tempfile
import unittest
from datetime import datetime
from decimal import Decimal

import pandas as pd
import pytz

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trade import RiskManager, plot_results, save_trade_history  # noqa: E402


class TestTradeRiskHelpers(unittest.TestCase):
    def setUp(self):
        self.eastern = pytz.timezone("US/Eastern")

    def test_calculate_position_size_floors_to_whole_contracts(self):
        rm = RiskManager(initial_balance=100000.0, transaction_cost=0.0)
        rm.position_size = Decimal("0.05")

        self.assertEqual(rm.calculate_position_size(100.0), 500)
        self.assertEqual(rm.calculate_position_size(50000.0), 1)

    def test_check_daily_risk_limit_resets_by_trading_day_and_trips(self):
        rm = RiskManager(initial_balance=10000.0, transaction_cost=0.0, daily_risk_limit=100.0)
        ts_day1 = pd.Timestamp(datetime(2024, 1, 2, 10, 0, 0), tz=self.eastern)
        ts_day2 = pd.Timestamp(datetime(2024, 1, 3, 10, 0, 0), tz=self.eastern)

        exceeded, reason = rm.check_daily_risk_limit(ts_day1)
        self.assertFalse(exceeded)
        self.assertEqual(reason, "")

        rm.net_worth = Decimal("9875.0")
        exceeded, reason = rm.check_daily_risk_limit(ts_day1)
        self.assertTrue(exceeded)
        self.assertEqual(reason, "daily_risk_limit")

        rm.net_worth = Decimal("10000.0")
        exceeded, reason = rm.check_daily_risk_limit(ts_day2)
        self.assertFalse(exceeded)
        self.assertEqual(reason, "")
        self.assertEqual(rm.daily_start_balance, Decimal("10000.0"))

    def test_percentage_take_profit_exit_uses_high_low_path(self):
        rm = RiskManager(
            initial_balance=10000.0,
            transaction_cost=0.0,
            take_profit_pct=1.0,
            take_profit_mode="percentage",
        )
        ts = pd.Timestamp(datetime(2024, 1, 2, 10, 0, 0), tz=self.eastern)
        rm.enter_position(1, 100.0, ts, contracts=100)

        triggered, reason = rm.check_exits(close_price=100.2, high_price=111.0, low_price=99.5)

        self.assertTrue(triggered)
        self.assertEqual(reason, "take_profit")

    def test_atr_stop_loss_and_take_profit_use_precomputed_prices(self):
        ts = pd.Timestamp(datetime(2024, 1, 2, 10, 0, 0), tz=self.eastern)

        long_rm = RiskManager(
            initial_balance=10000.0,
            transaction_cost=0.0,
            stop_loss_mode="atr",
            stop_loss_atr_multiplier=2.0,
        )
        long_rm.enter_position(1, 100.0, ts, contracts=10, atr_value=2.5)
        self.assertEqual(long_rm.stop_loss_price, Decimal("95.0"))
        triggered, reason = long_rm.check_exits(close_price=99.0, high_price=100.0, low_price=94.5)
        self.assertTrue(triggered)
        self.assertEqual(reason, "stop_loss")

        short_rm = RiskManager(
            initial_balance=10000.0,
            transaction_cost=0.0,
            take_profit_mode="atr",
            take_profit_atr_multiplier=3.0,
        )
        short_rm.enter_position(-1, 100.0, ts, contracts=10, atr_value=2.0)
        self.assertEqual(short_rm.take_profit_price, Decimal("94.0"))
        triggered, reason = short_rm.check_exits(close_price=97.0, high_price=101.0, low_price=93.5)
        self.assertTrue(triggered)
        self.assertEqual(reason, "take_profit")

    def test_trailing_stop_triggers_after_profit_retraces(self):
        rm = RiskManager(
            initial_balance=10000.0,
            transaction_cost=0.0,
            trailing_stop_pct=1.0,
        )
        ts = pd.Timestamp(datetime(2024, 1, 2, 10, 0, 0), tz=self.eastern)
        rm.enter_position(1, 100.0, ts, contracts=100)

        rm.update_stops(close_price=120.0, high_price=120.0, low_price=100.0)
        triggered, reason = rm.check_exits(close_price=109.0, high_price=109.0, low_price=108.0)

        self.assertTrue(triggered)
        self.assertEqual(reason, "trailing_stop")

    def test_exit_position_books_profit_and_resets_state(self):
        rm = RiskManager(initial_balance=10000.0, transaction_cost=0.0)
        ts = pd.Timestamp(datetime(2024, 1, 2, 10, 0, 0), tz=self.eastern)
        rm.enter_position(1, 100.0, ts, contracts=100)

        rm.exit_position(110.0, ts, "test_exit")

        self.assertEqual(rm.position, 0)
        self.assertEqual(rm.current_contracts, 0)
        self.assertEqual(rm.entry_price, Decimal("0.0"))
        self.assertEqual(rm.net_worth, Decimal("10100.0"))
        self.assertEqual(rm.trade_history[-1]["exit_reason"], "test_exit")

    def test_save_trade_history_writes_csv(self):
        trades = [
            {"date": "2024-01-02 10:00:00", "action": "buy", "price": 100.0},
            {"date": "2024-01-02 10:05:00", "action": "sell", "price": 101.0, "profit": 10.0},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trade_history.csv")
            save_trade_history(trades, path)
            self.assertTrue(os.path.exists(path))
            loaded = pd.read_csv(path)
            self.assertEqual(len(loaded), 2)
            self.assertIn("action", loaded.columns)

    def test_save_trade_history_handles_trade_type_format_and_empty_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = os.path.join(tmpdir, "empty.csv")
            save_trade_history([], empty_path)
            self.assertTrue(os.path.exists(empty_path))

            typed_path = os.path.join(tmpdir, "typed.csv")
            save_trade_history(
                [
                    {"date": "2024-01-02 10:00:00", "trade_type": "buy", "price": 100.0},
                    {"date": "2024-01-02 10:05:00", "trade_type": "sell", "price": 101.0, "exit_reason": "end_of_day"},
                ],
                typed_path,
            )
            loaded = pd.read_csv(typed_path)
            self.assertIn("position_from", loaded.columns)
            self.assertIn("position_to", loaded.columns)
            self.assertEqual(int(loaded.iloc[0]["position_to"]), 1)
            self.assertEqual(int(loaded.iloc[1]["position_to"]), 0)

    def test_plot_results_writes_html_and_separates_eod_markers(self):
        results = {
            "dates": ["2024-01-02 10:00:00", "2024-01-02 10:05:00"],
            "price_history": [100.0, 101.0],
            "portfolio_history": [10000.0, 10010.0],
            "trade_history": [
                {"date": "2024-01-02 10:00:00", "action": "buy", "price": 100.0},
                {"date": "2024-01-02 10:05:00", "action": "sell", "price": 101.0, "exit_reason": "end_of_day"},
            ],
            "buy_dates": ["2024-01-02 10:00:00"],
            "buy_prices": [100.0],
            "sell_dates": ["2024-01-02 10:05:00"],
            "sell_prices": [101.0],
            "final_portfolio_value": 10010.0,
            "total_return_pct": 0.1,
            "exit_reasons": {"end_of_day": 1},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with unittest.mock.patch("trade.make_subplots") as subplots_mock, \
                 unittest.mock.patch("trade.go.Scatter", side_effect=lambda **kwargs: kwargs):
                fig = unittest.mock.MagicMock()
                subplots_mock.return_value = fig
                plot_results(results, plots_dir=tmpdir)

            fig.write_html.assert_called_once()
            written_path = fig.write_html.call_args[0][0]
            self.assertTrue(written_path.startswith(tmpdir))
            self.assertTrue(written_path.endswith(".html"))
            fig.show.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
