#!/usr/bin/env python
"""Tests for walk-forward HTML report generation."""

import json
import os
import shutil
import tempfile
import unittest

from reporting.walk_forward_report import generate_walk_forward_report


class TestWalkForwardReport(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="wf_report_test_")
        self.session_dir = os.path.join(self.tmpdir, "models", "session_test")
        os.makedirs(os.path.join(self.session_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "models", "window_1"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_json(self, rel_path, payload):
        path = os.path.join(self.session_dir, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def test_generate_html_report(self):
        self._write_json("reports/session_parameters.json", {
            "timestamp": "20260414_120000",
            "data_start": "2021-01-01 14:30:00",
            "data_end": "2021-03-31 20:00:00",
            "evaluation_metric": "sortino",
        })
        self._write_json("reports/summary_results.json", {
            "avg_return": 3.2,
            "avg_sortino": 1.8,
            "avg_portfolio": 103200.0,
            "avg_trades": 214.0,
            "num_windows": 1,
        })
        self._write_json("reports/best_hyperparameters.json", {
            "learning_rate": 0.0002,
            "reward_calm_holding_bonus": 0.0012,
        })
        self._write_json("models/window_1/report_data.json", {
            "window": 1,
            "metrics": {
                "return_pct": 3.2,
                "final_portfolio_value": 103200.0,
                "trade_count": 214,
                "hit_rate": 51.4,
                "max_drawdown": -6.8,
                "calmar_ratio": 0.47,
                "sortino_ratio": 1.8,
            },
            "window_periods": {
                "train_start": "2021-01-01 14:30:00",
                "test_end": "2021-03-31 20:00:00",
            },
            "series": {
                "timestamps": ["2021-03-01 14:30:00", "2021-03-01 14:35:00"],
                "prices": [13000.0, 13010.0],
                "portfolio_timestamps": ["2021-03-01 14:30:00", "2021-03-01 14:35:00"],
                "portfolio_values": [100000.0, 103200.0],
                "drawdown_timestamps": ["2021-03-01 14:30:00", "2021-03-01 14:35:00"],
                "drawdown_values": [0.0, -1.2],
            },
            "trade_history": [
                {"date": "2021-03-01 14:35:00+00:00", "trade_type": "Long", "price": 13010.0}
            ],
            "action_counts": {"0": 10, "1": 8, "2": 2},
            "training_iterations": [
                {"iteration": 0, "return_pct": 3.2, "sortino_ratio": 1.8, "trade_count": 214, "is_best": True, "collapse_flags": []},
                {"iteration": 1, "return_pct": -2.0, "sortino_ratio": -0.5, "trade_count": 300, "warning_metric_drop": True, "collapse_flags": ["metric_drop"]},
            ],
            "validation_results": {
                "final_portfolio_value": 104500.0,
                "total_return_pct": 4.5,
                "sortino_ratio": 3.1,
            },
            "best_iteration": 0,
        })

        output_path = generate_walk_forward_report(self.session_dir)

        self.assertTrue(os.path.isfile(output_path))
        with open(output_path, "r") as f:
            html_doc = f.read()
        self.assertIn("Walk-Forward Performance Report", html_doc)
        self.assertIn("Window 1", html_doc)
        self.assertIn("Cross-Window Diagnostics", html_doc)
        self.assertIn("reward_calm_holding_bonus", html_doc)
        self.assertIn("Plotly.newPlot", html_doc)

    def test_missing_window_payloads_raises(self):
        self._write_json("reports/session_parameters.json", {"timestamp": "20260414_120000"})
        self._write_json("reports/summary_results.json", {"num_windows": 0})
        with self.assertRaises(FileNotFoundError):
            generate_walk_forward_report(self.session_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
