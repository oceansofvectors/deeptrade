#!/usr/bin/env python
"""Tests for ANSI/color log formatting helpers."""

import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.log_format import (  # noqa: E402
    ANSI_BOLD,
    ANSI_GREEN,
    ANSI_RED,
    ANSI_RESET,
    AnsiStrippingFormatter,
    bold,
    color_pct,
    color_value,
    format_action_distribution,
    strip_ansi,
)


class TestLogFormatHelpers(unittest.TestCase):
    def test_color_helpers_and_bold_wrap_values(self):
        self.assertEqual(color_value(1.5), f"{ANSI_GREEN}1.50{ANSI_RESET}")
        self.assertEqual(color_value(-1.5), f"{ANSI_RED}-1.50{ANSI_RESET}")
        self.assertEqual(color_pct(2.0), f"{ANSI_GREEN}2.00%{ANSI_RESET}")
        self.assertEqual(bold("x"), f"{ANSI_BOLD}x{ANSI_RESET}")

    def test_action_distribution_formats_none_dict_and_array(self):
        self.assertEqual(format_action_distribution(None), "no actions")
        self.assertEqual(format_action_distribution([]), "no actions")
        self.assertEqual(
            format_action_distribution({0: 2, 5: 1, 6: 3}),
            "LONG_1=2 LONG_2=0 LONG_5=0 SHORT_1=0 SHORT_2=0 SHORT_5=1 FLAT=3",
        )
        self.assertEqual(
            format_action_distribution([0, 0, 5, 6]),
            "LONG_1=2 LONG_2=0 LONG_5=0 SHORT_1=0 SHORT_2=0 SHORT_5=1 FLAT=1",
        )

    def test_strip_ansi_and_formatter_remove_color_codes(self):
        colored = f"{ANSI_GREEN}profit{ANSI_RESET}"
        self.assertEqual(strip_ansi(colored), "profit")

        formatter = AnsiStrippingFormatter("%(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=colored,
            args=(),
            exc_info=None,
        )
        self.assertEqual(formatter.format(record), "profit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
