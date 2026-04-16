#!/usr/bin/env python
"""Tests for the ANSI/readability logging helpers in walk_forward.py."""

import logging
import re
import sys
import os
import io
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.log_format import (  # noqa: E402
    ANSI_BOLD,
    ANSI_GREEN,
    ANSI_RED,
    ANSI_RESET,
    ACTION_NAMES,
    AnsiStrippingFormatter,
    bold,
    color_pct,
    color_value,
    format_action_distribution,
    strip_ansi,
)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class TestColorPct(unittest.TestCase):
    def test_positive_is_green(self):
        s = color_pct(12.34)
        self.assertIn(ANSI_GREEN, s)
        self.assertIn("12.34%", s)
        self.assertTrue(s.endswith(ANSI_RESET))

    def test_negative_is_red(self):
        s = color_pct(-5.0)
        self.assertIn(ANSI_RED, s)
        self.assertIn("-5.00%", s)

    def test_zero_is_green(self):
        s = color_pct(0.0)
        self.assertIn(ANSI_GREEN, s)

    def test_strip_returns_plain_number(self):
        self.assertEqual(ANSI_RE.sub("", color_pct(3.14)), "3.14%")


class TestBold(unittest.TestCase):
    def test_bold_wraps_with_reset(self):
        s = bold("hi")
        self.assertTrue(s.startswith(ANSI_BOLD))
        self.assertTrue(s.endswith(ANSI_RESET))
        self.assertIn("hi", s)


class TestActionDistribution(unittest.TestCase):
    def test_action_names_mapping(self):
        self.assertEqual(ACTION_NAMES[0], "LONG_1")
        self.assertEqual(ACTION_NAMES[5], "SHORT_5")
        self.assertEqual(ACTION_NAMES[6], "FLAT")

    def test_empty_history(self):
        self.assertEqual(format_action_distribution([]), "no actions")
        self.assertEqual(format_action_distribution(None), "no actions")

    def test_counts(self):
        history = [0, 0, 4, 6, 6, 6]
        s = format_action_distribution(history)
        self.assertIn("LONG_1=2", s)
        self.assertIn("SHORT_2=1", s)
        self.assertIn("FLAT=3", s)

    def test_numpy_array_input(self):
        import numpy as np
        history = np.array([0, 5, 5, 6])
        s = format_action_distribution(history)
        self.assertIn("LONG_1=1", s)
        self.assertIn("SHORT_5=2", s)
        self.assertIn("FLAT=1", s)

    def test_dict_input(self):
        counts = {0: 5, 5: 0, 6: 3}
        s = format_action_distribution(counts)
        self.assertIn("LONG_1=5", s)
        self.assertIn("SHORT_5=0", s)
        self.assertIn("FLAT=3", s)


class TestColorValue(unittest.TestCase):
    def test_positive_no_suffix(self):
        s = color_value(1.23)
        self.assertIn(ANSI_GREEN, s)
        self.assertIn("1.23", s)
        self.assertNotIn("%", s)

    def test_negative(self):
        s = color_value(-0.5)
        self.assertIn(ANSI_RED, s)
        self.assertIn("-0.50", s)

    def test_custom_suffix(self):
        s = color_value(2.0, suffix="x")
        self.assertIn("2.00x", s)


class TestAnsiStrippingFormatter(unittest.TestCase):
    def test_formatter_strips_ansi(self):
        fmt = AnsiStrippingFormatter("%(message)s")
        rec = logging.LogRecord(
            name="t", level=logging.INFO, pathname="", lineno=0,
            msg=f"Return={color_pct(5.0)}, Sortino={bold('1.23')}",
            args=None, exc_info=None,
        )
        out = fmt.format(rec)
        self.assertNotIn("\x1b[", out)
        self.assertIn("5.00%", out)
        self.assertIn("1.23", out)

    def test_stream_handler_preserves_ansi(self):
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        lg = logging.getLogger("test_ansi_preserve")
        lg.handlers = [handler]
        lg.setLevel(logging.INFO)
        lg.propagate = False
        lg.info(f"Return={color_pct(-2.5)}")
        self.assertIn(ANSI_RED, buf.getvalue())

    def test_strip_ansi_helper(self):
        self.assertEqual(strip_ansi(f"{ANSI_GREEN}ok{ANSI_RESET}"), "ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
