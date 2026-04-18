"""Shared ANSI color / readability helpers for training and walk-forward logs."""

import logging
import re

import numpy as np

from action_space import ACTION_COUNT, action_label

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

ACTION_NAMES = {idx: action_label(idx) for idx in range(ACTION_COUNT)}


def color_value(value: float, fmt: str = ".2f", suffix: str = "") -> str:
    """Format a number, green if >=0, red if <0."""
    color = ANSI_GREEN if value >= 0 else ANSI_RED
    return f"{color}{value:{fmt}}{suffix}{ANSI_RESET}"


def color_pct(value: float, fmt: str = ".2f") -> str:
    """Format as percentage, green if >=0, red if <0."""
    return color_value(value, fmt=fmt, suffix="%")


def bold(text: str) -> str:
    return f"{ANSI_BOLD}{text}{ANSI_RESET}"


def format_action_distribution(action_history) -> str:
    """Format an action history iterable as 'LONG_1=N ... FLAT=N'."""
    if action_history is None:
        return "no actions"
    if isinstance(action_history, dict):
        parts = []
        for code, name in ACTION_NAMES.items():
            parts.append(f"{name}={int(action_history.get(code, 0))}")
        return " ".join(parts)
    arr = np.asarray(action_history)
    if arr.size == 0:
        return "no actions"
    arr = arr.astype(int)
    parts = []
    for code, name in ACTION_NAMES.items():
        parts.append(f"{name}={int((arr == code).sum())}")
    return " ".join(parts)


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class AnsiStrippingFormatter(logging.Formatter):
    """Formatter that strips ANSI color codes. Use for file handlers."""

    def format(self, record):
        return strip_ansi(super().format(record))
