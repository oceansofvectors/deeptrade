"""Shared action-space helpers for target-allocation trading policies."""

from __future__ import annotations

from typing import Optional


TARGET_ALLOCATIONS = (
    0.01,
    0.02,
    0.05,
    -0.01,
    -0.02,
    -0.05,
    0.0,
)

ACTION_LABELS = {
    0: "LONG_1",
    1: "LONG_2",
    2: "LONG_5",
    3: "SHORT_1",
    4: "SHORT_2",
    5: "SHORT_5",
    6: "FLAT",
}


def target_allocation_for_action(action: int) -> float:
    """Return signed target allocation percentage for an action id."""
    return TARGET_ALLOCATIONS[int(action)]


def action_label(action: int) -> str:
    """Return a compact human-readable label for an action id."""
    return ACTION_LABELS.get(int(action), f"UNKNOWN_{int(action)}")


def action_direction(action: int) -> int:
    """Return the sign of the target allocation: 1, -1, or 0."""
    target = target_allocation_for_action(action)
    if target > 0:
        return 1
    if target < 0:
        return -1
    return 0


def action_size_pct(action: int) -> float:
    """Return absolute target allocation size as a portfolio fraction."""
    return abs(target_allocation_for_action(action))


def action_from_target_allocation(target_allocation: float) -> Optional[int]:
    """Find the action id matching the supplied signed allocation, if any."""
    rounded = round(float(target_allocation), 4)
    for idx, allocation in enumerate(TARGET_ALLOCATIONS):
        if round(allocation, 4) == rounded:
            return idx
    return None
