#!/usr/bin/env python
"""Tests for shared action-space helper semantics."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from action_space import (  # noqa: E402
    action_direction,
    action_from_target_allocation,
    action_label,
    action_size_pct,
    target_allocation_for_action,
)


class TestActionSpaceHelpers(unittest.TestCase):
    def test_target_allocations_match_expected_mapping(self):
        expected = [0.01, 0.02, 0.05, -0.01, -0.02, -0.05, 0.0]
        self.assertEqual([target_allocation_for_action(i) for i in range(7)], expected)

    def test_labels_and_directions_match_action_semantics(self):
        self.assertEqual(action_label(0), "LONG_1")
        self.assertEqual(action_label(5), "SHORT_5")
        self.assertEqual(action_label(6), "FLAT")
        self.assertEqual(action_direction(1), 1)
        self.assertEqual(action_direction(4), -1)
        self.assertEqual(action_direction(6), 0)

    def test_size_and_reverse_lookup(self):
        self.assertAlmostEqual(action_size_pct(2), 0.05)
        self.assertAlmostEqual(action_size_pct(4), 0.02)
        self.assertEqual(action_from_target_allocation(0.05), 2)
        self.assertEqual(action_from_target_allocation(-0.02), 4)
        self.assertEqual(action_from_target_allocation(0.0), 6)
        self.assertIsNone(action_from_target_allocation(0.03))

    def test_unknown_label_and_rounded_reverse_lookup(self):
        self.assertEqual(action_label(99), "UNKNOWN_99")
        self.assertEqual(action_from_target_allocation(0.01001), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
