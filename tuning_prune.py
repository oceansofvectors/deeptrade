"""Shared tuning-prune helpers used by walk-forward optimization."""

from __future__ import annotations

from typing import Dict


def normalize_action_counts(action_counts: Dict | None) -> Dict[int, int]:
    """Normalize action-count mappings to integer-keyed dictionaries."""
    normalized = {i: 0 for i in range(7)}
    if not action_counts:
        return normalized
    for raw_key, raw_value in action_counts.items():
        try:
            normalized[int(raw_key)] = int(raw_value)
        except (TypeError, ValueError):
            continue
    return normalized


def tuning_prune_diagnostics(results: Dict) -> Dict[str, float | int | bool]:
    """Extract simple collapse diagnostics used for early trial pruning."""
    action_counts = normalize_action_counts(results.get("action_counts"))
    total_actions = max(1, sum(action_counts.values()))
    flat_action_pct = 100.0 * action_counts.get(6, 0) / total_actions
    active_action_count = sum(1 for count in action_counts.values() if count > 0)
    trade_count = int(results.get("trade_count", results.get("num_trades", 0)))
    max_drawdown_pct = abs(float(results.get("max_drawdown", 0.0)))
    return {
        "trade_count": trade_count,
        "flat_action_pct": flat_action_pct,
        "active_action_count": active_action_count,
        "max_drawdown_pct": max_drawdown_pct,
    }


def should_hard_prune_trial(results: Dict, tuning_config: Dict) -> tuple[bool, list[str]]:
    """Return whether a trial should be pruned immediately for degenerate behavior."""
    diag = tuning_prune_diagnostics(results)
    reasons = []
    if tuning_config.get("early_prune_zero_trade", True) and diag["trade_count"] == 0:
        reasons.append("zero_trade")
    if diag["flat_action_pct"] >= float(tuning_config.get("early_prune_flat_action_pct", 99.0)):
        reasons.append("all_flat")
    if tuning_config.get("early_prune_single_action", True) and diag["active_action_count"] <= 1:
        reasons.append("single_action")
    max_allowed_drawdown_pct = float(tuning_config.get("early_prune_max_drawdown_pct", 40.0))
    if diag["max_drawdown_pct"] > max_allowed_drawdown_pct:
        reasons.append("excessive_drawdown")
    return bool(reasons), reasons
