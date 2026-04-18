"""Shared action-space helpers for discrete trading policies."""

from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR
from typing import Optional

import constants
import money
from config import config

DEFAULT_ACTION_MODE = "atr_risk"
VALID_ACTION_MODES = {"fixed_contracts", "atr_risk"}
DEFAULT_TARGET_CONTRACTS = (
    1,
    2,
    3,
    -1,
    -2,
    -3,
    0,
)
DEFAULT_RISK_BUCKETS = (
    0.5,
    1.0,
    1.5,
    -0.5,
    -1.0,
    -1.5,
    0.0,
)


def _load_action_mode() -> str:
    configured = str(config.get("action_space", {}).get("mode", DEFAULT_ACTION_MODE)).lower()
    if configured not in VALID_ACTION_MODES:
        return DEFAULT_ACTION_MODE
    return configured


def _load_target_contracts() -> tuple[int, ...]:
    configured = config.get("action_space", {}).get("target_contracts", DEFAULT_TARGET_CONTRACTS)
    try:
        contracts = tuple(int(value) for value in configured)
    except (TypeError, ValueError):
        contracts = DEFAULT_TARGET_CONTRACTS
    if len(contracts) != 7:
        return DEFAULT_TARGET_CONTRACTS
    return contracts


def _load_risk_buckets() -> tuple[float, ...]:
    configured = config.get("action_space", {}).get("risk_buckets", DEFAULT_RISK_BUCKETS)
    try:
        buckets = tuple(float(value) for value in configured)
    except (TypeError, ValueError):
        buckets = DEFAULT_RISK_BUCKETS
    if len(buckets) != 7:
        return DEFAULT_RISK_BUCKETS
    if sum(1 for value in buckets if value == 0.0) != 1:
        return DEFAULT_RISK_BUCKETS
    return buckets


def _load_positive_int(name: str, default: int) -> int:
    try:
        value = int(config.get("action_space", {}).get(name, default))
    except (TypeError, ValueError):
        return default
    return max(1, value)


def _load_positive_decimal(name: str, default: float) -> Decimal:
    try:
        value = money.to_decimal(config.get("action_space", {}).get(name, default))
    except (TypeError, ValueError):
        return Decimal(str(default))
    if value <= 0:
        return Decimal(str(default))
    return value


ACTION_MODE = _load_action_mode()
TARGET_CONTRACTS = _load_target_contracts()
RISK_BUCKETS = _load_risk_buckets()
ACTION_COUNT = len(TARGET_CONTRACTS)
RISK_BUDGET_PCT = _load_positive_decimal("risk_budget_pct", 0.01)
STOP_ATR_MULTIPLE = _load_positive_decimal("stop_atr_multiple", 2.0)
MAX_CONTRACTS = _load_positive_int("max_contracts", 12)
MIN_NON_FLAT_CONTRACTS = _load_positive_int("min_non_flat_contracts", 1)


def _action_values() -> tuple[float, ...]:
    if ACTION_MODE == "atr_risk":
        return RISK_BUCKETS
    return tuple(float(value) for value in TARGET_CONTRACTS)


ACTION_VALUES = _action_values()
FLAT_ACTION = next((idx for idx, value in enumerate(ACTION_VALUES) if value == 0.0), ACTION_COUNT - 1)
LONG_ACTIONS = tuple(idx for idx, value in enumerate(ACTION_VALUES) if value > 0.0)
SHORT_ACTIONS = tuple(idx for idx, value in enumerate(ACTION_VALUES) if value < 0.0)


def _format_bucket(value: float) -> str:
    rendered = f"{abs(float(value)):.2f}".rstrip("0").rstrip(".")
    return rendered or "0"


def _build_action_labels() -> dict[int, str]:
    labels: dict[int, str] = {}
    for idx, value in enumerate(ACTION_VALUES):
        if value > 0:
            if ACTION_MODE == "atr_risk":
                labels[idx] = f"LONG_{_format_bucket(value)}R"
            else:
                labels[idx] = f"LONG_{abs(int(value))}"
        elif value < 0:
            if ACTION_MODE == "atr_risk":
                labels[idx] = f"SHORT_{_format_bucket(value)}R"
            else:
                labels[idx] = f"SHORT_{abs(int(value))}"
        else:
            labels[idx] = "FLAT"
    return labels


ACTION_LABELS = _build_action_labels()


def action_value(action: int) -> float:
    """Return the configured signed action value.

    In ``fixed_contracts`` mode this is the signed target contract count.
    In ``atr_risk`` mode this is the signed risk bucket in R units.
    """
    return float(ACTION_VALUES[int(action)])


def action_mode() -> str:
    """Return the configured action-space mode."""
    return ACTION_MODE


def atr_dollar_risk_per_contract(atr: Decimal | float | int | str | None) -> Decimal:
    """Return the configured ATR-stop dollar risk per contract."""
    if atr is None:
        return Decimal("0")
    try:
        atr_value = money.to_decimal(atr)
    except (TypeError, ValueError):
        return Decimal("0")
    if not atr_value.is_finite() or atr_value <= 0:
        return Decimal("0")
    return atr_value * money.to_decimal(constants.CONTRACT_POINT_VALUE) * STOP_ATR_MULTIPLE


def target_contracts_for_action(
    action: int,
    *,
    net_worth: Decimal | float | int | str | None = None,
    atr: Decimal | float | int | str | None = None,
    price: Decimal | float | int | str | None = None,
) -> int:
    """Return signed target contracts for an action id.

    In ``atr_risk`` mode the action value is a signed risk bucket and the
    function converts that to executable contracts using the configured risk
    budget and ATR stop multiple. When there is not enough context to compute
    ATR-scaled sizing, the fixed target-contract grid is used as a safe fallback.
    """
    action = int(action)
    if ACTION_MODE == "fixed_contracts":
        return TARGET_CONTRACTS[action]

    bucket = Decimal(str(RISK_BUCKETS[action]))
    if bucket == 0:
        return 0

    if net_worth is None or atr is None:
        return TARGET_CONTRACTS[action]

    try:
        net_worth_value = money.to_decimal(net_worth)
    except (TypeError, ValueError):
        return TARGET_CONTRACTS[action]
    if not net_worth_value.is_finite() or net_worth_value <= 0:
        return TARGET_CONTRACTS[action]

    per_contract_risk = atr_dollar_risk_per_contract(atr)
    if per_contract_risk <= 0:
        return TARGET_CONTRACTS[action]

    risk_budget_dollars = net_worth_value * RISK_BUDGET_PCT * abs(bucket)
    contracts = int((risk_budget_dollars / per_contract_risk).to_integral_value(rounding=ROUND_FLOOR))
    if contracts <= 0:
        contracts = MIN_NON_FLAT_CONTRACTS
    contracts = min(contracts, MAX_CONTRACTS)
    return contracts if bucket > 0 else -contracts


def action_label(action: int) -> str:
    """Return a compact human-readable label for an action id."""
    return ACTION_LABELS.get(int(action), f"UNKNOWN_{int(action)}")


def action_direction(action: int) -> int:
    """Return the sign of the configured action semantics: 1, -1, or 0."""
    value = action_value(action)
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def action_size_contracts(action: int) -> float:
    """Return the absolute configured action magnitude.

    This is the contract count in ``fixed_contracts`` mode and the absolute
    risk bucket in ``atr_risk`` mode.
    """
    return abs(action_value(action))


def action_from_target_contracts(target_contracts: int) -> Optional[int]:
    """Find the action id matching the fixed contract grid, if any."""
    target_contracts = int(target_contracts)
    for idx, contracts in enumerate(TARGET_CONTRACTS):
        if contracts == target_contracts:
            return idx
    return None
