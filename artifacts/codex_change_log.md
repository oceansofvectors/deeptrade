# Codex Change Log

## Retained Code Changes

### [environment.py](/home/orion/Documents/deeptrade/environment.py:930)

- Passed `opportunity_step_idx=max(0, self.current_step - 1)` into reward calculation.
- This removes the reward-shaping leak where the flat inactivity penalty could read the next bar's realized volatility regime.

### [environment.py](/home/orion/Documents/deeptrade/environment.py:968)

- Extended `_calculate_risk_adjusted_reward(...)` to accept `opportunity_step_idx`.
- Applied the flat-time penalty only when `_is_opportunity_regime(step_idx=opportunity_step_idx)` is true.

### [environment.py](/home/orion/Documents/deeptrade/environment.py:1059)

- Updated `_is_opportunity_regime(...)` to accept an explicit step index and clamp it safely.

### [tests/test_environment_reward_and_stops.py](/home/orion/Documents/deeptrade/tests/test_environment_reward_and_stops.py:59)

- Added `test_flat_inactivity_penalty_uses_current_bar_regime_not_next_bar`.
- This is the regression guard for the reward-timing issue.

### [train.py](/home/orion/Documents/deeptrade/train.py:1410)

- Added `_serialize_validation_results_for_report(...)`.
- The saved `validation_results.json` now keeps report-critical fields:
  - completed trades
  - rebalance count
  - Sortino / Calmar / max drawdown
  - collapse flags
  - fallback checkpoint-selection diagnostics

### [walk_forward.py](/home/orion/Documents/deeptrade/walk_forward.py:1270)

- `build_window_report_payload(...)` now writes completed trades and rebalances into each window payload.
- Walk-forward summary output now records `avg_completed_trades` and `avg_rebalances` alongside the legacy aggregate trade count.

### [reporting/walk_forward_report.py](/home/orion/Documents/deeptrade/reporting/walk_forward_report.py:67)

- Window and session summaries now separate completed trades from raw rebalances.
- Validation tables now include Sortino, Calmar, and max drawdown from the saved validation payload.
- HTML reports now surface fallback checkpoint selection when validation gates reject all checkpoints.
- Replay markers now infer entries/exits from the actual `trade_history` schema (`old_contracts`, `new_contracts`, `realized_trade`) instead of relying on simplified labels.

### [tests/test_train_helpers_extra.py](/home/orion/Documents/deeptrade/tests/test_train_helpers_extra.py:158)

- Added regression coverage for validation-result serialization.
- Confirmed that report-critical metrics and fallback flags persist to disk.

### [tests/test_walk_forward_report.py](/home/orion/Documents/deeptrade/tests/test_walk_forward_report.py:38)

- Extended HTML report coverage for:
  - completed-trade / rebalance summaries
  - validation Sortino rows
  - replay marker labels
  - fallback checkpoint-selection notes

### [scripts/codex_walk_forward_validation.py](/home/orion/Documents/deeptrade/scripts/codex_walk_forward_validation.py:1)

- Kept the reproducible validation runner.
- Extended consolidated outputs so scenario summaries now include completed-trade and rebalance metrics instead of collapsing everything into one ambiguous trade count.

## Fresh Validation Run

Executed on 2026-04-18:

- smoke report: `models/session_20260418_093705_762409/reports/walk_forward_report.html`
- default baseline: `models/session_20260418_093719_556246/reports/walk_forward_report.html`
- default candidate: `models/session_20260418_093752_742661/reports/walk_forward_report.html`
- 2023 baseline: `models/session_20260418_093826_590622/reports/walk_forward_report.html`
- 2023 candidate: `models/session_20260418_093854_683682/reports/walk_forward_report.html`
- 2024 stress baseline: `models/session_20260418_093923_100716/reports/walk_forward_report.html`
- 2024 stress candidate: `models/session_20260418_093942_180302/reports/walk_forward_report.html`
- 2022 baseline: `models/session_20260418_094001_839657/reports/walk_forward_report.html`
- 2022 candidate: `models/session_20260418_094035_364521/reports/walk_forward_report.html`

Relevant automated checks also passed:

- `venv/bin/python -m unittest tests.test_environment_reward_and_stops tests.test_train_iterative_selection tests.test_walk_forward_integration tests.test_walk_forward_tuning tests.test_train_evaluate_agent tests.test_action_space_helpers`
- `venv/bin/python -m unittest tests.test_walk_forward_report`
- `venv/bin/python -m unittest tests.test_walk_forward_report tests.test_train_helpers_extra tests.test_walk_forward_integration tests.test_walk_forward_plotting_and_config`

## Tested But Not Promoted

### [config.yaml](/home/orion/Documents/deeptrade/config.yaml:129)

- Re-tested `augmentation.synthetic_bears.oversample_ratio=0.08` against the baseline `0.18`.
- The lower-augmentation candidate was rejected again:
  - worse on the default full-cycle run
  - worse on `btc_recovery_2023`
  - much worse on the `btc_post_run_chop_2024` stress slice
  - slightly better bear-regime risk texture in `btc_bear_market_2022`, but still worse average return
- Final repo default remains `0.18`.

## Output Artifacts

- [docs/codex_overnight_improvement_report.md](/home/orion/Documents/deeptrade/docs/codex_overnight_improvement_report.md:1)
- [artifacts/codex_experiment_summary.json](/home/orion/Documents/deeptrade/artifacts/codex_experiment_summary.json:1)
- [artifacts/codex_validation_raw.json](/home/orion/Documents/deeptrade/artifacts/codex_validation_raw.json:1)
- [artifacts/walk_forward_validation_summary.csv](/home/orion/Documents/deeptrade/artifacts/walk_forward_validation_summary.csv:1)
- [artifacts/walk_forward_validation_summary.md](/home/orion/Documents/deeptrade/artifacts/walk_forward_validation_summary.md:1)

## Net Result

- Reward shaping is more temporally correct.
- The reporting path is materially more trustworthy because the HTML report now sees the full validation payload and the real trade-event schema.
- Validation coverage is reproducible and backed by fresh scenario runs and HTML reports.
- No new default candidate was promoted because profitability still did not improve robustly across scenarios.
