# Codex Overnight Improvement Report

## Scope

Objective: improve out-of-sample robustness of the QR-DQN trading path without adding external data, and make the walk-forward reporting path faithful enough that validation conclusions are based on the repo's actual reporting artifacts rather than raw console output.

Validation date: 2026-04-18

## Audit Findings

1. Reward correctness had a temporal bug: the flat-time inactivity penalty could key off the next bar's realized volatility regime instead of the current realized bar.
2. Validation/reporting had a second structural issue:
   - `validation_results.json` dropped important checkpoint-selection diagnostics and validation risk metrics before the HTML report consumed them.
   - the trade replay plot expected simplified `trade_type` labels that did not match the repo's actual `trade_history` schema, so markers could silently disappear.
3. The action-space and raw-vs-scaled risk plumbing are otherwise aligned in the current tree:
   - ATR-based sizing uses `ATR_RAW`
   - fixed ATR stops also use raw ATR units
   - evaluation and training both consume the same post-scaling feature layout
4. QR-DQN remains the only benchmarkable policy path worth carrying for this pass. PPO smoke references in the prior overnight work were flat or zero-trade under comparable short budgets and were not promoted.
5. A lower synthetic-bear oversample ratio (`0.08`) remained the most plausible simple candidate to test against the baseline default (`0.18`), but it still had to survive broader walk-forward validation before promotion.

## Implemented / Retained Changes

- Fixed the reward-timing issue in [environment.py](/home/orion/Documents/deeptrade/environment.py:930) so the inactivity penalty uses the current realized bar via `opportunity_step_idx=max(0, self.current_step - 1)`.
- Added the regression guard in [tests/test_environment_reward_and_stops.py](/home/orion/Documents/deeptrade/tests/test_environment_reward_and_stops.py:59).
- Preserved report-critical validation fields in [train.py](/home/orion/Documents/deeptrade/train.py:1410) so the saved `validation_results.json` now carries completed-trade counts, rebalance counts, Sortino, Calmar, max drawdown, collapse flags, and fallback checkpoint-selection diagnostics.
- Carried completed-trade vs rebalance metrics through [walk_forward.py](/home/orion/Documents/deeptrade/walk_forward.py:1270) and the walk-forward session summary at [walk_forward.py](/home/orion/Documents/deeptrade/walk_forward.py:3072).
- Fixed the HTML reporting layer in [reporting/walk_forward_report.py](/home/orion/Documents/deeptrade/reporting/walk_forward_report.py:67) so it:
  - displays completed trades separately from rebalances
  - surfaces validation Sortino/Calmar/max drawdown
  - flags fallback checkpoint selection in-window
  - renders replay markers from the real `trade_history` schema
- Kept and extended the reproducible validation runner in [scripts/codex_walk_forward_validation.py](/home/orion/Documents/deeptrade/scripts/codex_walk_forward_validation.py:1) so its consolidated artifacts also include completed-trade and rebalance metrics per scenario.

No new default model/config was promoted beyond the reward/reporting correctness fixes. The repo default remains `augmentation.synthetic_bears.oversample_ratio=0.18`.

## Verification

Relevant automated checks passed on 2026-04-18:

- `venv/bin/python -m unittest tests.test_environment_reward_and_stops tests.test_train_iterative_selection tests.test_walk_forward_integration tests.test_walk_forward_tuning tests.test_train_evaluate_agent tests.test_action_space_helpers`
- `venv/bin/python -m unittest tests.test_walk_forward_report`
- `venv/bin/python -m unittest tests.test_walk_forward_report tests.test_train_helpers_extra tests.test_walk_forward_integration tests.test_walk_forward_plotting_and_config`

Result:

- the earlier targeted regression bundle passed
- the report/train-helper suite passed
- the broader touched-area walk-forward/reporting suite passed

## Walk-Forward Validation Setup

Fresh smoke validation:

- session: `models/session_20260418_093705_762409`
- HTML report: `models/session_20260418_093705_762409/reports/walk_forward_report.html`
- budget: `initial_timesteps=2000`, `additional_timesteps=1000`, `max_iterations=1`, `max_windows=1`

Broader validation matrix:

- walk-forward budget:
  - `window_size=120`
  - `step_size=24`
  - `train_ratio=0.6`
  - `validation_ratio=0.2`
  - `embargo_days=1`
  - `initial_timesteps=5000`
  - `additional_timesteps=2000`
  - `max_iterations=2`
  - `max_windows=2`
  - no retuning during validation
- baseline: QR-DQN with `augmentation.synthetic_bears.oversample_ratio=0.18`
- candidate: QR-DQN with `augmentation.synthetic_bears.oversample_ratio=0.08`
- scenarios:
  - default full-cycle run
  - `btc_recovery_2023`
  - `btc_post_run_chop_2024` as the adverse / stress scenario
  - `btc_bear_market_2022`

## Walk-Forward Validation Results

### Smoke run

| Run | Return | Final Portfolio | Sortino | Calmar | Max Drawdown | Completed Trades | Rebalances | Hit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke candidate `0.08` | `-4.15%` | `$95,851.50` | `-2.19` | `-0.35` | `-12.02%` | `239.0` | `273.0` | `60.25%` |

### Scenario matrix

| Scenario | Model | Avg Return | Final Portfolio | Sortino | Calmar | Worst MaxDD | Completed Trades | Rebalances | Hit Rate | Profitable Windows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `default_full_cycle` | baseline `0.18` | `-6.53%` | `$93,474.25` | `-2.91` | `-0.51` | `-13.59%` | `228.5` | `237.0` | `65.16%` | `0/2` |
| `default_full_cycle` | candidate `0.08` | `-15.45%` | `$84,552.25` | `-6.89` | `-0.74` | `-25.01%` | `139.5` | `196.0` | `27.42%` | `0/2` |
| `btc_recovery_2023` | baseline `0.18` | `-3.83%` | `$96,173.50` | `-6.70` | `-0.57` | `-7.31%` | `311.0` | `377.0` | `57.15%` | `0/2` |
| `btc_recovery_2023` | candidate `0.08` | `-5.57%` | `$94,436.50` | `-9.00` | `-0.76` | `-7.67%` | `314.5` | `370.0` | `60.59%` | `0/2` |
| `btc_post_run_chop_2024` | baseline `0.18` | `14.38%` | `$114,378.50` | `5.13` | `1.27` | `-11.35%` | `79.0` | `188.0` | `59.49%` | `1/1` |
| `btc_post_run_chop_2024` | candidate `0.08` | `3.27%` | `$103,269.00` | `1.14` | `0.17` | `-19.36%` | `401.0` | `480.0` | `57.11%` | `1/1` |
| `btc_bear_market_2022` | baseline `0.18` | `-6.96%` | `$93,040.25` | `-6.47` | `-0.53` | `-13.96%` | `167.5` | `169.5` | `54.82%` | `0/2` |
| `btc_bear_market_2022` | candidate `0.08` | `-7.81%` | `$92,190.00` | `-5.78` | `-0.71` | `-11.16%` | `155.0` | `162.0` | `59.02%` | `0/2` |

### Reports reviewed

- smoke: `models/session_20260418_093705_762409/reports/walk_forward_report.html`
- default baseline: `models/session_20260418_093719_556246/reports/walk_forward_report.html`
- default candidate: `models/session_20260418_093752_742661/reports/walk_forward_report.html`
- 2023 baseline: `models/session_20260418_093826_590622/reports/walk_forward_report.html`
- 2023 candidate: `models/session_20260418_093854_683682/reports/walk_forward_report.html`
- 2024 stress baseline: `models/session_20260418_093923_100716/reports/walk_forward_report.html`
- 2024 stress candidate: `models/session_20260418_093942_180302/reports/walk_forward_report.html`
- 2022 baseline: `models/session_20260418_094001_839657/reports/walk_forward_report.html`
- 2022 candidate: `models/session_20260418_094035_364521/reports/walk_forward_report.html`

The regenerated HTML reports now surface fallback checkpoint selection where it occurred and distinguish completed trades from rebalances in both the summary tables and training diagnostics.

## Interpretation

- `default_full_cycle`: candidate materially failed. It lost far more return, had much worse Sortino/Calmar, and collapsed from `228.5` completed trades to `139.5`.
- `btc_recovery_2023`: candidate again underperformed on return, Sortino, Calmar, and worst-case drawdown. Slightly higher completed-trade count did not translate into better economics.
- `btc_post_run_chop_2024`: both variants were profitable, but the candidate gave up most of the PnL, took much worse drawdown, and exploded from `79.0` completed trades / `188.0` rebalances to `401.0` completed trades / `480.0` rebalances.
- `btc_bear_market_2022`: candidate improved some risk-shape diagnostics, but still did not improve average return or Calmar and therefore still does not qualify as a robust promotion.

## Final Assessment

What improved:

- Reward shaping is now temporally correct.
- Validation artifacts now preserve the checkpoint-selection diagnostics and validation risk metrics that the HTML report needs.
- The reporting pipeline now shows the economically relevant split between completed trades and raw rebalances.
- Validation is reproducible and backed by fresh scenario-level HTML reports and consolidated summaries.

What did not improve:

- Profitability did **not** improve robustly across scenarios.
- The `0.08` synthetic-bear candidate does not survive broader walk-forward validation after costs under the configured execution assumptions.

Decision:

- Keep the reward-correctness fix, the reporting-path fix, and the regression coverage.
- Keep the walk-forward validation tooling and regenerated summaries.
- Do **not** promote `augmentation.synthetic_bears.oversample_ratio=0.08`; leave the repo default at `0.18`.

Bottom line:

- The repo is more correct and easier to audit through its own reporting system.
- It is still **not** robustly profitable across the validated walk-forward scenarios.
