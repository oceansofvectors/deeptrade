# Experiments

## 1. Target-Contract Action Space
- Hypothesis: executable target contracts will eliminate allocation-rounding dead zones and materially reduce zero-trade policy collapse.
- Code/config switch: `config.yaml -> action_space.target_contracts`; compare current `[1, 2, 3, -1, -2, -3, 0]` against the legacy allocation branch if kept in git history.
- Primary metrics: stage-1 completed-trial rate, validation trade count, validation Sortino, holdout composite score.
- Failure criteria: >50% of stage-1 trials still hard-prune for `zero_trade` or `single_action`.
- Acceptance threshold: at least one completed stage-1 trial on each BTC scenario and a clear increase in median trade count.
- Artifact/report: `reports/hyperparameter_tuning_summary.json` plus a short scenario comparison table.

## 2. Raw-Unit Reward/Risk Inputs
- Hypothesis: keeping `ATR_RAW`, `ROLLING_DD_RAW`, and `VOL_PERCENTILE_RAW` out of scaling prevents reward/risk logic from reading distorted units.
- Code/config switch: current code path vs a branch that intentionally falls back to scaled `ATR`/regime columns.
- Primary metrics: stop/target trigger sanity, reward distribution, walk-forward max drawdown, policy stability across reruns.
- Failure criteria: materially different stop distances or reward spikes when scaler type changes.
- Acceptance threshold: reward/risk outcomes stay stable across scaler changes and ATR-based stops match raw price units.
- Artifact/report: per-window diagnostic dump of raw vs scaled regime columns and stop prices.

## 3. VAE Ablation
- Hypothesis: the LSTM VAE may help in some windows but can also add instability or redundant compression.
- Code/config switch: `indicators.lstm_features.enabled` on/off; also compare smaller latent sizes and lower `beta`.
- Primary metrics: validation Sortino, holdout composite, training stability, runtime.
- Failure criteria: VAE-on is slower and no better than VAE-off across most windows.
- Acceptance threshold: VAE-on beats VAE-off on median holdout composite by a meaningful margin.
- Artifact/report: `reports/lstm_tuning_results.json` plus an ablation summary table.

## 4. Tuning/Pruning After Action-Space Repair
- Hypothesis: once contract actions are executable, stage-1 pruning can be made stricter without killing the search.
- Code/config switch: sweep `pruning_eval_steps`, `early_prune_zero_trade`, `early_prune_zero_trade_min_step`, and `early_prune_max_drawdown_pct`.
- Primary metrics: completed-trial rate, best composite score, time-to-first-completed-trial.
- Failure criteria: all-pruned stage 1 or no improvement in completed-trial quality.
- Acceptance threshold: high completed-trial rate with fewer degenerate finalists.
- Artifact/report: tuning sweep summary by pruning policy.

## 5. Cost-Realism Sweep
- Hypothesis: policies that only survive at near-zero friction are not usable.
- Code/config switch: vary `environment.transaction_cost`, `execution_costs.half_spread_points`, and `execution_costs.base_slippage_points`.
- Primary metrics: return, Sortino, Calmar, trade count sensitivity.
- Failure criteria: strategy edge disappears under modestly realistic costs.
- Acceptance threshold: positive holdout composite under at least one realistic cost regime.
- Artifact/report: cost-sensitivity chart per scenario.

## 6. Supervised Benchmark Suite
- Hypothesis: a simpler supervised baseline may outperform or at least calibrate the RL stack on the same features and splits.
- Code/config switch: build baseline classifiers/regressors on the exact walk-forward features and windows.
- Primary metrics: directional accuracy, PnL after the same execution model, trade frequency.
- Failure criteria: RL underperforms simple baselines consistently.
- Acceptance threshold: RL matches or exceeds the best supervised baseline on holdout composite.
- Artifact/report: benchmark leaderboard saved under `reports/benchmarks/`.

## Next Code Plan
- Add a small walk-forward experiment harness that records scenario, config diff, metrics, and artifact paths in one JSON manifest.
- Add a dedicated ablation switch for VAE on/off and raw-risk fallback so experiments do not require manual code edits.
- Add per-trial action-distribution summaries to tuning reports for faster diagnosis of collapse modes.
- Add a cost-sweep runner that reuses saved window models and only reruns evaluation when possible.
