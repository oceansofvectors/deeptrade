# Live Trading Runbook (Paper + Rollover)

This runbook describes how to (a) train the "current" model, (b) verify it
offline with the replay harness, and (c) paper-trade NQ against Interactive
Brokers. It also covers the rollover cadence.

## 1. Train the current model

The `train_current_model.py` script trains exactly **one** RecurrentPPO
policy on the most-recent walk-forward-sized window and emits a self-contained
bundle suitable for live deployment.

```
python train_current_model.py --data data/NQ_live.csv
```

What this does, in order:
1. Loads the CSV and filters to NYSE market hours (`utils/data_utils.py`).
2. Slices the **last `config.walk_forward.window_size`** trading days (120 by
   default).
3. Splits train/validation/test = `config.data.{train,validation,test}_ratio`
   (60/20/20 by default) with `embargo_days=1`.
4. Fits an `LSTMFeatureGenerator` per
   `config.indicators.lstm_features.*` and transforms all three slices,
   saving `lstm_generator.pkl` and `lstm_autoencoder_checkpoint.pt`.
5. (Optional) augments train with synthetic bear segments if
   `config.augmentation.synthetic_bears.enabled`.
6. Fits the per-window RobustScaler (`normalization.scale_window`), saving
   `indicator_scaler.pkl` and `close_norm_params.pkl`.
7. Persists `feature_order.json` — the authoritative obs column layout the
   live side must reproduce.
8. Trains the RecurrentPPO model iteratively (`train.train_agent_iteratively`)
   and writes `best_model.zip`.
9. Evaluates on the test slice and writes `training_manifest.json` with
   full metadata and metrics.
10. Updates `models/current/latest` → the new bundle.

Useful overrides:

```
# Smoke-test the data pipeline without training
python train_current_model.py --skip-training

# Fast training (debugging only)
python train_current_model.py --timesteps 20000 --max-iterations 2

# Train on a deeper history file without touching the default
python train_current_model.py --data data/NQ_5min_2020_2025.csv

# Target a specific output directory and skip the latest symlink
python train_current_model.py --output-dir models/current/experiment_a --no-symlink
```

Bundle layout:

```
models/current/<timestamp>/
├── best_model.zip             # RecurrentPPO policy
├── indicator_scaler.pkl       # {'scaler', 'cols_to_scale', 'scaler_type'}
├── close_norm_params.pkl      # {'close_min', 'close_max'}
├── lstm_generator.pkl         # LSTMFeatureGenerator pickle
├── lstm_autoencoder_checkpoint.pt
├── feature_order.json         # {close_norm, indicators, state_features, observation_dim}
└── training_manifest.json     # git sha, data range, metrics, hyperparams
```

### Data freshness

The source CSV must cover **at least** `window_size` trading days (120)
ending as close to "now" as possible. The `data/NQ_live.csv` shipped in the
repo is stale — refresh it with your historical source before training for
live use. The yfinance fetcher in `get_data.download_data(period, interval)`
only supports 60 days at the 5-minute interval, so for longer windows you'll
want Interactive Brokers historical bars or a TradingView export.

## 2. Offline replay (required before paper trading)

```
python scripts/replay_live.py \
    --bundle models/current/latest \
    --data data/NQ_live.csv \
    --warmup 250 \
    --max-bars 1500
```

The harness instantiates `ModelTrader(ib_instance=None)` and feeds CSV bars
one at a time through `preprocess_bar` → `get_prediction`, simulating fills
so `current_position`, `unrealized_pnl`, `time_in_position`, and
`drawdown_pct` evolve as they would live.

Success criteria:
- `preprocess_failures` ≤ `warmup` (failures during indicator ramp-up are
  expected, past the warmup they should be zero).
- `position_flips > 0` — the model is actually trading.
- `obs_dim` matches the bundle's `observation_dim`.
- `drawdown_pct_final`, `session_peak_equity`, etc. are plausible.

If `obs` values are saturated at ±1 for most features, the model was trained
on data from a very different price regime than the replay data — retrain on
more-recent data before paper trading.

## 3. Paper trade NQ on port 7497

Prerequisites:
- TWS or IB Gateway running in **paper** mode with API access enabled on
  `127.0.0.1:7497`.
- A valid replay report (step 2) against the same bundle.
- `models/current/latest` points at the bundle you want to run.

Command:

```
python live_trading.py \
    --model models/current/latest \
    --contract NQ \
    --paper \
    --no_risk
```

`--no_risk` is recommended for the first paper session because the training
environment does **not** have stop-loss or take-profit enabled
(`config.risk_management.enabled: false`). Risk management lives in a
separate model (`risk_model.py`) that has not yet been wired into the live
loop — do not enable it here without revisiting the plan.

Watch the first full 5-minute cycle in `live_trading_<timestamp>.log`:
1. `Added bar to bucket ... 60/60 bars` — 5-second aggregation complete.
2. `Processing complete 5-minute bar` — synchronize_bars succeeded.
3. `Model action: 0|1|2` — prediction produced.
4. `Placed order ...` / `Position PnL Update` — order placed and PnL callback
   firing.

If the action stays flat (`2`) for the first full session, investigate
before concluding the model is broken — it may simply be waiting for a
signal.

Hold paper trading for **at least 3 full sessions** before any discussion of
going live.

## 4. Rollover cadence

The natural rollover interval is `config.walk_forward.step_size` trading
days (24 by default, ~monthly). Retrain sooner if any of the following fire:

- **Distribution drift**: live 5-day realized volatility leaves the
  10–90th percentile band of the training window's `VOL_PERCENTILE` for 3
  consecutive sessions.
- **Drawdown breach**: live daily drawdown exceeds 1.5× the
  training-window `max_drawdown` for 3 consecutive sessions.
- **Regime change**: live `ROLLING_DD` crosses a pre-declared threshold
  (e.g. −8%).

Rollover workflow:
1. Refresh `data/NQ_live.csv` with bars through "yesterday".
2. `python train_current_model.py` — produces a new timestamped bundle and
   updates `models/current/latest`.
3. Compare the new `training_manifest.json` metrics against the previous
   one. If the new bundle is worse on the test slice and on the replay
   harness, **do not** rotate — investigate first.
4. Rerun the replay harness against the new bundle.
5. Restart `live_trading.py` pointing at `models/current/latest`.
6. Monitor the first session carefully.

## 5. Known gaps (as of initial bring-up)

- `data/NQ_live.csv` is stale (~2025-05-28). Must be refreshed before any
  real paper trading — the existing artifact is only usable for plumbing
  smoke tests.
- The replay harness does not yet diff live obs vectors against what a
  `TradingEnv` would produce on the same bars. A stricter parity check is
  left as a follow-up.
- Risk model is not wired into live. Paper trade with `--no_risk`.
- `live_trading.py` still writes `trader_state.pkl` to CWD; make sure the
  working directory is the repo root when launching.
