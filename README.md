# DeepTrade

DeepTrade is a reinforcement-learning trading system built around `stable-baselines3`, a custom `gymnasium` trading environment, and a feature stack that combines hand-crafted indicators with a pre-trained LSTM-VAE latent representation.

The project currently targets CME Micro Bitcoin futures semantics in code (`MBT`, point value `0.10`), and the default dataset in `config.yaml` is a minute-bar CSV at `data/glbx-mdp3-20210414-20260413.ohlcv-1m.csv`.

## How The Project Works

At a high level, the system does this:

1. Load OHLCV data from CSV or yfinance.
2. Compute configured indicators and session-derived features.
3. Optionally augment the training slice with synthetic bearish episodes.
4. Optionally fit an LSTM-VAE on train data only and append latent features `LSTM_F*`.
5. Fit normalization on the train split only and transform validation/test with the same parameters.
6. Train a PPO or RecurrentPPO policy in `TradingEnv`.
7. Select checkpoints on validation metrics with guardrails against degenerate policies.
8. Evaluate on held-out data, or repeat this inside walk-forward windows.

The main implementation files are:

- `get_data.py`: data loading and indicator generation
- `normalization.py`: train-only scaling and `close_norm` generation
- `indicators/lstm_features.py`: LSTM-VAE feature pretraining and latent extraction
- `environment.py`: trading environment, action space, execution model, reward function
- `train.py`: iterative PPO/RecurrentPPO training and evaluation
- `walk_forward.py`: rolling-window experiments, Optuna tuning, reporting
- `train_current_model.py`: train one current production-style model bundle
- `risk_model.py` and `train_risk_model.py`: separate risk gate pipeline

## Data

The default dataset is configured in `config.yaml` under `data.csv_path`. `get_data.py` expects OHLCV data and normalizes supported timestamp names to the internal `timestamp` field. In practice it accepts CSVs that use `timestamp`, `ts_event`, or `time`.

The default source file contains columns such as:

- `ts_event`
- `open`, `high`, `low`, `close`, `volume`
- metadata columns like `symbol`, `rtype`, `publisher_id`, `instrument_id`

Data processing in `get_data()`:

- loads CSV or downloads from yfinance
- coerces OHLCV columns to numeric
- converts the timestamp column to a sorted `DatetimeIndex`
- drops duplicate timestamps
- computes enabled indicators
- optionally filters to regular market hours when `data.market_hours_only` is enabled
- trims the warmup region at the front of the dataset
- splits into train/validation/test using `data.train_ratio`, `data.validation_ratio`, and `data.test_ratio`

For walk-forward runs, `walk_forward.py` does not use one fixed global split. It loads the full dataset, slices rolling windows by trading day, and then performs a fresh train/validation/test split inside each window with an embargo between validation and test.

Session-aware features are driven by `data.session` in `config.yaml`. The current defaults describe a session in `America/Chicago` from `17:00` to `16:00`, which is used by helpers in `utils/session_context.py`.

## Features

Feature engineering is config-driven. `process_technical_indicators()` in `get_data.py` computes only the indicators enabled in `config.yaml`.

Available feature families include:

- momentum and oscillators: `RSI`, `CCI`, `ROC`, `WILLIAMS_R`, stochastic
- trend: `ADX`, `MACD`, `SMA`, `EMA`, `SUPERTREND`, `DISPARITY`, `PSAR`
- volatility and structure: `ATR`, `Z_SCORE`, `ROLLING_DD`, `VOL_PERCENTILE`
- volume and price-location: `OBV`, `CMF`, `VWAP`, volume MA features
- session features: opening range, day of week, minutes since open
- anomaly features: `RRCF` anomaly score
- learned latent features: `LSTM_F0...LSTM_Fn`

The environment observation is:

- `close_norm`
- all numeric indicator/feature columns that survive preprocessing
- signed exposure
- normalized unrealized PnL
- normalized time in position
- current drawdown from peak

Normalization is handled in `normalization.scale_window()`:

- the scaler is fit on the training split only
- validation and test are transformed with train-fit parameters
- supported scalers are `minmax`, `robust`, and `quantile`
- the current config uses `robust`
- `close_norm` is also computed from train-only `close` min/max to avoid leakage

Synthetic bear augmentation lives in `utils/synthetic_bears.py`. When enabled, it appends contiguous bearish OHLCV segments to the training data before indicators and LSTM features are recomputed. The current synthetic regimes are:

- `mirrored_selloff`
- `accelerated_downtrend`
- `panic_crash`

## VAE

The learned feature block is implemented in `indicators/lstm_features.py`. Despite the older "LSTM features" naming, the current model is an LSTM variational autoencoder.

What it does:

- builds a 5-channel sequence input from raw OHLCV:
  - returns
  - high-low range
  - close-open difference
  - volume change
  - short rolling volatility
- creates sliding windows of length `lookback`
- trains an `LSTMVAE` on train sequences only
- uses the encoder mean as the latent representation
- appends latent columns `LSTM_F0`, `LSTM_F1`, ... to each split

The VAE is a beta-VAE:

- LSTM encoder
- mean and log-variance heads
- reparameterization trick
- LSTM decoder
- reconstruction loss plus weighted KL penalty
- KL warmup over the first configured epochs
- early stopping on monitored train/validation VAE loss

Important config keys live under `indicators.lstm_features`:

- `lookback`
- `hidden_size`
- `num_layers`
- `output_size`
- `beta`
- `kl_warmup_epochs`
- `pretrain_epochs`
- `pretrain_lr`
- `pretrain_batch_size`
- `pretrain_patience`
- `pretrain_min_delta`

There is also optional Optuna tuning for the VAE block under `indicators.lstm_features.tuning`.

## Model Pipeline(s)

### 1. Base Train/Test Pipeline

`train.py` is the direct training entry point:

1. Load train/validation/test from `get_data()`.
2. Scale features using train-only statistics.
3. Train a PPO or RecurrentPPO agent iteratively on the train split.
4. Evaluate each checkpoint on validation.
5. Keep the best checkpoint according to the configured metric.
6. Run final evaluation on the held-out test split.

### 2. Walk-Forward Pipeline

`walk_forward.py` is the main research pipeline and the most complete representation of how the project is intended to be used.

Per walk-forward window it does the following:

1. Slice a rolling trading-day window from the full dataset.
2. Split the window into train/validation/test with an embargo.
3. Optionally augment the train slice with synthetic bears.
4. Optionally fit the LSTM-VAE and append latent features.
5. Fit a train-only scaler and transform validation/test.
6. Drop redundant columns not needed at runtime.
7. Optionally run Optuna tuning for PPO/reward parameters.
8. Train the trading policy iteratively.
9. Evaluate on the test slice.
10. Save metrics, artifacts, plots, and an HTML report under `models/session_*`.

### 3. Current/Production Bundle Pipeline

`train_current_model.py` trains one self-contained bundle on the most recent walk-forward-sized window. It mirrors the walk-forward feature pipeline and writes deployable artifacts such as:

- `best_model.zip`
- `indicator_scaler.pkl`
- `close_norm_params.pkl`
- `lstm_generator.pkl`
- `lstm_autoencoder_checkpoint.pt`
- `feature_order.json`
- `training_manifest.json`

This is the path used by the live-trading runbook in `docs/live_trading_runbook.md`.

### 4. Risk Model Pipeline

`risk_model.py` implements a separate second-stage risk gate. It exists alongside the main trading policy rather than replacing the primary `TradingEnv` training path.

The flow is:

1. Train or load a trading model.
2. Replay it on data and collect trade history.
3. Build a `RiskManagementEnv` from those decisions.
4. Train a PPO model to `ALLOW` or `BLOCK` proposed trades.
5. Evaluate the trading model with and without the risk gate.

There is also a rule-based `RiskModelWrapper` mode with drawdown, cooldown, loss-streak, daily-loss, and minimum-bars-between-trades rules.

## Environment

`environment.py` defines `TradingEnv`, the core RL environment.

Action space:

- discrete mode: 7 actions
  - `0,1,2` -> target long allocations of `1%`, `2%`, `5%`
  - `3,4,5` -> target short allocations of `1%`, `2%`, `5%`
  - `6` -> flat
- dynamic risk mode: `MultiDiscrete([7, sl_choices, tp_choices])`
  - the first dimension is still the target allocation action
  - the other two dimensions choose ATR-based stop-loss and take-profit multipliers

Execution model:

- target allocation is converted to whole-contract exposure
- contract math uses constants from `constants.py`
- transaction costs are applied from config
- spread and time-of-day slippage are modeled via `execution_costs`
- optional fixed or dynamic ATR/percentage stops can force exits

State tracked by the environment includes:

- current contracts and signed exposure
- weighted entry price
- net worth and peak net worth
- time in position
- time spent flat
- trade count
- drawdown

The project supports standard PPO and recurrent PPO:

- `PPO` with `MlpPolicy`
- `RecurrentPPO` with `MlpLstmPolicy` when `sequence_model.enabled` is true and `sb3-contrib` is installed

## Config

Runtime configuration is loaded directly from `config.yaml` by `config.py`, so changing the YAML changes behavior across the whole project.

The most important sections are:

- `seed` and `reproducibility`
- `training`
- `data`
- `normalization`
- `indicators`
- `augmentation`
- `environment`
- `execution_costs`
- `reward`
- `model`
- `sequence_model`
- `risk_management`
- `walk_forward`
- `hyperparameter_tuning`

Notable current defaults:

- train/validation/test ratios: `0.6 / 0.2 / 0.2`
- walk-forward window size: `120` trading days
- walk-forward step size: `24` trading days
- evaluation metric: `sortino`
- normalization scaler: `robust`
- sequence model: enabled
- hyperparameter tuning: enabled
- synthetic bear augmentation: enabled
- risk management in the trading environment: currently disabled by default

## Tuning

There are three tuning paths in the repo.

### PPO/Reward Tuning

`walk_forward.py` contains the main Optuna tuning flow. The current search focuses on:

- `learning_rate`
- `n_steps`
- `ent_coef`
- `reward_turnover_penalty`
- `reward_calm_holding_bonus`

The tuning flow is staged:

- stage 1 searches a broader space
- stage 2 narrows around the stage-1 winner

Trials are pruned when they show clear signs of collapse or unusable behavior, for example:

- zero or too few trades
- flat-action dominance
- single-action collapse
- excessive drawdown

### LSTM-VAE Tuning

`indicators/lstm_features.py` can tune latent feature hyperparameters such as:

- `hidden_size`
- `num_layers`
- `output_size`
- `lookback`
- `pretrain_lr`
- `beta`

### Augmentation Tuning

`walk_forward.py --tune-augmentation` can sweep synthetic bear oversampling ratios using the candidates in `augmentation.tuning`.

## Training

The training loop in `train.train_agent_iteratively()` is iterative rather than one-shot.

Behavior:

- train for `initial_timesteps`
- evaluate on validation
- continue in `additional_timesteps` increments
- stop after `n_stagnant_loops` iterations without meaningful validation improvement
- save the best validation checkpoint, not just the latest checkpoint

Selection is metric-driven and guarded:

- the chosen metric can be return, hit rate, prediction accuracy, Calmar, or Sortino
- the current config uses Sortino
- candidates can be rejected if they do not trade enough, if action usage collapses, or if drawdown is too high
- if all candidates fail those gates, the code still keeps the least-bad fallback checkpoint

Useful commands:

```bash
python train.py
python walk_forward.py
python walk_forward.py --scenario btc_bear_market_2022
python walk_forward.py --tune-augmentation
python train_current_model.py --data data/NQ_live.csv
python train_risk_model.py
```

## Evaluation

The main evaluation helper is `train.evaluate_agent()`. It runs a trained policy through `TradingEnv` and returns:

- final portfolio value
- total return percent
- trade count
- completed trades and hit rate
- action counts
- max drawdown
- Calmar ratio
- Sortino ratio
- portfolio, position, and drawdown histories
- trade history records

`walk_forward.py` also includes `evaluate_agent_prediction_accuracy()`, which measures next-bar directional correctness instead of only PnL. That is useful when the configured optimization target is `prediction_accuracy`.

Walk-forward evaluation aggregates metrics across windows and writes reports and plots under `models/session_*`. The repository also has a substantial test suite under `tests/` covering:

- environment behavior
- reward logic and stops
- indicator calculations
- normalization
- LSTM-VAE behavior
- walk-forward orchestration
- trading/risk helpers

## Rewards

The trading reward in `TradingEnv` is portfolio-return based and is intended to align better with risk-adjusted objectives than raw profit deltas.

The reward combines:

- scaled log return of net worth
- extra asymmetry on losses via `reward.loss_multiplier`
- drawdown penalty once drawdown breaches `drawdown_penalty_threshold`
- turnover penalty when the agent changes position
- calm-holding bonus in low-volatility / low-drawdown regimes
- flat-time penalty after the configured grace period

Implementation notes:

- reward is scaled by `reward.base_scale`
- losses are penalized more heavily than gains when configured
- calm holding currently uses `ROLLING_DD` and `VOL_PERCENTILE` when those features are present
- reward is clipped to `[-10, 10]` to avoid extreme updates

The separate risk model has its own reward logic in `RiskManagementEnv`:

- allowing winning trades is rewarded
- allowing losing trades is penalized
- blocking losing trades gets a positive reward
- blocking winning trades gets a smaller penalty
- low drawdown receives a small bonus

## Environment Setup

The repository includes a `requirements.txt` with the core stack:

- PyTorch
- stable-baselines3
- sb3-contrib
- gymnasium
- pandas / numpy / scikit-learn
- optuna
- plotly / matplotlib
- yfinance / ib_insync
- pytz / pyyaml / tqdm
- `rrcf`

Install into a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Notes:

- `pandas_ta` is required by some indicator code and may need manual installation.
- recurrent training uses `sb3-contrib`.
- device selection is centralized in `utils/device.py`.
- deterministic mode and global seeding are handled in `utils/seeding.py`.

## Output Artifacts

Training and research runs write artifacts to:

- `models/session_*`: walk-forward sessions, reports, plots, tuning databases
- `models/current/*`: current production-style bundles
- `models/logs/*`: log files
- `data/*.csv`: cached processed splits

## Practical Starting Point

If you want the most representative end-to-end run for this repository, start with:

```bash
python walk_forward.py
```

If you want a single deployable bundle for live or replay workflows, use:

```bash
python train_current_model.py --data data/NQ_live.csv
```

If you want the direct non-walk-forward baseline, use:

```bash
python train.py
```
