# DeepTrade

DeepTrade is an end-to-end research framework that leverages modern Reinforcement Learning (RL) to build, train, and evaluate trading agents on high-frequency futures data (default: Nasdaq-100 micro futures \(NQ\)).  It is designed for practitioners who want to experiment with cutting-edge techniques—such as walk-forward validation, on-the-fly data normalization, and automatic hyper-parameter tuning—without spending weeks on plumbing code.

## Why DeepTrade?

* **Fully automated workflow** – Download or ingest data, engineer dozens of technical indicators, normalise them per training window, train an RL agent, then evaluate it on unseen data – all with a single command.
* **Emphasis on realistic testing** – The framework enforces strict out-of-sample evaluation via walk-forward windows and contains safeguards against data leakage.
* **Extensible & research-friendly** – Clean modular code makes it easy to plug-in new reward functions, indicators, risk-management rules, or even entirely different RL algorithms.

## Key Features

• **Proximal Policy Optimisation (PPO)** powered trading environment built on top of `gymnasium`.
• **Walk-Forward Testing** with optional parallel execution to simulate real-time deployment.
• **Hyper-parameter Optimisation** using Optuna with adaptive pruning.
• **Data Augmentation** (jitter, cut-&-paste, bootstrap, sliding window) to fight over-fitting.
• **Sigmoid-based Normalisation** that prevents look-ahead bias by fitting only on historical data.
• **Risk-Management Hooks** (stop-loss, take-profit, trailing stops, daily loss limits).
• **Rich Analytics & Plots** – Sharpe ratio, max draw-down, Calmar ratio, cumulative PnL and more.

## Tech Stack

* Python 3.10+
* [Stable-Baselines3](https://github.com/DLR-RMC/stable-baselines3) (PPO)
* Gymnasium (custom trading environment)
* Pandas, NumPy, SciPy – data wrangling & statistics
* Optuna – hyper-parameter tuning
* Matplotlib / Plotly – visualisations
* PyYAML – configuration

All heavy lifting happens on CPU, so no GPU is required.

## Project Layout (high-level)

```
├── environment.py          # Custom Gym trading environment
├── train.py                # One-shot training script
├── walk_forward.py         # Walk-forward validation runner
├── data/                   # Raw & processed datasets
├── config.yaml             # All tunable parameters in one place
└── models/                 # Saved models, logs & reports
```

## Quick Start

1. **Clone & set up Python environment**
   ```bash
   git clone https://github.com/your-user/deeptrade.git
   cd deeptrade
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download or provide data**

   DeepTrade expects a CSV with `timestamp, open, high, low, close, volume` (UTC).  A sample TradingView export is located at `data/NQ_2024_unix.csv`.  Alternatively let the framework fetch fresh Yahoo Finance data by setting `use_yfinance=True` in `train.py`.

3. **Train a baseline agent**
   ```bash
   python train.py  # uses parameters from config.yaml
   ```
   The script will create `models/` with the best model, metrics JSON and plots.

4. **Run walk-forward testing (recommended)**
   ```bash
   python walk_forward.py
   ```
   Progress and per-window reports are stored under `models/session_<timestamp>/`.

### Customising Experiments

Most knobs (indicator list, reward function, risk settings, PPO hyper-parameters, augmentation frequency, etc.) live in `config.yaml`.  Tweak them, then rerun `train.py` or `walk_forward.py`.

## Reproducibility

A global RNG seed is defined in `config.yaml` (default `42`) and propagated to NumPy, PyTorch, Gymnasium and Optuna to make experiments deterministic.

## Contributing

Bug reports, feature requests and pull-requests are welcome!  Please open an issue first to discuss major changes.

## License

MIT – see `LICENSE` file for details.

## Observation Features & Indicator Library

DeepTrade converts every candle into a rich feature vector consumed by the RL agent:

1. **Price & Position Context**
   * `close_norm` – last close price normalised to \[0, 1\].
   * `position` – current holding (-1 = short, 0 = flat, +1 = long).
   * `unrealised_profit_norm` – running P&L on the open position, rescaled to \[-1, 1\].
   * `steps_since_last_trade_norm` – how long the agent has been idle (caps at 50).

2. **Technical Indicators** (computed in `indicators/` and enabled via `config.yaml`)

   * Trend & Direction: **Supertrend**, **ADX / DI+, DI-**, **PSAR direction**
   * Momentum: **RSI**, **CCI**, **Rate-of-Change (ROC)**, **MACD (line / signal / hist)**, **Stochastic %K / %D**
   * Moving Averages: **SMA**, **EMA**, **Disparity Index** (price vs. SMA)
   * Volatility: **ATR** (Average True Range)
   * Volume–based: **OBV**, **Chaikin Money Flow (CMF)**, **VWAP**, rolling **Volume MA**
   * Seasonality / Calendar: **Day-of-Week sin/cos**, optional **Minutes-Since-Open sin/cos**
   * Statistical: **Z-Score** of price, **RRCF Anomaly Score** (Robust Random Cut Forest)

   Every indicator is first transformed with a sigmoid mapping to keep values inside \[-1, 1\] and to avoid leaking future information (parameters are estimated on the expanding training window only).

The final observation vector therefore looks like:

```
[ close_norm, indicator_1, …, indicator_N, position, unrealised_profit_norm, steps_since_last_trade_norm ]
```

where *N* depends on which indicators you enable. 