# AlgoTrader2

An AI-powered algorithmic trading system for automated futures trading using reinforcement learning.

## Overview

AlgoTrader2 is a production-ready algorithmic trading platform that uses Proximal Policy Optimization (PPO) to trade futures contracts, primarily NASDAQ-100 E-mini (NQ/MNQ). The system integrates with Interactive Brokers for live trading and employs 28+ technical indicators to make intelligent trading decisions.

## Core Features

### Machine Learning Engine
- **Reinforcement Learning**: PPO algorithm from Stable-Baselines3
- **Custom Trading Environment**: Gymnasium-based environment simulating futures trading
- **Action Space**: Long (buy), Short (sell), or Hold
- **Reward System**: Logarithmic returns-based reward calculation

### Technical Analysis
The system uses 28+ configurable technical indicators:
- **Trend**: SuperTrend, SMA, EMA, MACD, Disparity Index
- **Momentum**: RSI, ROC, Williams %R, Stochastic Oscillator
- **Volatility**: ATR (Average True Range)
- **Volume**: OBV, CMF, Volume MA, VWAP
- **Directional**: ADX, ADX+, ADX-, CCI
- **Patterns**: Parabolic SAR
- **Time-based**: Day of week (sine/cosine encoded), minutes since market open

### Live Trading
- **Interactive Brokers Integration**: Real-time trading via IB API
- **5-Second Bar Processing**: Aggregates to 5-minute decision intervals
- **Automatic Contract Selection**: Chooses most liquid contract month
- **State Persistence**: Recovers from disconnections
- **Position Tracking**: Real-time monitoring of positions, orders, and executions

### Risk Management
- **Stop Loss**: Percentage-based or ATR-based (configurable)
- **Take Profit**: Automatic profit-taking (currently 20%)
- **Trailing Stop**: Dynamic stop loss following price movements
- **Daily Limits**:
  - Maximum daily loss: $3,000
  - Daily take profit target: $2,000
  - Automatic shutdown when limits hit
- **Position Sizing**: Configurable contract quantity

### Advanced Training

**Walk-Forward Optimization**:
- Sliding window approach with 120-day windows and 30-day steps
- Prevents overfitting through out-of-sample testing
- Parallel processing support

**Hyperparameter Tuning**:
- Optuna-based optimization
- Tunes learning rate, n_steps, entropy coefficient, batch size, gamma, gae_lambda
- 30 trials with multiple evaluation metrics

**Iterative Training**:
- Starts with 120,000 timesteps
- Continues training if validation improves
- Early stopping after 20 iterations without improvement

## Architecture

### Key Components

**live_trading.py**: Main entry point for live trading
- Connects to IB API
- Aggregates 5-second bars to 5-minute intervals
- Real-time PnL monitoring
- Automatic reconnection logic

**train_live_model.py**: Trains models for live deployment
- Loads data from data/live.csv
- Applies same indicators as live trading
- Saves as best_model.zip

**environment.py**: Custom Gymnasium trading environment
- Simulates futures trading (1:1 position tracking)
- $20 per point for NQ futures
- Calculates P&L based on price changes

**trading/model_trader.py**: Bridges model with IB API
- Preprocesses live bars
- Generates predictions
- Executes trades

**trading/bar_handler.py**: Real-time bar aggregation
- Aggregates 5-second to 5-minute bars
- Time synchronization
- Bucket management

## How It Works

### Training Phase
1. Download historical data or load from CSV
2. Calculate 28+ technical indicators
3. Normalize features to [-1, 1] range
4. Split data (60/20/20 train/validation/test)
5. Optionally run hyperparameter tuning
6. Train PPO agent with early stopping
7. Evaluate and save best model

### Live Trading Phase
1. Load trained model and scalers
2. Connect to Interactive Brokers
3. Subscribe to 5-second real-time bars
4. Aggregate to 5-minute intervals
5. On bar completion:
   - Calculate technical indicators
   - Normalize features
   - Get model prediction
   - Execute trade if action changes
6. Monitor PnL and daily limits
7. Auto-shutdown if limits hit

### Risk Management Flow
- Track entry price for each position
- Calculate unrealized P&L continuously
- Check stop loss/take profit levels
- Close position if risk limits breached
- Monitor daily cumulative P&L
- Hard stop at daily loss/profit limits

## Configuration

**config.yaml**: Main configuration
- Indicator settings
- Training parameters
- Data sources
- Feature normalization options
- Risk management settings (stop loss, take profit, trailing stop, position sizing)

**constants.py**: System constants
- Futures contract specifications (point values)
- Trading constraints and thresholds
- IB connection parameters
- File paths and defaults

## Technical Details

**Model Architecture**:
- Multi-Layer Perceptron (MLP) policy
- 29 features observation space
- Continuous normalized state space
- Deterministic predictions in live trading

**Training Configuration**:
- Total timesteps: 120,000 initial
- Learning rate: 0.00299
- N-steps: 562
- Batch size: 22
- Gamma: 0.95
- GAE Lambda: 0.95

**Supported Contracts**:
- MNQ (Micro E-mini NASDAQ-100)
- NQ (E-mini NASDAQ-100)
- ES (E-mini S&P 500)

## Data Pipeline

**Data Sources**:
- Yahoo Finance (historical)
- Interactive Brokers (live)
- Custom CSV files

**Processing**:
- MinMaxScaler normalization
- Per-window scaling to prevent leakage
- Market hours filtering (optional)
- 35-bar warmup period for indicators

## Current Status

- Last trading: 2025-05-29
- Active model: best_model.zip
- Risk settings: 20% take profit, no stop loss
- Position size: 1 contract
- All 28 indicators enabled

## Key Strengths

1. Production-ready with robust error handling
2. Comprehensive technical indicator suite
3. Walk-forward optimization prevents overfitting
4. State persistence and auto-recovery
5. Real-time monitoring and safety limits
6. Hyperparameter optimization support
7. Clean separation of training/live code
8. Timezone-aware market hours handling
9. Detailed logging for debugging
10. Parallel processing for efficiency
