"""
Central constants for the AlgoTrader2 system.

This module contains all hardcoded values used throughout the trading system,
making it easier to maintain and modify settings.
"""

# Futures contract specifications
NQ_POINT_VALUE = 20.0  # Dollar value per point for NQ/MNQ futures
ES_POINT_VALUE = 50.0  # Dollar value per point for ES futures

# Trading constraints
MIN_BALANCE_PERCENTAGE = 0.01  # Minimum balance as percentage of initial
MIN_EXECUTION_INTERVAL = 10  # Minimum seconds between trade executions
DATA_FLOW_THRESHOLD = 60  # Seconds before considering data flow stale
MAX_RECONNECTION_ATTEMPTS = 3  # Maximum IB reconnection attempts

# Risk management thresholds
UNREALISTIC_PROFIT_THRESHOLD = 10000.0  # Flag profits above this value
MULTIPLIER_THRESHOLD = 2.0  # Flag profit multipliers above this value
MIN_BALANCE_REQUIRED = 100.0  # Minimum account balance to continue trading

# Data processing
NORMALIZATION_WINDOW = 100  # Bars to use for rolling min-max normalization
WARMUP_BARS = 35  # Number of bars to trim for indicator warmup

# Interactive Brokers connection
IB_LIVE_PORT = 7496  # TWS/Gateway live trading port
IB_PAPER_PORT = 7497  # TWS/Gateway paper trading port
IB_TIMEOUT = 120  # Connection timeout in seconds
IB_CLIENT_ID = 1  # Client ID for IB connection

# Model parameters
DEFAULT_INITIAL_TIMESTEPS = 120000  # Initial training timesteps
MAX_STAGNANT_ITERATIONS = 20  # Iterations without improvement before stopping
IMPROVEMENT_THRESHOLD = 0.20  # Minimum improvement to continue training

# File paths
LIVE_DATA_FILE = "data/live.csv"
CONFIG_FILE = "config.yaml"
BEST_MODEL_FILE = "best_model.zip"

# Logging
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
