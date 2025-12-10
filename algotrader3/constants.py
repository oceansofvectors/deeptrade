"""
Constants for AlgoTrader3 World Model Trading System
"""

# Futures contract specifications
NQ_POINT_VALUE = 20.0  # $20 per point for NQ/MNQ futures
ES_POINT_VALUE = 50.0  # $50 per point for ES futures

# Action space
ACTION_LONG = 0    # Enter/maintain long position
ACTION_SHORT = 1   # Enter/maintain short position
ACTION_HOLD = 2    # Close position / go flat (exit any open trade)
NUM_ACTIONS = 3

# Position states
POSITION_FLAT = 0
POSITION_LONG = 1
POSITION_SHORT = -1

# Default training parameters
DEFAULT_LATENT_DIM = 32
DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_MIXTURES = 5
DEFAULT_SEQUENCE_LENGTH = 50

# Minimum balance threshold (fraction of initial)
MIN_BALANCE_PERCENTAGE = 0.01

# File paths
DEFAULT_DATA_PATH = "../data/NQ_2024_unix.csv"
DEFAULT_MODEL_DIR = "./checkpoints"
DEFAULT_LOG_DIR = "./runs"
