"""Utility functions for AlgoTrader3."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(preference: str = 'auto') -> torch.device:
    """
    Get the best available device.

    Args:
        preference: 'auto', 'cpu', 'cuda', or 'mps'

    Returns:
        torch.device
    """
    if preference == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(preference)
