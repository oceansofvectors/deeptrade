# utils/seeding.py
import os, random
import numpy as np
import torch
import multiprocessing
from config import config

# --- walk_forward.py -------------------------------------------
seed_value = config.get("seed", 42)

def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)      # python‑level determinism
    os.environ["CUDA_VISIBLE_DEVICES"] = ""       # Force CPU usage
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Removed CUDA seeding - forcing CPU usage

def seed_worker(window_idx: int):
    set_global_seed(seed_value + window_idx)
