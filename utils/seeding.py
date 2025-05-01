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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seed_worker(window_idx: int):
    set_global_seed(seed_value + window_idx)
