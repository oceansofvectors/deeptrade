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

def seed_worker(base_seed: int):
    """Initialize worker process with a unique seed based on the process ID."""
    # Use process ID to create unique seed for each worker
    process_id = multiprocessing.current_process().pid
    unique_seed = base_seed + (process_id % 10000)  # Mod to keep numbers reasonable
    set_global_seed(unique_seed)
