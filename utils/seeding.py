# utils/seeding.py
import os, random
import numpy as np
import torch
import multiprocessing

def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)      # python‑level determinism
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seed_worker(base_seed: int):
    """
    Initialize random seeds for a worker process.
    Each worker gets a unique seed based on the base_seed and its process ID.
    
    Args:
        base_seed: The base seed to use (usually from config)
    """
    # Get current process ID to derive a unique worker seed
    worker_id = multiprocessing.current_process()._identity[0] if multiprocessing.current_process()._identity else 0
    worker_seed = base_seed + worker_id
    print(f"Worker {worker_id} using seed {worker_seed}")
    set_global_seed(worker_seed)
