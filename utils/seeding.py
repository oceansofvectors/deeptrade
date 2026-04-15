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


def enable_full_determinism(seed: int):
    """Enable stricter deterministic behavior for PyTorch-backed training."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    set_global_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(seed_offset: int = 0):
    """Seed worker processes uniquely but reproducibly."""
    identity = multiprocessing.current_process()._identity
    worker_rank = identity[0] if identity else 0
    set_global_seed(seed_value + seed_offset + worker_rank)
