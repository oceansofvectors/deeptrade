"""
Device detection utility for PyTorch.

Provides automatic detection of the best available compute device:
CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU

Note: MPS has issues with LSTM/RecurrentPPO (MPSNDArray slice errors),
so recurrent models should use CPU on Apple Silicon.
"""
import logging

logger = logging.getLogger(__name__)


def get_device(preferred: str = "auto", for_recurrent: bool = False) -> str:
    """
    Get the best available compute device for PyTorch.

    Args:
        preferred: Device preference. Options:
            - "auto": Automatically detect best device (CUDA > MPS > CPU)
            - "cuda": Force CUDA (will fall back to CPU if unavailable)
            - "mps": Force MPS (will fall back to CPU if unavailable)
            - "cpu": Force CPU
        for_recurrent: If True, avoid MPS for recurrent models (LSTM) due to
            Metal Performance Shaders bugs with tensor slicing operations.

    Returns:
        Device string compatible with PyTorch/Stable-Baselines3: "cuda", "mps", or "cpu"
    """
    import torch

    if preferred == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            logger.debug(f"Using CUDA device: {device_name}")
        elif torch.backends.mps.is_available() and not for_recurrent:
            device = "mps"
            logger.debug("Using MPS device (Apple Silicon)")
        elif torch.backends.mps.is_available() and for_recurrent:
            device = "cpu"
            logger.debug("Using CPU device (MPS has LSTM compatibility issues)")
        else:
            device = "cpu"
            logger.debug("Using CPU device")
    elif preferred == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
            logger.debug(f"Using CUDA device: {device_name}")
        else:
            device = "cpu"
            logger.debug("CUDA requested but not available, falling back to CPU")
    elif preferred == "mps":
        if torch.backends.mps.is_available() and not for_recurrent:
            device = "mps"
            logger.debug("Using MPS device (Apple Silicon)")
        elif for_recurrent:
            device = "cpu"
            logger.debug("MPS requested but using CPU for recurrent model (LSTM compatibility)")
        else:
            device = "cpu"
            logger.debug("MPS requested but not available, falling back to CPU")
    else:
        device = "cpu"
        logger.debug("Using CPU device")

    return device


def get_device_info() -> dict:
    """
    Get detailed information about available compute devices.

    Returns:
        Dictionary with device availability and details
    """
    import torch

    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cpu_available": True,
        "recommended_device": get_device("auto"),
    }

    if info["cuda_available"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)

    return info
