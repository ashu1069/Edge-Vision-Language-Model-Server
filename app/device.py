"""
Device detection and management utilities.

Provides automatic detection of available compute devices
(CUDA, MPS, CPU) for model inference.
"""

import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "mps", "cpu"]


def get_device() -> DeviceType:
    """
    Auto-detect the best available compute device.
    
    Checks for CUDA (NVIDIA GPU), then MPS (Apple Silicon),
    and falls back to CPU if neither is available.
    
    The DEVICE environment variable can override auto-detection:
    - 'auto': Auto-detect (default)
    - 'cuda': Force CUDA
    - 'mps': Force MPS
    - 'cpu': Force CPU
    
    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    device_override = os.getenv("DEVICE", "auto").lower()
    
    if device_override != "auto":
        logger.info(f"Using device override: {device_override}")
        return device_override  # type: ignore
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA available: {device_name}")
            return "cuda"
        
        if torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) available")
            return "mps"
        
    except ImportError:
        logger.debug("PyTorch not installed, using CPU")
    except Exception as e:
        logger.warning(f"Error detecting device: {e}")
    
    logger.info("Using CPU for inference")
    return "cpu"


def get_device_info() -> dict:
    """
    Get detailed information about the current device.
    
    Returns:
        dict with device type, name, memory info (if available)
    """
    device = get_device()
    info = {
        "device": device,
        "name": None,
        "memory_total_gb": None,
        "memory_available_gb": None,
    }
    
    try:
        import torch
        
        if device == "cuda":
            info["name"] = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            info["memory_total_gb"] = round(total_memory / (1024**3), 2)
            
            # Get available memory
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
            free = total_memory - reserved
            info["memory_available_gb"] = round(free / (1024**3), 2)
            
        elif device == "mps":
            info["name"] = "Apple Silicon GPU"
            # MPS doesn't expose memory info easily
            
        else:
            info["name"] = "CPU"
            # Could add psutil for CPU memory info if needed
            
    except Exception as e:
        logger.warning(f"Error getting device info: {e}")
    
    return info


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Sets seeds for Python's random, NumPy, and PyTorch
    (if available). For deterministic CUDA operations,
    additional configuration may be needed.
    
    Args:
        seed: Random seed value (default: 42)
    """
    import random
    random.seed(seed)
    
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # For full determinism (may impact performance):
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
            
    except ImportError:
        pass
    
    logger.debug(f"Random seed set to {seed}")
