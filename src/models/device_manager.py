"""
Device Manager for PyTorch CUDA / CPU Hardware Selection

Safely checks GPU availability, tests tensor execution (handling new architectures like
RTX 50-series Blackwell sm_120 gracefully), and provides consistent device references.
"""

from __future__ import annotations

import logging
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("device_manager")


def get_optimal_device(force_cpu: bool = False) -> torch.device:
    """
    Determines whether CUDA can execute tensor kernels on the active GPU.
    If torch.cuda is available and kernel execution succeeds, returns torch.device('cuda').
    Otherwise, logs an informative notice and falls back gracefully to torch.device('cpu').
    """
    if force_cpu or not torch.cuda.is_available():
        logger.info("Using device: CPU (CUDA not requested or not available)")
        return torch.device("cpu")

    device_name = torch.cuda.get_device_name(0)
    try:
        # Probe tensor execution on GPU
        test_x = torch.zeros(2, 2, device="cuda")
        test_y = test_x + 1.0
        _ = test_y.sum().item()
        logger.info(f"Using device: CUDA [GPU 0: {device_name}]")
        return torch.device("cuda")
    except Exception as e:
        logger.warning(
            f"CUDA device detected ({device_name}), but kernel execution failed ({e}). "
            f"Falling back safely to high-performance CPU execution."
        )
        return torch.device("cpu")


if __name__ == "__main__":
    dev = get_optimal_device()
    print(f"Active Device: {dev}")

