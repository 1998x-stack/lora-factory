"""Reproducibility RNG seeding helpers."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed all RNGs (Python, NumPy, PyTorch CPU/CUDA) for reproducibility.

    Also enables deterministic cuDNN. Safe to call repeatedly; only affects
    the current process's random state.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
