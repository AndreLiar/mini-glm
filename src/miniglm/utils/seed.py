"""Deterministic seeding across every RNG we touch.

Reproducibility is operating-principle #2. A result nobody can reproduce is not evidence.
We seed Python, NumPy and PyTorch (including MPS) from a single integer.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
