"""
Fixed seed utility — ensures full reproducibility.
Import at the top of any module that uses randomness.

Usage:
    from src.utils.seed import set_global_seed
    set_global_seed()  # uses default seed
    set_global_seed(123)  # custom seed
"""

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Set seed for all RNGs: stdlib, numpy. Call once at entry point."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
