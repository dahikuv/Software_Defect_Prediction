"""Random seed utilities for reproducibility."""

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set common random seeds used in the project."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
