"""Random seed utilities for reproducibility."""

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set common random seeds used in the project.

    Seeds Python's ``random``, NumPy, and (when installed) PyTorch, which backs
    the SBERT/transformer embedding step. ``PYTHONHASHSEED`` only takes effect
    when set before the interpreter starts, so we set it for any *child*
    processes but cannot change hash randomization for the running process; set
    ``PYTHONHASHSEED`` in the environment before launching Python for fully
    deterministic hashing.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except AttributeError:
        pass
