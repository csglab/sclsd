"""Centralized RNG management for reproducibility.

CRITICAL: This module ensures identical results across runs by properly
seeding all random number generators used in the LSD model:
- Python's random module
- NumPy's random module
- PyTorch's random generators (CPU and CUDA)
- Pyro's random generators

The order of seeding matters for Pyro reproducibility. This module
must be used before any stochastic operations.
"""

from __future__ import annotations

import random
from typing import Optional, Union

import numpy as np
import torch
import pyro


def set_all_seeds(seed: int = 42) -> None:
    """Set all random seeds for complete reproducibility.

    This function seeds all random number generators to ensure
    deterministic behavior across runs. MUST be called before
    any stochastic operations.

    Parameters
    ----------
    seed : int, default=42
        The random seed to use for all generators.

    Notes
    -----
    The order of seeding is important for Pyro reproducibility.
    This function also sets PyTorch to deterministic mode.

    Examples
    --------
    >>> from sclsd.utils import set_all_seeds
    >>> set_all_seeds(42)  # Call before training
    """
    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch random (CPU)
    torch.manual_seed(seed)

    # PyTorch random (GPU) - must be after torch.manual_seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Pyro random - must be set for variational inference
    pyro.set_rng_seed(seed)

    # Ensure deterministic behavior for CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Get the appropriate torch device.

    Parameters
    ----------
    device : str, torch.device, or None
        The device to use. If None, automatically selects CUDA if available.

    Returns
    -------
    torch.device
        The selected device.

    Examples
    --------
    >>> device = get_device()  # Auto-select
    >>> device = get_device("cuda:0")  # Specific GPU
    >>> device = get_device("cpu")  # Force CPU
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def clear_pyro_state() -> None:
    """Clear Pyro's parameter store for fresh training.

    This should be called before training a new model to ensure
    no leftover parameters from previous runs.
    """
    pyro.clear_param_store()


def enable_pyro_validation(enable: bool = True) -> None:
    """Enable or disable Pyro's validation checks.

    Parameters
    ----------
    enable : bool, default=True
        Whether to enable validation checks.
    """
    pyro.enable_validation(enable)
