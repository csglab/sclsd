"""Model save/load utilities for lsdpy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import torch
import pyro

if TYPE_CHECKING:
    from sclsd.core.model import LSDModel


def save_checkpoint(
    model: "LSDModel",
    path: Union[str, Path],
    *,
    extra_state: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model checkpoint including Pyro parameters.

    Parameters
    ----------
    model : LSDModel
        The LSD model to save.
    path : str or Path
        Path to save the checkpoint.
    extra_state : dict, optional
        Additional state to save with the checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "pyro_params": pyro.get_param_store().get_state(),
    }

    if extra_state is not None:
        checkpoint.update(extra_state)

    torch.save(checkpoint, path)


def load_checkpoint(
    model: "LSDModel",
    path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load model checkpoint including Pyro parameters.

    Parameters
    ----------
    model : LSDModel
        The LSD model to load weights into.
    path : str or Path
        Path to the checkpoint file.
    device : torch.device, optional
        Device to load the model onto.

    Returns
    -------
    dict
        The full checkpoint dictionary (may contain extra state).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load Pyro parameters
    pyro.get_param_store().set_state(checkpoint["pyro_params"])

    return checkpoint
