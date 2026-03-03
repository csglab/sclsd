"""Dataclass-based configuration objects for the LSD model.

This module provides configuration classes for the LSD model,
including model architecture, optimizer settings, and random walk
parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


class LogCoshActivation(nn.Module):
    """Smooth activation function using log(cosh(x)).

    This activation provides smooth gradients and is used in the
    potential network for learning the Waddington landscape.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply log-cosh activation."""
        return torch.log(torch.cosh(x))


# Type alias for activation specification
ActivationSpec = Union[str, nn.Module, type, None]


def resolve_activation(spec: ActivationSpec) -> nn.Module:
    """Convert activation specification to a concrete nn.Module instance.

    Parameters
    ----------
    spec : str, nn.Module, type, or None
        The activation specification. Can be:
        - A string: "logcosh", "relu", "softplus"/"sp", "identity"/"id"
        - An nn.Module instance (returned as-is)
        - A class that subclasses nn.Module (instantiated)
        - None (returns nn.Identity)

    Returns
    -------
    nn.Module
        The activation module.

    Raises
    ------
    ValueError
        If the string specification is not recognized.
    TypeError
        If the specification type is not supported.
    """
    if isinstance(spec, nn.Module):
        return spec
    if spec is None:
        return nn.Identity()
    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return spec()

    if isinstance(spec, str):
        key = spec.lower()
        if key == "logcosh":
            return LogCoshActivation()
        if key == "relu":
            return nn.ReLU()
        if key in {"softplus", "sp"}:
            return nn.Softplus()
        if key in {"identity", "id"}:
            return nn.Identity()
        raise ValueError(f"Unknown activation keyword: {spec}")

    raise TypeError(f"Unsupported activation spec: {spec}")


@dataclass
class LayerDims:
    """Layer dimensions for all neural network components.

    Attributes
    ----------
    B_decoder : List[int]
        Hidden layer sizes for the state decoder (B -> z).
    z_decoder : List[int]
        Hidden layer sizes for the expression decoder (z -> x).
    x_encoder : List[int]
        Hidden layer sizes for the expression encoder (x -> z).
    z_encoder : List[int]
        Hidden layer sizes for the state encoder (z -> B).
    xl_encoder : List[int]
        Hidden layer sizes for the library size encoder.
    potential : List[int]
        Hidden layer sizes for the potential network.
    potential_af : ActivationSpec
        Activation function for the potential network.
    """

    B_decoder: List[int] = field(default_factory=lambda: [32, 64])
    z_decoder: List[int] = field(default_factory=lambda: [128, 256])
    x_encoder: List[int] = field(default_factory=lambda: [512, 256])
    z_encoder: List[int] = field(default_factory=lambda: [64, 32])
    xl_encoder: List[int] = field(default_factory=lambda: [64, 32])
    potential: List[int] = field(default_factory=lambda: [32, 32])
    potential_af: ActivationSpec = "logcosh"

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with resolved activation.

        Returns
        -------
        dict
            Dictionary with layer dimensions and resolved activation.
        """
        dims = {
            "B_decoder": list(self.B_decoder),
            "z_decoder": list(self.z_decoder),
            "x_encoder": list(self.x_encoder),
            "z_encoder": list(self.z_encoder),
            "xl_encoder": list(self.xl_encoder),
            "potential": list(self.potential),
        }
        dims["potential_af"] = resolve_activation(self.potential_af)
        return dims


@dataclass
class AdamConfig:
    """Adam optimizer configuration with cosine annealing.

    Attributes
    ----------
    lr : float
        Initial learning rate.
    eta_min : float
        Minimum learning rate for cosine annealing.
    T_0 : int
        Number of iterations for the first restart.
    T_mult : int
        Factor for increasing T_i after each restart.
    """

    lr: float = 1e-3
    eta_min: float = 1e-5
    T_0: int = 40
    T_mult: int = 1

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "lr": self.lr,
            "eta_min": self.eta_min,
            "T_0": self.T_0,
            "T_mult": self.T_mult,
        }


@dataclass
class KLScheduleConfig:
    """KL divergence annealing schedule configuration.

    Attributes
    ----------
    af : float
        Annealing factor for KL term (affects B prior strength).
    """

    af: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with schedule parameters."""
        return {
            "min_af": self.af,
            "max_af": self.af,
            "max_epoch": 100,
        }


@dataclass
class WassersteinScheduleConfig:
    """Wasserstein regularization schedule configuration.

    Attributes
    ----------
    min_W : float
        Minimum Wasserstein coefficient.
    max_W : float
        Maximum Wasserstein coefficient.
    max_epoch : int
        Epoch at which minimum is reached.
    """

    min_W: float = 1e-4
    max_W: float = 1e-3
    max_epoch: int = 50

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_W": self.min_W,
            "max_W": self.max_W,
            "max_epoch": self.max_epoch,
        }


@dataclass
class OptimizerConfig:
    """Combined optimizer and schedule configuration.

    Attributes
    ----------
    adam : AdamConfig
        Adam optimizer settings.
    kl_schedule : KLScheduleConfig
        KL annealing schedule.
    wasserstein_schedule : WassersteinScheduleConfig
        Wasserstein regularization schedule.
    """

    adam: AdamConfig = field(default_factory=AdamConfig)
    kl_schedule: KLScheduleConfig = field(default_factory=KLScheduleConfig)
    wasserstein_schedule: WassersteinScheduleConfig = field(
        default_factory=WassersteinScheduleConfig
    )


@dataclass
class WalkConfig:
    """Random walk configuration for trajectory generation.

    Attributes
    ----------
    batch_size : int
        Number of walks per training batch.
    path_len : int
        Length of each random walk trajectory.
    num_walks : int
        Total number of walks to generate.
    random_state : int
        Random seed for walk generation.
    """

    batch_size: int = 256
    path_len: int = 10
    num_walks: int = 4096
    random_state: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration.

    Attributes
    ----------
    z_dim : int
        Dimension of the latent cell state representation.
    B_dim : int
        Dimension of the differentiation state (typically 2).
    V_coeff : float
        Regularization coefficient for the Waddington potential.
    layer_dims : LayerDims
        Layer dimensions for all network components.
    """

    z_dim: int = 10
    B_dim: int = 2
    V_coeff: float = 0.01
    layer_dims: LayerDims = field(default_factory=LayerDims)


@dataclass
class LSDConfig:
    """Complete configuration for the LSD model.

    This is the main configuration class that combines model,
    optimizer, and walk settings.

    Attributes
    ----------
    model : ModelConfig
        Model architecture settings.
    optimizer : OptimizerConfig
        Optimizer and schedule settings.
    walks : WalkConfig
        Random walk generation settings.

    Examples
    --------
    >>> from sclsd import LSDConfig
    >>> cfg = LSDConfig()
    >>> cfg.model.z_dim = 20
    >>> cfg.walks.path_len = 50
    >>> cfg.optimizer.adam.lr = 2e-3
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    walks: WalkConfig = field(default_factory=WalkConfig)


def default_config() -> LSDConfig:
    """Create a default LSD configuration.

    Returns
    -------
    LSDConfig
        Configuration with default values.
    """
    return LSDConfig()
