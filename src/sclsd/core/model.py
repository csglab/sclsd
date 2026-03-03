"""LSD Model with Pyro generative model and variational guide.

CRITICAL: The order of pyro.sample() calls in model() and guide()
determines the random number sequence and MUST NOT be changed.
Any reordering will produce different results even with the same seed.

Sample order in model():
1. pyro.module("LSD", self)
2. pyro.param("inverse_dispersion", ...)
3. pyro.sample('B', Normal(...))        # Within annealing scale
4. pyro.sample('z', Normal(...))
5. pyro.sample("xl", LogNormal(...))    # Conditional
6. pyro.sample("x", ZINB(...), obs=...) # Observed

Sample order in guide():
1. pyro.module("LSD", self)
2. pyro.factor("V_l2_reg", ...)
3. pyro.sample("z", Normal(...))
4. pyro.sample('B', Normal(...))        # Within annealing scale
5. pyro.sample("xl", LogNormal(...))    # Conditional
6. pyro.factor("W2", ...)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints
from torchdiffeq import odeint

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from sclsd.core.networks import (
    StateDecoder,
    ZDecoder,
    XEncoder,
    ZEncoder,
    LEncoder,
    PotentialNet,
    GradientNet,
)


# CI test flag for faster testing
smoke_test = "CI" in os.environ


def wasserstein_distance(
    encoder: XEncoder,
    x: torch.Tensor,
    path_len: int,
    latent_dim: int,
) -> torch.Tensor:
    """Compute Wasserstein-2 distance between consecutive steps along paths.

    Parameters
    ----------
    encoder : XEncoder
        The encoder network.
    x : torch.Tensor
        Input expression data of shape (batch * path_len, num_genes).
    path_len : int
        Length of each trajectory.
    latent_dim : int
        Dimension of latent space.

    Returns
    -------
    torch.Tensor
        Wasserstein-2 distance scalar.
    """
    W2 = 0
    mu, sigma = encoder(x)
    mu = mu.reshape(int(len(x) / path_len), path_len, latent_dim)
    sigma = sigma.reshape(int(len(x) / path_len), path_len, latent_dim)
    j = 0
    for i in range(1, path_len):
        term1 = torch.sum((mu[:, i, :] - mu[:, j, :]) ** 2)
        term2 = torch.sum((sigma[:, i, :] - sigma[:, j, :]) ** 2)
        j += 1
        W2 += term1 + term2

    return W2


def entropy_reg(
    state_enc: ZEncoder,
    s: torch.Tensor,
    path_len: int,
    state_dim: int,
) -> torch.Tensor:
    """Entropy regularization for the state encoder.

    Parameters
    ----------
    state_enc : ZEncoder
        The state encoder network.
    s : torch.Tensor
        Latent state of shape (batch * path_len, latent_dim).
    path_len : int
        Length of each trajectory.
    state_dim : int
        Dimension of differentiation state.

    Returns
    -------
    torch.Tensor
        Negative entropy regularization term.
    """
    _, sigma = state_enc(s)
    sigma = sigma.reshape(int(len(s) / path_len), path_len, state_dim)
    j = 0
    S = 0
    for i in range(1, path_len):
        term1 = 0.5 * torch.log(sigma[:, i, :] ** 2).sum(axis=-1)
        term2 = 0.5 * torch.log(sigma[:, j, :] ** 2).sum(axis=-1)
        gate = F.relu(term1 - term2)
        S += torch.sum(gate)

    return -S


class LSDModel(nn.Module):
    """Latent State Dynamics (LSD) Model.

    Implements the core LSD model for single-cell trajectories with
    neural ODE dynamics as a gradient flow in Waddington landscape.

    Parameters
    ----------
    B_dim : int
        Dimension of the differentiation state variable B (typically 2).
    z_dim : int
        Dimension of the cell state latent representation z.
    num_genes : int
        Number of gene features in the data.
    layer_dims : dict
        Dictionary specifying hidden layer sizes for all neural network modules.
    batch_size : int
        Number of cells or samples per batch.
    path_len : int
        Length of synthetic random walks or time points per trajectory.
    device : torch.device or str
        Device to use ("cpu" or "cuda").
    scale_factor : float, optional
        Multiplier for ELBO loss to scale gradients.
    V_coeff : float, optional
        Regularization coefficient for the Waddington potential.
    xl_loc : float, optional
        Prior mean parameter for library size distribution.
    xl_scale : float, optional
        Prior scale parameter for library size distribution.

    Examples
    --------
    >>> model = LSDModel(
    ...     B_dim=2,
    ...     z_dim=10,
    ...     num_genes=2000,
    ...     layer_dims={
    ...         "B_decoder": [32, 32],
    ...         "z_decoder": [64, 32],
    ...         "x_encoder": [128, 64],
    ...         "z_encoder": [128, 64],
    ...         "potential": [32, 16],
    ...         "potential_af": nn.ReLU(),
    ...     },
    ...     batch_size=128,
    ...     path_len=16,
    ...     device="cuda",
    ... )
    """

    def __init__(
        self,
        B_dim: int,
        z_dim: int,
        num_genes: int,
        layer_dims: Dict[str, Any],
        batch_size: int,
        path_len: int,
        device: torch.device,
        scale_factor: float = 1.0,
        V_coeff: float = 0.0,
        xl_loc: Optional[float] = None,
        xl_scale: Optional[float] = None,
    ):
        super().__init__()
        self.B_dim = B_dim
        self.z_dim = z_dim
        self.x_dim = num_genes
        self.layer_dims = layer_dims
        # Convert to Python float to ensure compatibility with PyTorch distributions
        # (numpy scalars are not accepted by PyTorch's distribution constructors)
        self.xl_loc = float(xl_loc) if xl_loc is not None else None
        self.xl_scale = float(xl_scale) if xl_scale is not None else None
        self.scale_factor = scale_factor
        self.V_coeff = V_coeff
        self.path_len = path_len
        self.batch_size = batch_size
        self.device = device

        # Model components - order of initialization does not affect RNG
        self.B_decoder = StateDecoder(
            hidden_dims=self.layer_dims["B_decoder"],
            latent_dim=self.z_dim,
            state_dim=self.B_dim,
        )
        self.z_decoder = ZDecoder(
            hidden_dims=self.layer_dims["z_decoder"],
            num_genes=self.x_dim,
            latent_dim=self.z_dim,
        )
        self.x_encoder = XEncoder(
            hidden_dims=self.layer_dims["x_encoder"],
            latent_dim=self.z_dim,
            num_genes=self.x_dim,
        )
        self.z_encoder = ZEncoder(
            hidden_dims=self.layer_dims["x_encoder"],  # Uses x_encoder dims
            latent_dim=self.z_dim,
            state_dim=self.B_dim,
        )
        self.xl_encoder = LEncoder(
            hidden_dims=self.layer_dims["z_decoder"],  # Uses z_decoder dims
            num_genes=self.x_dim,
        )
        self.potential = PotentialNet(
            hidden_dims=self.layer_dims["potential"],
            latent_dim=self.z_dim,
            af=self.layer_dims["potential_af"],
        )
        self.gradnet = GradientNet(self.potential)

        # RNN for prior of B
        self.rnn = nn.RNN(
            input_size=self.x_dim,
            hidden_size=self.B_dim,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
            num_layers=2,
        )
        self.h_0 = nn.Parameter(torch.zeros(2, 1, self.B_dim))
        self.epsilon = 1e-3

    def model(
        self,
        x_raw: torch.Tensor,
        x: torch.Tensor,
        xl: Optional[torch.Tensor] = None,
        annealing_factor: float = 1.0,
        W_coeff: float = 0.01,
    ) -> None:
        """Pyro generative model for Latent State Dynamics.

        CRITICAL: The order of pyro.sample() calls MUST NOT be changed.
        Changing the order will consume different random numbers and
        produce different results.

        Sample order (DO NOT CHANGE):
        1. pyro.module("LSD", self)
        2. pyro.param("inverse_dispersion", ...)
        3. pyro.sample('B', ...) - within annealing scale
        4. pyro.sample('z', ...)
        5. pyro.sample("xl", ...) - conditional
        6. pyro.sample("x", ..., obs=...) - observed

        Parameters
        ----------
        x_raw : torch.Tensor
            Observed gene expression raw counts.
        x : torch.Tensor
            Log-normalized expression profile.
        xl : torch.Tensor, optional
            Library size (if None, will be sampled from a prior).
        annealing_factor : float
            Annealing weight for the B prior.
        W_coeff : float
            Coefficient for Wasserstein regularization (unused in model).
        """
        # 1. Register module with Pyro
        pyro.module("LSD", self)

        # 2. Dispersion parameter for ZINB
        theta_x = pyro.param(
            "inverse_dispersion",
            100.0 * x.new_ones(self.x_dim),
            constraint=constraints.positive,
        )

        # Initial hidden state for RNN (expand to batch)
        h_0_contig = self.h_0.expand(
            2, int(len(x) / self.path_len), self.B_dim
        ).contiguous()

        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            # Prepare RNN input: [batch, path_len, features]
            x_rnn = x.reshape(int(len(x) / self.path_len), self.path_len, self.x_dim)
            rnn_output, _ = self.rnn(x_rnn, h_0_contig)
            rnn_output = rnn_output.reshape(len(x), self.B_dim)

            # 3. Sample B with annealing factor
            with poutine.scale(None, annealing_factor):
                B = pyro.sample(
                    "B",
                    dist.Normal(rnn_output, 10 * x.new_ones(self.B_dim)).to_event(1),
                )

            z_loc, z_scale = self.B_decoder(B)

            # Timepoint lattice for variance parameter cell state
            t = torch.exp(torch.linspace(0, 1, self.path_len).view(1, self.path_len, 1))
            t = t.expand(
                int(len(x) / self.path_len), self.path_len, self.z_dim
            ).contiguous()
            z_scale = z_scale.reshape(
                int(len(x) / self.path_len), self.path_len, self.z_dim
            )
            if torch.cuda.is_available():
                t = t.to(self.device)
            z_scale = (z_scale * t).reshape(-1, self.z_dim)

            # 4. Sample z
            z = pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))

            gate_logits, mu = self.z_decoder(z)

            # 5. Sample xl (conditional)
            if xl is None:
                xl_scale = self.xl_scale * x.new_ones(1)
                xl = pyro.sample(
                    "xl",
                    dist.LogNormal(self.xl_loc, xl_scale, validate_args=False).to_event(
                        1
                    ),
                )

            rate = (xl * mu + self.epsilon).log() - (theta_x + self.epsilon).log()
            x_dist = dist.ZeroInflatedNegativeBinomial(
                gate_logits=gate_logits,
                total_count=theta_x,
                logits=rate,
                validate_args=False,
            )

            # 6. Sample x (observed)
            pyro.sample("x", x_dist.to_event(1), obs=x_raw)

    def guide(
        self,
        x_raw: torch.Tensor,
        x: torch.Tensor,
        xl: Optional[torch.Tensor] = None,
        annealing_factor: float = 1.0,
        W_coeff: float = 0.01,
    ) -> None:
        """Pyro variational guide for Latent State Dynamics.

        CRITICAL: The order of pyro.sample() calls MUST NOT be changed.
        Changing the order will consume different random numbers and
        produce different results.

        Sample order (DO NOT CHANGE):
        1. pyro.module("LSD", self)
        2. pyro.factor("V_l2_reg", ...)
        3. pyro.sample("z", ...)
        4. pyro.sample('B', ...) - within annealing scale
        5. pyro.sample("xl", ...) - conditional
        6. pyro.factor("W2", ...)

        Parameters
        ----------
        x_raw : torch.Tensor
            Observed gene expression raw counts.
        x : torch.Tensor
            Log-normalized expression profile.
        xl : torch.Tensor, optional
            Library size (if None, will be sampled from guide).
        annealing_factor : float
            Annealing weight for the B posterior.
        W_coeff : float
            Coefficient for Wasserstein regularization.
        """
        # 1. Register module with Pyro
        pyro.module("LSD", self)

        with pyro.plate("batch", len(x)), poutine.scale(scale=self.scale_factor):
            z_loc, z_scale = self.x_encoder(x)
            V = self.potential(z_loc)

            # 2. L2 regularization on potential
            pyro.factor("V_l2_reg", self.V_coeff * V.pow(2).max(), has_rsample=True)

            # Timepoint lattice for neural ODE
            t = torch.linspace(0, 1, self.path_len)
            if torch.cuda.is_available():
                t = t.to(self.device)
            z_t = z_loc.reshape(
                int(len(x) / self.path_len), self.path_len, self.z_dim
            )
            z0 = z_t[:, 0, :].requires_grad_(True)

            # ODE solve
            assert t.shape[0] >= 2, (
                f"path_len must be >= 2 for ODE integration, got t={t}"
            )
            z_hat = odeint(self.gradnet, z0, t)
            z_hat = torch.transpose(z_hat, 0, 1)

            # Calculate loss term
            z_hat_loc = z_hat.reshape(-1, self.z_dim)

            # 3. Sample z
            z = pyro.sample("z", dist.Normal(z_hat_loc, z_scale).to_event(1))

            B_loc, B_scale = self.z_encoder(z)
            B_scale = B_scale.reshape(
                int(len(x) / self.path_len), self.path_len, self.B_dim
            )
            t = torch.exp(torch.linspace(0, 1, self.path_len).view(1, self.path_len, 1))
            t = t.expand(
                int(len(x) / self.path_len), self.path_len, self.B_dim
            ).contiguous()
            if torch.cuda.is_available():
                t = t.to(self.device)
            B_scale = (B_scale * t).reshape(-1, self.B_dim)

            # 4. Sample B with annealing factor
            with poutine.scale(None, annealing_factor):
                B = pyro.sample("B", dist.Normal(B_loc, B_scale).to_event(1))

            # 5. Sample xl (conditional)
            if xl is None:
                xl_loc, xl_scale = self.xl_encoder(x)
                pyro.sample(
                    "xl",
                    dist.LogNormal(xl_loc, xl_scale, validate_args=False).to_event(1),
                )

            # 6. Wasserstein regularization
            W2 = wasserstein_distance(self.x_encoder, x, self.path_len, self.z_dim)
            pyro.factor("W2", W_coeff * W2, has_rsample=True)
