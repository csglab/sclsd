"""Neural network architectures for the LSD model.

This module contains the encoder and decoder networks used in the
LSD variational autoencoder, as well as the potential network for
learning the Waddington landscape.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_fc(dims: List[int]) -> nn.Sequential:
    """Create a fully-connected network with BatchNorm and Softplus activations.

    Parameters
    ----------
    dims : List[int]
        List of layer dimensions [input, hidden1, hidden2, ..., output].

    Returns
    -------
    nn.Sequential
        Network with Linear -> BatchNorm -> Softplus for each layer,
        except the last layer which has no activation.
    """
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.Softplus())
    # Exclude the last Softplus for final output layer
    return nn.Sequential(*layers[:-1])


def make_fc_wo_batch_norm(dims: List[int]) -> nn.Sequential:
    """Create a fully-connected network with Softplus but no BatchNorm.

    Parameters
    ----------
    dims : List[int]
        List of layer dimensions [input, hidden1, hidden2, ..., output].

    Returns
    -------
    nn.Sequential
        Network with Linear -> Softplus for each layer,
        except the last layer which has no activation.
    """
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.Softplus())
    return nn.Sequential(*layers[:-1])


def make_f(dims: List[int], af: nn.Module) -> nn.Sequential:
    """Create a fully-connected network with custom activation.

    Parameters
    ----------
    dims : List[int]
        List of layer dimensions.
    af : nn.Module
        Activation function to use after each linear layer.

    Returns
    -------
    nn.Sequential
        Network with Linear -> activation for all layers.
    """
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(af)
    return nn.Sequential(*layers)


def split_in_half(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split the last dimension of a tensor in half.

    Used for splitting encoder outputs into mean and scale components.

    Parameters
    ----------
    t : torch.Tensor
        Input tensor with even-sized last dimension.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Two tensors, each with half the last dimension.
    """
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)


class StateDecoder(nn.Module):
    """Decoder network: p(z|B).

    Maps differentiation state B to latent cell state z.

    Parameters
    ----------
    hidden_dims : List[int]
        Hidden layer dimensions.
    latent_dim : int, default=50
        Dimension of latent cell state z.
    state_dim : int, default=2
        Dimension of differentiation state B.
    """

    def __init__(
        self,
        hidden_dims: List[int],
        latent_dim: int = 50,
        state_dim: int = 2,
    ):
        super().__init__()
        dims = [state_dim] + hidden_dims + [2 * latent_dim]
        self.fc = make_fc_wo_batch_norm(dims)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Differentiation state B of shape (..., state_dim).

        Returns
        -------
        loc : torch.Tensor
            Mean of z distribution, shape (..., latent_dim).
        scale : torch.Tensor
            Standard deviation of z distribution, shape (..., latent_dim).
        """
        hidden = self.fc(x)
        hidden = hidden.reshape(x.shape[:-1] + hidden.shape[-1:])
        loc, scale = split_in_half(hidden)
        scale = self.softplus(scale)
        return loc, scale


class ZDecoder(nn.Module):
    """Decoder network: p(x|z) with ZINB likelihood.

    Maps latent cell state z to gene expression parameters.

    Parameters
    ----------
    hidden_dims : List[int]
        Hidden layer dimensions.
    num_genes : int
        Number of genes in the expression matrix.
    latent_dim : int
        Dimension of latent cell state z.
    """

    def __init__(
        self,
        hidden_dims: List[int],
        num_genes: int,
        latent_dim: int,
    ):
        super().__init__()
        dims = [latent_dim] + hidden_dims + [2 * num_genes]
        self.fc = make_fc(dims)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        z : torch.Tensor
            Latent cell state of shape (..., latent_dim).

        Returns
        -------
        gate_logits : torch.Tensor
            Logits for zero-inflation gate, shape (..., num_genes).
        mu : torch.Tensor
            Normalized expression rates, shape (..., num_genes).
        """
        gate, mu = split_in_half(self.fc(z))
        mu = F.softmax(mu, dim=-1)
        return gate, mu


class XEncoder(nn.Module):
    """Encoder network: q(z|x).

    Maps gene expression x to latent cell state z.

    Parameters
    ----------
    hidden_dims : List[int]
        Hidden layer dimensions.
    latent_dim : int
        Dimension of latent cell state z.
    num_genes : int
        Number of genes in the expression matrix.
    """

    def __init__(
        self,
        hidden_dims: List[int],
        latent_dim: int,
        num_genes: int,
    ):
        super().__init__()
        dims = [num_genes] + hidden_dims + [2 * latent_dim]
        self.fc = make_fc(dims)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Gene expression of shape (..., num_genes).

        Returns
        -------
        loc : torch.Tensor
            Mean of z distribution, shape (..., latent_dim).
        scale : torch.Tensor
            Standard deviation of z distribution, shape (..., latent_dim).
        """
        x = x.float()
        hidden = self.fc(x)
        hidden = hidden.reshape(x.shape[:-1] + hidden.shape[-1:])
        loc, scale = split_in_half(hidden)
        scale = self.softplus(scale)
        return loc, scale


class ZEncoder(nn.Module):
    """Encoder network: q(B|z).

    Maps latent cell state z to differentiation state B.

    Parameters
    ----------
    hidden_dims : List[int]
        Hidden layer dimensions.
    latent_dim : int, default=50
        Dimension of latent cell state z.
    state_dim : int, default=2
        Dimension of differentiation state B.
    """

    def __init__(
        self,
        hidden_dims: List[int],
        latent_dim: int = 50,
        state_dim: int = 2,
    ):
        super().__init__()
        dims = [latent_dim] + hidden_dims + [2 * state_dim]
        self.fc = make_fc_wo_batch_norm(dims)
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        z : torch.Tensor
            Latent cell state of shape (..., latent_dim).

        Returns
        -------
        loc : torch.Tensor
            Mean of B distribution, shape (..., state_dim).
        scale : torch.Tensor
            Standard deviation of B distribution, shape (..., state_dim).
        """
        hidden = self.fc(z)
        hidden = hidden.reshape(z.shape[:-1] + hidden.shape[-1:])
        loc, scale = split_in_half(hidden)
        scale = self.softplus(scale)
        return loc, scale


class LEncoder(nn.Module):
    """Library size encoder: q(xl|x).

    Parameters
    ----------
    hidden_dims : List[int]
        Hidden layer dimensions.
    num_genes : int
        Number of genes in the expression matrix.
    """

    def __init__(self, hidden_dims: List[int], num_genes: int):
        super().__init__()
        dims = [num_genes] + hidden_dims + [2]
        self.fc = make_fc(dims)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        s : torch.Tensor
            Gene expression of shape (..., num_genes).

        Returns
        -------
        l_loc : torch.Tensor
            Mean of library size distribution, shape (..., 1).
        l_scale : torch.Tensor
            Scale of library size distribution, shape (..., 1).
        """
        l_loc, l_scale = split_in_half(self.fc(s))
        l_scale = F.softplus(l_scale)
        return l_loc, l_scale


class PotentialNet(nn.Module):
    """Potential energy neural network for Waddington landscape.

    This network learns a scalar potential function V(z) that defines
    the energy landscape for cell differentiation dynamics.

    Parameters
    ----------
    hidden_dims : List[int]
        Hidden layer dimensions.
    latent_dim : int
        Dimension of latent cell state z.
    af : nn.Module
        Activation function (typically LogCosh for smooth gradients).
    """

    def __init__(
        self,
        hidden_dims: List[int],
        latent_dim: int,
        af: nn.Module,
    ):
        super().__init__()
        dims = [latent_dim] + hidden_dims + [1]
        self.fc = make_f(dims, af)
        self.lin = nn.Linear(latent_dim, 1)
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.sigmoid = nn.Sigmoid()
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Latent cell state of shape (..., latent_dim).

        Returns
        -------
        torch.Tensor
            Potential energy values, shape (..., 1).
        """
        gate = self.sigmoid(self.gate)
        out = gate * self.fc(x)
        return out


class GradientNet(nn.Module):
    """Gradient of potential for ODE dynamics.

    Computes -grad(V) for neural ODE integration, defining the
    gradient flow dynamics on the Waddington landscape.

    Parameters
    ----------
    potential : PotentialNet
        The potential network to differentiate.
    """

    def __init__(self, potential: PotentialNet):
        super().__init__()
        self.potential = potential

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for ODE solver.

        Parameters
        ----------
        t : torch.Tensor
            Time point (unused, required by ODE solver interface).
        x : torch.Tensor
            Latent cell state of shape (batch, latent_dim).

        Returns
        -------
        torch.Tensor
            Negative gradient of potential, shape (batch, latent_dim).
        """
        if not x.requires_grad:
            x = x.requires_grad_(True)
        potential = self.potential(x)
        grad = torch.autograd.grad(
            potential,
            x,
            grad_outputs=torch.ones_like(potential),
            create_graph=True,
        )[0]
        return -grad
