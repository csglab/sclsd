"""Random walk generation for trajectory sampling.

This module provides functions for generating random walks on the
cell-cell transition graph for training the LSD model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import scipy.sparse as sp


def prepare_transition_matrix_gpu(
    connectivity: sp.spmatrix,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Convert sparse connectivity matrix to dense transition matrix on GPU.

    Parameters
    ----------
    connectivity : scipy.sparse matrix
        Cell-cell connectivity matrix (e.g., from scanpy neighbors).
    device : torch.device
        Device to place the tensor on.

    Returns
    -------
    torch.Tensor
        Row-normalized transition probability matrix on GPU.
    """
    connectivity_dense = connectivity.toarray()
    row_sums = connectivity_dense.sum(axis=1, keepdims=True)
    T = connectivity_dense / row_sums
    T_tensor = torch.tensor(T, dtype=torch.float32, device=device)
    return T_tensor


def random_walks_gpu(
    T: torch.Tensor,
    n_steps: int,
    n_trajectories: int,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Generate random walks in parallel on GPU.

    Parameters
    ----------
    T : torch.Tensor
        Dense transition probability matrix of shape (n_cells, n_cells).
    n_steps : int
        Number of steps per walk (path_len).
    n_trajectories : int
        Number of random walks to generate.
    device : torch.device
        Device for computation.

    Returns
    -------
    torch.Tensor
        Tensor of shape (n_trajectories, n_steps) containing cell indices.
    """
    n_cells = T.shape[0]

    # Initialize all walks with random starting cells
    current_states = torch.randint(
        0, n_cells, (n_trajectories,), device=device, dtype=torch.long
    )
    walks = torch.empty((n_trajectories, n_steps), dtype=torch.long, device=device)
    walks[:, 0] = current_states

    # Simulate walks in parallel
    for step in range(1, n_steps):
        next_states = torch.multinomial(T[current_states], num_samples=1).squeeze(1)
        walks[:, step] = next_states
        current_states = next_states

    return walks


def prepare_walks(
    transition_matrix: torch.Tensor,
    n_trajectories: int,
    path_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Prepare random walks using GPU acceleration.

    Parameters
    ----------
    transition_matrix : torch.Tensor
        Transition probability matrix.
    n_trajectories : int
        Number of random walks to generate.
    path_len : int
        Length of each walk.
    device : torch.device
        Device for computation.

    Returns
    -------
    torch.Tensor
        Random walk indices on CPU.
    """
    T = transition_matrix.to(device)
    walks = random_walks_gpu(T, path_len, n_trajectories, device)
    return walks.cpu()


def prepare_simple_walks(
    n_steps: int,
    n_trajectories: int,
    connectivity: sp.spmatrix,
) -> torch.Tensor:
    """Prepare random walks using CPU (legacy function).

    Parameters
    ----------
    n_steps : int
        Number of steps per walk.
    n_trajectories : int
        Number of walks to generate.
    connectivity : scipy.sparse matrix
        Connectivity matrix.

    Returns
    -------
    torch.Tensor
        Random walk indices.
    """
    walks = []
    n_cells = connectivity.shape[0]

    # Normalize to transition probabilities
    row_sums = np.array(connectivity.sum(axis=1)).flatten()
    transition_probs = sp.diags(1.0 / row_sums) @ connectivity

    for _ in range(n_trajectories):
        walk = np.zeros(n_steps, dtype=int)
        walk[0] = np.random.randint(n_cells)

        for i in range(1, n_steps):
            probs = transition_probs[walk[i - 1]].toarray().flatten()
            walk[i] = np.random.choice(n_cells, p=probs)

        walks.append(walk)

    return torch.from_numpy(np.array(walks))
