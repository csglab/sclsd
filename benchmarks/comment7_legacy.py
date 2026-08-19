"""Frozen pre-comment-7 implementations used for equivalence benchmarks.

These functions intentionally preserve the dense behavior that existed before
the sparse-memory revision. They must not be used by production code.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import torch


def get_all_descendants(
    cluster: str,
    phylogeny: Dict[str, List[str]],
    descendants: Optional[set] = None,
) -> set:
    """Frozen helper used by the legacy phylogeny implementation."""
    if descendants is None:
        descendants = set()

    direct_children = phylogeny.get(cluster, [])
    descendants.update(direct_children)
    return descendants


def create_phylogeny_matrix(adata, phylogeny, cluster_key="clusters"):
    """Exact dense implementation from preprocessing/prior.py."""
    clusters = adata.obs[cluster_key].unique().tolist()
    cell_to_cluster = dict(zip(adata.obs_names, adata.obs[cluster_key]))

    all_descendants = {}
    for cluster in clusters:
        all_descendants[cluster] = get_all_descendants(cluster, phylogeny)

    n_cells = adata.shape[0]
    phylo_matrix = np.zeros((n_cells, n_cells))

    for i, cell_i in enumerate(adata.obs_names):
        cluster_i = cell_to_cluster[cell_i]
        for j, cell_j in enumerate(adata.obs_names):
            cluster_j = cell_to_cluster[cell_j]

            if cluster_i == cluster_j:
                phylo_matrix[i, j] = 1
            elif cluster_j in all_descendants.get(cluster_i, set()):
                phylo_matrix[i, j] = 1

    return sp.csr_matrix(phylo_matrix)


def calculate_transition_probs(
    potential: np.ndarray,
    connectivity_matrix: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """Exact dense implementation from train/trainer.py."""
    potential = potential.astype(float)
    energy_diff = potential[None, :] - potential[:, None]
    boltzmann_weights = np.exp(-beta * energy_diff)
    boltzmann_weights *= connectivity_matrix
    row_sums = boltzmann_weights.sum(axis=1, keepdims=True) + 1e-12
    transition_matrix = boltzmann_weights / row_sums
    return transition_matrix


def random_walks_dense(
    transition: torch.Tensor,
    n_trajectories: int,
    path_len: int,
    random_state: int,
) -> torch.Tensor:
    """Exact dense CPU equivalent of LSD._random_walks."""
    torch.manual_seed(random_state)
    n_cells = transition.shape[0]
    current_states = torch.randint(0, n_cells, (n_trajectories,), dtype=torch.long)
    walks = torch.empty((n_trajectories, path_len), dtype=torch.int)
    walks[:, 0] = current_states

    for step in range(1, path_len):
        next_states = torch.multinomial(
            transition[current_states], num_samples=1
        ).squeeze(1)
        walks[:, step] = next_states
        current_states = next_states

    return walks
