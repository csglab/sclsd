"""Evaluation metrics for trajectory inference.

This module provides metrics for evaluating the quality of inferred
cell trajectories, including cross-boundary correctness and in-cluster
coherence scores.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    from anndata import AnnData
except ImportError:
    AnnData = None


def _get_neighbor_indices(adata: "AnnData") -> np.ndarray:
    """Get neighbor indices from AnnData in a scanpy version-compatible way.

    In scanpy >= 1.11, neighbor indices are no longer stored directly in
    adata.uns['neighbors']['indices']. Instead, we extract them from the
    connectivities matrix in adata.obsp.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with computed neighbors.

    Returns
    -------
    np.ndarray
        Array of shape (n_cells,) containing arrays of neighbor indices for each cell.
    """
    # First try the legacy location (scanpy < 1.11)
    if "neighbors" in adata.uns and "indices" in adata.uns["neighbors"]:
        return adata.uns["neighbors"]["indices"]

    # For scanpy >= 1.11, extract from connectivities matrix
    connectivities_key = adata.uns.get("neighbors", {}).get("connectivities_key", "connectivities")
    if connectivities_key in adata.obsp:
        conn = adata.obsp[connectivities_key]
    elif "connectivities" in adata.obsp:
        conn = adata.obsp["connectivities"]
    else:
        raise KeyError(
            "Could not find neighbor indices. Make sure sc.pp.neighbors() has been run. "
            "Neither 'indices' in uns['neighbors'] nor connectivities in obsp was found."
        )

    # Extract indices from sparse connectivities matrix
    # Returns an array of arrays (one per cell)
    n_cells = conn.shape[0]
    indices = np.empty(n_cells, dtype=object)
    for i in range(n_cells):
        indices[i] = conn[i].nonzero()[1]

    return indices


def summary_scores(
    all_scores: Dict[str, List[float]],
) -> Tuple[Dict[str, float], float]:
    """Summarize group scores.

    Parameters
    ----------
    all_scores : dict
        Dictionary mapping group name to list of scores for individual cells.

    Returns
    -------
    sep_scores : dict
        Group-wise mean scores.
    overall_agg : float
        Overall mean score aggregated across all groups.
    """
    sep_scores = {k: np.mean(s) for k, s in all_scores.items() if s}
    overall_agg = np.mean([s for s in sep_scores.values() if s])
    return sep_scores, overall_agg


def _keep_type(
    adata: "AnnData",
    nodes: np.ndarray,
    target: str,
    k_cluster: str,
) -> np.ndarray:
    """Select cells of targeted type.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    nodes : np.ndarray
        Indices for cells.
    target : str
        Cluster name.
    k_cluster : str
        Cluster key in adata.obs.

    Returns
    -------
    np.ndarray
        Selected cell indices.
    """
    return nodes[adata.obs[k_cluster].iloc[nodes].values == target]


def cross_boundary_correctness(
    adata: "AnnData",
    k_cluster: str,
    k_velocity: str,
    cluster_edges: List[Tuple[str, str]],
    return_raw: bool = False,
    x_emb: str = "X_umap",
) -> Union[Dict, Tuple[Dict[Tuple[str, str], float], float]]:
    """Cross-Boundary Direction Correctness Score (A->B).

    Evaluates whether velocity vectors point from cluster A toward cluster B
    for cells near the boundary between clusters.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    k_cluster : str
        Key to the cluster column in adata.obs.
    k_velocity : str
        Key to the velocity matrix in adata.obsm.
    cluster_edges : list of tuples
        Pairs of clusters with transition direction (A, B) meaning A->B.
    return_raw : bool
        If True, return raw scores for each cell.
    x_emb : str
        Key to embedding for visualization (X_umap or X_tsne).

    Returns
    -------
    scores : dict
        Mean scores indexed by cluster_edges.
    mean_score : float
        Averaged score over all edges (only if return_raw=False).

    Examples
    --------
    >>> edges = [("Stem", "Prog"), ("Prog", "Mature")]
    >>> scores, mean = cross_boundary_correctness(adata, "clusters", "velocity", edges)
    """
    scores = {}
    all_scores = {}

    # Get velocity embedding
    if x_emb == "X_umap":
        v_emb = adata.obsm[f"{k_velocity}_umap"]
    elif x_emb == "X_tsne":
        v_emb = adata.obsm[f"{k_velocity}_tsne"]
    else:
        # Find the first matching key
        matching_keys = [key for key in adata.obsm if key.startswith(k_velocity)]
        if not matching_keys:
            raise KeyError(f"No velocity embedding found starting with '{k_velocity}'")
        v_emb = adata.obsm[matching_keys[0]]

    x_emb_data = adata.obsm[x_emb]

    # Get neighbor indices (compatible with scanpy >= 1.11)
    all_neighbor_indices = _get_neighbor_indices(adata)

    for u, v in cluster_edges:
        sel = adata.obs[k_cluster] == u
        sel_indices = np.where(sel)[0]
        nbs = [all_neighbor_indices[i] for i in sel_indices]

        boundary_nodes = [
            _keep_type(adata, np.array(nodes), v, k_cluster) for nodes in nbs
        ]
        x_points = x_emb_data[sel]
        x_velocities = v_emb[sel]

        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0:
                continue

            position_dif = x_emb_data[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(np.mean(dir_scores))

        scores[(u, v)] = np.mean(type_score) if type_score else 0.0
        all_scores[(u, v)] = type_score

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values()])


def inner_cluster_coh(
    adata: "AnnData",
    k_cluster: str,
    k_velocity: str,
    return_raw: bool = False,
) -> Union[Dict, Tuple[Dict[str, float], float]]:
    """In-cluster Coherence Score.

    Measures how coherent velocity vectors are within each cluster.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    k_cluster : str
        Key to the cluster column in adata.obs.
    k_velocity : str
        Key to the velocity matrix in adata.layers.
    return_raw : bool
        If True, return raw scores for each cell.

    Returns
    -------
    scores : dict
        Mean scores indexed by cluster name.
    mean_score : float
        Averaged score over all clusters (only if return_raw=False).

    Examples
    --------
    >>> scores, mean = inner_cluster_coh(adata, "clusters", "velocity")
    """
    clusters = np.unique(adata.obs[k_cluster])
    scores = {}
    all_scores = {}

    # Get neighbor indices (compatible with scanpy >= 1.11)
    all_neighbor_indices = _get_neighbor_indices(adata)

    for cat in clusters:
        sel = adata.obs[k_cluster] == cat
        sel_indices = np.where(sel)[0]
        nbs = [all_neighbor_indices[i] for i in sel_indices]
        same_cat_nodes = [
            _keep_type(adata, nodes, cat, k_cluster) for nodes in nbs
        ]

        velocities = adata.layers[k_velocity]
        cat_vels = velocities[sel]
        cat_score = [
            cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean()
            for ith, nodes in enumerate(same_cat_nodes)
            if len(nodes) > 0
        ]
        all_scores[cat] = cat_score
        scores[cat] = np.mean(cat_score) if cat_score else 0.0

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values()])


def evaluate(
    adata: "AnnData",
    cluster_edges: List[Tuple[str, str]],
    k_cluster: str,
    k_velocity: str = "velocity",
    x_emb: str = "X_umap",
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Evaluate velocity estimation results using multiple metrics.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    cluster_edges : list of tuples
        Pairs of clusters with transition direction (A, B).
    k_cluster : str
        Key to the cluster column in adata.obs.
    k_velocity : str
        Key to the velocity matrix.
    x_emb : str
        Key to embedding for visualization.
    verbose : bool
        Whether to print summary statistics.

    Returns
    -------
    dict
        Dictionary containing all metric scores.

    Examples
    --------
    >>> edges = [("Stem", "Prog"), ("Prog", "Mature")]
    >>> results = evaluate(adata, edges, "clusters", "velocity")
    """
    crs_bdr_crc = cross_boundary_correctness(
        adata, k_cluster, k_velocity, cluster_edges, return_raw=True, x_emb=x_emb
    )
    ic_coh = inner_cluster_coh(adata, k_cluster, k_velocity, return_raw=True)

    if verbose:
        sep_scores, overall = summary_scores(crs_bdr_crc)
        print(f"# Cross-Boundary Direction Correctness (A->B)")
        print(f"{sep_scores}")
        print(f"Total Mean: {overall}")

        sep_scores, overall = summary_scores(ic_coh)
        print(f"\n# In-cluster Coherence")
        print(f"{sep_scores}")
        print(f"Total Mean: {overall}")

    return {
        "Cross-Boundary Direction Correctness (A->B)": crs_bdr_crc,
        "In-cluster Coherence": ic_coh,
    }
