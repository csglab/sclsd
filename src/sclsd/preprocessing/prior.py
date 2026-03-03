"""Prior pseudotime and phylogeny inference for LSD models.

This module provides functions for inferring prior pseudotime values
and constructing phylogenetic relationships between cell clusters.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist

import torch
import pyro

try:
    import scanpy as sc
    from anndata import AnnData
except ImportError:
    sc = None
    AnnData = None


def get_prior_pseudotime(
    data_dict: Dict[str, Any],
    lsd_model: torch.nn.Module,
    origin_cluster: str | List[str],
    k_cluster: str = "clusters",
    device: torch.device = torch.device("cpu"),
    max_pseudotime_per_cluster: Optional[Dict[str, float]] = None,
) -> "AnnData":
    """Estimate prior pseudotime based on origin clusters.

    Parameters
    ----------
    data_dict : dict
        Must contain "normal_counts" and "adata".
    lsd_model : torch.nn.Module
        Trained LSD model with x_encoder and z_encoder.
    origin_cluster : str or list of str
        Cluster(s) in adata.obs[k_cluster] to treat as origin.
    k_cluster : str
        Name of column in adata.obs containing cluster labels.
    device : torch.device
        Device for computation.
    max_pseudotime_per_cluster : dict, optional
        Dictionary mapping cluster name to maximum pseudotime.
        Defaults to 1.0 for all clusters.

    Returns
    -------
    AnnData
        Copy of input AnnData with prior_pseudotime columns in .obs.
    """
    if isinstance(origin_cluster, str):
        origin_clusters = [origin_cluster]
    else:
        origin_clusters = origin_cluster

    if max_pseudotime_per_cluster is None:
        max_pseudotime_per_cluster = {orig: 1.0 for orig in origin_clusters}

    temp_adata = data_dict["adata"].copy()

    # Get latent representations
    x = torch.from_numpy(data_dict["normal_counts"]).to(device)
    with torch.no_grad():
        z_loc = lsd_model.x_encoder(x)[0].cpu().numpy()
        B_loc, _ = lsd_model.z_encoder(lsd_model.x_encoder(x)[0])
        B_loc = B_loc.cpu().numpy()

    # Store latent representations
    temp_adata.obsm["X_diff_state"] = B_loc
    temp_adata.obsm["X_cell_state"] = z_loc

    diff_rep = temp_adata.obsm["X_diff_state"]
    pseudotime_dict = {}

    for orig in origin_clusters:
        cluster_mask = temp_adata.obs[k_cluster] == orig
        cluster_cells = diff_rep[cluster_mask.values]
        other_cells = diff_rep[~cluster_mask.values]

        if cluster_cells.shape[0] == 0 or other_cells.shape[0] == 0:
            pseudotime = np.zeros(diff_rep.shape[0])
        else:
            distances = cdist(cluster_cells, other_cells, metric="euclidean")
            max_dist_idx = np.argmax(np.mean(distances, axis=1))
            cell_of_origin = cluster_cells[max_dist_idx]
            all_distances = np.linalg.norm(diff_rep - cell_of_origin, axis=1)

            if np.max(all_distances) == np.min(all_distances):
                normalized = np.zeros_like(all_distances)
            else:
                normalized = (all_distances - np.min(all_distances)) / (
                    np.max(all_distances) - np.min(all_distances)
                )

            scale = max_pseudotime_per_cluster.get(orig, 1.0)
            pseudotime = normalized * scale

        pseudotime_dict[orig] = pseudotime
        temp_adata.obs[f"prior_pseudotime_{orig}"] = pseudotime

    return temp_adata


def prior_transition_matrix(
    adata: "AnnData",
    timekey: str,
    beta_t: float,
    obsp_key: str = "connectivities",
) -> sp.csr_matrix:
    """Build transition matrix from pseudotime.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    timekey : str
        Key in adata.obs where pseudotime values are stored.
    beta_t : float
        Scaling parameter for pseudotime differences.
    obsp_key : str
        Key in adata.obsp for connectivity matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse transition matrix.
    """
    connectivity = adata.obsp[obsp_key].tocsr()
    pseudotime = adata.obs[timekey].values
    n_cells = connectivity.shape[0]

    row_idx, col_idx, data_vals = [], [], []

    for i in range(n_cells):
        start = connectivity.indptr[i]
        end = connectivity.indptr[i + 1]
        neighbors = connectivity.indices[start:end]

        if len(neighbors) == 0:
            continue

        dt = pseudotime[neighbors] - pseudotime[i]
        weights = np.exp(beta_t * dt)

        Z = weights.sum()
        normalized_weights = weights / Z if Z > 0 else np.zeros_like(weights)

        row_idx.extend([i] * len(neighbors))
        col_idx.extend(neighbors)
        data_vals.extend(normalized_weights)

    transition = sp.csr_matrix((data_vals, (row_idx, col_idx)), shape=(n_cells, n_cells))
    return transition


def compute_normalized_cluster_connectivity(
    adata: "AnnData",
    cluster_key: str = "clusters",
) -> pd.DataFrame:
    """Compute normalized connectivity between clusters.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    cluster_key : str
        Key in adata.obs for cluster labels.

    Returns
    -------
    pd.DataFrame
        Normalized cluster connectivity matrix.
    """
    conn = adata.obsp["connectivities"]
    if sp.issparse(conn):
        conn = conn.tocsr()

    clusters = adata.obs[cluster_key].astype(str)
    cluster_ids = clusters.unique()
    cluster_to_idx = {cl: np.where(clusters == cl)[0] for cl in cluster_ids}

    cluster_conn_matrix = pd.DataFrame(
        np.zeros((len(cluster_ids), len(cluster_ids))),
        index=cluster_ids,
        columns=cluster_ids,
    )

    for ci in cluster_ids:
        idx_i = cluster_to_idx[ci]
        for cj in cluster_ids:
            idx_j = cluster_to_idx[cj]
            sub_conn = conn[np.ix_(idx_i, idx_j)]
            total = sub_conn.sum()
            norm = len(idx_i) * len(idx_j)
            cluster_conn_matrix.loc[ci, cj] = total / norm if norm > 0 else 0

    cluster_conn_matrix /= cluster_conn_matrix.sum(axis=0)
    for ci in cluster_ids:
        cluster_conn_matrix.loc[ci, ci] = 0

    return cluster_conn_matrix


def _find_highly_connected_dict(
    cluster_conn: pd.DataFrame,
    ratio: float = 0.1,
    hot_ratio: float = 0.05,
) -> Dict[str, List[str]]:
    """Find highly connected cluster pairs."""
    conn = cluster_conn.copy()
    np.fill_diagonal(conn.values, np.nan)

    high_conn_dict = {cluster: [] for cluster in conn.index}

    for source in conn.index:
        max_conn = np.nanmax(conn.loc[:, source].values)
        candidates = [conn.loc[source, :].idxmax()]
        high_conn_dict[source].append(candidates[0])
        threshold = max_conn * ratio

        for target in conn.columns:
            if source != target and conn.loc[source, target] >= threshold:
                if target not in candidates:
                    high_conn_dict[source].append(target)
                    candidates.append(target)

        for target in conn.columns:
            if source != target and conn.loc[source, target] >= hot_ratio * max_conn:
                if target not in candidates:
                    if all(
                        conn.loc[c, target] < conn.loc[source, target]
                        for c in candidates
                    ):
                        high_conn_dict[source].append(target)

    return high_conn_dict


def _deduplicate_targets_by_connectivity(
    connectivity_dict: Dict[str, List[str]],
    cluster_conn: pd.DataFrame,
    clusters_of_interest: List[str],
) -> Dict[str, List[str]]:
    """Deduplicate targets to ensure unique assignment."""
    reverse_map = defaultdict(list)
    for source in clusters_of_interest:
        for target in connectivity_dict.get(source, []):
            reverse_map[target].append(source)

    final_dict = {source: [] for source in clusters_of_interest}

    for target, sources in reverse_map.items():
        if len(sources) == 1:
            final_dict[sources[0]].append(target)
        else:
            best_source = max(sources, key=lambda src: cluster_conn.loc[src, target])
            final_dict[best_source].append(target)

    return final_dict


def infer_phylo(
    adata: "AnnData",
    root: str,
    cluster_key: str = "clusters",
    ratio: float = 0.3,
    hot_ratio: float = 0.05,
) -> Dict[str, List[str]]:
    """Infer phylogenetic tree structure from data.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    root : str
        Root cluster name.
    cluster_key : str
        Key in adata.obs for cluster labels.
    ratio : float
        Connectivity ratio threshold.
    hot_ratio : float
        Hot connectivity ratio threshold.

    Returns
    -------
    dict
        Phylogeny dictionary {parent: [children]}.
    """
    if sc is None:
        raise ImportError("scanpy is required for phylogeny inference")

    phylo_adata = adata.copy()
    all_clusters = set(phylo_adata.obs[cluster_key].unique())
    assigned_clusters = {root}
    phylo = {root: []}
    origins = [root]

    while len(assigned_clusters) < len(all_clusters) and len(origins) > 0:
        conn = compute_normalized_cluster_connectivity(phylo_adata, cluster_key=cluster_key)
        conn_dict = _find_highly_connected_dict(conn, ratio=ratio, hot_ratio=hot_ratio)

        for k in conn_dict:
            conn_dict[k] = [t for t in conn_dict[k] if t not in origins]

        conn_dict = _deduplicate_targets_by_connectivity(conn_dict, conn, origins)

        new_origins = []
        for origin in origins:
            children = conn_dict.get(origin, [])
            phylo[origin] = children
            new_origins.extend(children)
            assigned_clusters.update(children)

        phylo_adata = phylo_adata[~phylo_adata.obs[cluster_key].isin(origins)]

        if phylo_adata.n_obs > 0:
            sc.pp.pca(phylo_adata)
            sc.pp.neighbors(phylo_adata)

        origins = new_origins

    unassigned = all_clusters - assigned_clusters
    for cluster in unassigned:
        phylo[cluster] = []
    for cluster in all_clusters:
        if cluster not in phylo:
            phylo[cluster] = []

    return phylo


def _get_all_descendants(
    cluster: str,
    phylogeny: Dict[str, List[str]],
    descendants: Optional[set] = None,
) -> set:
    """Recursively find all descendants of a cluster."""
    if descendants is None:
        descendants = set()

    direct_children = phylogeny.get(cluster, [])
    descendants.update(direct_children)

    return descendants


def _create_phylogeny_matrix(
    adata: "AnnData",
    phylogeny: Dict[str, List[str]],
    cluster_key: str = "clusters",
) -> sp.csr_matrix:
    """Create cell-cell matrix based on phylogeny relationships."""
    clusters = adata.obs[cluster_key].unique().tolist()
    cell_to_cluster = dict(zip(adata.obs_names, adata.obs[cluster_key]))

    all_descendants = {}
    for cluster in clusters:
        all_descendants[cluster] = _get_all_descendants(cluster, phylogeny)

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


def _get_tree_branches(
    phylo: Dict[str, List[str]],
    root: str,
) -> List[List[str]]:
    """Get all branches from root to leaves."""
    branches = []

    def dfs(node: str, path: List[str]) -> None:
        children = phylo.get(node, [])
        if not children:
            branches.append(path + [node])
        else:
            for child in children:
                dfs(child, path + [node])

    dfs(root, [])
    return branches


def infer_prior_time(
    data_dict: Dict[str, Any],
    device: torch.device,
    pseudotime_cluster: str | List[str],
    n_epochs: int = 20,
    random_state: int = 42,
) -> "AnnData":
    """Infer prior pseudotime using a temporary LSD model.

    Parameters
    ----------
    data_dict : dict
        Data dictionary from prepare_data_dict.
    device : torch.device
        Device for computation.
    pseudotime_cluster : str or list
        Origin cluster(s) for pseudotime inference.
    n_epochs : int
        Number of training epochs.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        AnnData with inferred prior pseudotime.
    """
    from sclsd.train.trainer import LSD
    from sclsd.core.config import LSDConfig
    from sclsd.utils.seed import set_all_seeds

    set_all_seeds(random_state)

    temp = data_dict["adata"].copy()

    # Prepare transition matrix
    P = temp.obsp["connectivities"]
    row_sums = np.array(P.sum(axis=1)).flatten()
    P = sp.diags(1.0 / row_sums) @ P
    n_trajectories = 2 ** (int(np.log2(len(temp))))

    # Configure model
    cfg = LSDConfig()
    cfg.walks.batch_size = 256
    cfg.walks.path_len = 2
    cfg.walks.num_walks = n_trajectories
    cfg.walks.random_state = random_state
    cfg.optimizer.wasserstein_schedule.max_W = 1
    cfg.optimizer.wasserstein_schedule.min_W = 1
    cfg.optimizer.adam.eta_min = 1e-4
    cfg.optimizer.adam.T_0 = 30
    cfg.model.z_dim = 50
    cfg.walks.batch_size = int(n_trajectories / 16)

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    pyro.clear_param_store()
    temp_lsd = LSD(
        data_dict["adata"],
        cfg,
        device=device,
        lib_size_key="librarysize",
        raw_count_key="raw",
    )
    temp_lsd.set_prior_transition(prior_transition=P)
    temp_lsd.prepare_walks(n_trajectories=n_trajectories)
    temp_lsd.train(num_epochs=n_epochs, plot_loss=False)

    temp_adata = get_prior_pseudotime(
        data_dict,
        temp_lsd.lsd,
        pseudotime_cluster,
        k_cluster="clusters",
        device=device,
    )

    return temp_adata


def get_prior_transition(
    adata: "AnnData",
    n_steps: int,
    n_trajectories: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    cluster_key: str = "clusters",
    phylogeny: Optional[Dict[str, List[str]]] = None,
    root_cluster: Optional[str] = None,
    time_key: Optional[str] = None,
    random_state: int = 42,
    beta_t: float = 50,
    ratio: float = 0.5,
    hot_ratio: float = 0.05,
) -> Tuple["AnnData", np.ndarray]:
    """Get prior transition matrix for random walks.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    n_steps : int
        Number of steps per walk.
    n_trajectories : int, optional
        Number of trajectories.
    device : torch.device
        Device for computation.
    cluster_key : str
        Key for cluster labels.
    phylogeny : dict, optional
        Pre-defined phylogeny.
    root_cluster : str, optional
        Root cluster for phylogeny inference.
    time_key : str, optional
        Key for pseudotime values.
    random_state : int
        Random seed.
    beta_t : float
        Temperature for transition probabilities.
    ratio : float
        Connectivity ratio for phylogeny inference.
    hot_ratio : float
        Hot ratio for phylogeny inference.

    Returns
    -------
    adata : AnnData
        Updated AnnData with transition matrix.
    transition : np.ndarray
        Dense transition matrix.
    """
    from sclsd.preprocessing.data import prepare_data_dict

    adata = adata.copy()

    if time_key is not None:
        if phylogeny is not None:
            A = _create_phylogeny_matrix(adata, phylogeny, cluster_key)
            connectivity = adata.obsp["connectivities"]
            adata.obsp["phylogeny_conn"] = connectivity.multiply(A)
            adata.obsp["prior_transition"] = prior_transition_matrix(
                adata, time_key, beta_t, obsp_key="phylogeny_conn"
            )
        else:
            adata.obsp["prior_transition"] = prior_transition_matrix(
                adata, time_key, beta_t, obsp_key="connectivities"
            )
    else:
        if phylogeny is not None:
            A = _create_phylogeny_matrix(adata, phylogeny, cluster_key)
            connectivity = adata.obsp["connectivities"]
            A_masked = connectivity.multiply(A)
            adata.obsp["phylogeny_conn"] = A_masked
            row_sums = np.array(A_masked.sum(axis=1)).ravel()
            valid_cells = row_sums > 0

            n_dead = np.sum(~valid_cells)
            if n_dead > 0:
                print(f"[LSD] Removing {n_dead} cells with no transitions")
                adata = adata[valid_cells]

            root = list(phylogeny.keys())[0]
            data_dict = prepare_data_dict(
                adata, n_top_genes=None, normalize=False, target_sum=1e4
            )
            temp = _infer_global_pseudotime(
                adata,
                phylogeny,
                root,
                device,
                cluster_key=cluster_key,
                n_epochs=5,
                random_state=random_state,
            )
            adata.obs["prior_pseudotime"] = temp.obs["prior_pseudotime"]
            adata.obsp["prior_transition"] = prior_transition_matrix(
                adata, "prior_pseudotime", beta_t, obsp_key="phylogeny_conn"
            )
        else:
            if root_cluster is None:
                raise ValueError("You should specify the root cluster.")

            phylo = infer_phylo(
                adata, root_cluster, cluster_key=cluster_key, ratio=ratio, hot_ratio=hot_ratio
            )
            A = _create_phylogeny_matrix(adata, phylo, cluster_key)
            connectivity = adata.obsp["connectivities"]
            adata.obsp["phylogeny_conn"] = connectivity.multiply(A)

            data_dict = prepare_data_dict(
                adata, n_top_genes=None, normalize=False, target_sum=1e4
            )
            temp = _infer_global_pseudotime(
                adata,
                phylo,
                root_cluster,
                device,
                cluster_key=cluster_key,
                n_epochs=5,
                random_state=random_state,
            )
            adata.obs["prior_pseudotime"] = temp.obs["prior_pseudotime"]
            adata.obsp["prior_transition"] = prior_transition_matrix(
                adata, "prior_pseudotime", beta_t, obsp_key="phylogeny_conn"
            )

    return adata, adata.obsp["prior_transition"].toarray()


def _shift_child_pseudotime(
    temp: "AnnData",
    phylo: Dict[str, List[str]],
    cluster_key: str = "clusters",
    pseudotime_key: str = "prior_pseudotime_mean",
) -> "AnnData":
    """Shift child pseudotime to ensure consistency."""
    visited = set()

    def adjust_children(parent: str) -> None:
        parent_mask = temp.obs[cluster_key] == parent
        parent_max = temp.obs.loc[parent_mask, pseudotime_key].max()

        for child in phylo.get(parent, []):
            if child in visited:
                continue

            child_mask = temp.obs[cluster_key] == child
            child_min = temp.obs.loc[child_mask, pseudotime_key].min()

            K = parent_max - child_min
            shift = np.maximum(0, K)

            if shift > 0:
                temp.obs.loc[child_mask, pseudotime_key] += shift

            visited.add(child)
            adjust_children(child)

    roots = [k for k in phylo if all(k not in v for v in phylo.values())]
    for root in roots:
        adjust_children(root)

    return temp


def _infer_global_pseudotime(
    adata: "AnnData",
    phylo: Dict[str, List[str]],
    root: str,
    device: torch.device,
    cluster_key: str = "clusters",
    n_epochs: int = 20,
    random_state: int = 42,
) -> "AnnData":
    """Infer global pseudotime across all branches."""
    from sclsd.preprocessing.data import prepare_data_dict

    main_adata = adata.copy()
    branches = _get_tree_branches(phylo, root)
    cols = []

    for branch in branches:
        branch_adata = adata[adata.obs[cluster_key].isin(branch)].copy()
        if "X_pca" in branch_adata.obsm:
            del branch_adata.obsm["X_pca"]
        if "connectivities" in branch_adata.obsp:
            del branch_adata.obsp["connectivities"]

        data_dict = prepare_data_dict(
            adata, n_top_genes=None, normalize=False, target_sum=1e4
        )

        root_cell = branch[0]
        branch_pseudotime_adata = infer_prior_time(
            data_dict, device, root_cell, n_epochs=n_epochs, random_state=random_state
        )

        main_adata.obs[f"prior_pseudotime_{branch[-1]}"] = None
        main_adata.obs.loc[
            adata.obs[cluster_key].isin(branch), f"prior_pseudotime_{branch[-1]}"
        ] = branch_pseudotime_adata.obs[f"prior_pseudotime_{root}"]
        cols.append(f"prior_pseudotime_{branch[-1]}")

    main_adata.obs["prior_pseudotime"] = main_adata.obs[cols].mean(axis=1, skipna=True)
    main_adata = _shift_child_pseudotime(
        main_adata, phylo, cluster_key=cluster_key, pseudotime_key="prior_pseudotime"
    )
    pseudotime = np.array(main_adata.obs["prior_pseudotime"], dtype=float)
    min_val = pseudotime.min()
    max_val = pseudotime.max()
    main_adata.obs["prior_pseudotime"] = (pseudotime - min_val) / (max_val - min_val)

    return main_adata
