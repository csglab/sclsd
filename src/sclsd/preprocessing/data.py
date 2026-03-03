"""Data preparation utilities for LSD models.

This module provides functions for preparing AnnData objects
for use with the LSD model.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    import scanpy as sc
    from anndata import AnnData
except ImportError:
    sc = None
    AnnData = None


def prepare_data_dict(
    adata: "AnnData",
    n_top_genes: Optional[int] = 5000,
    target_sum: float = 1e4,
    normalize: bool = True,
    log: bool = True,
    n_pcs: int = 50,
    use_rep: str = "X_pca",
    n_neighbors: int = 15,
    gene_selection: str = "hvg",
) -> Dict[str, Any]:
    """Prepare a dictionary of processed data from an AnnData object.

    This function preprocesses single-cell data for use with the LSD model,
    including gene selection, normalization, PCA, and neighbor graph construction.

    Parameters
    ----------
    adata : AnnData
        Input single-cell dataset.
    n_top_genes : int, optional
        Number of highly variable genes to retain (if gene_selection="hvg").
        Set to None to skip gene filtering.
    target_sum : float
        Target sum for total counts normalization.
    normalize : bool
        Whether to normalize counts.
    log : bool
        Whether to apply log1p transformation.
    n_pcs : int
        Number of principal components to compute.
    use_rep : str
        Key in adata.obsm to use for neighbors computation.
    n_neighbors : int
        Number of neighbors for graph construction.
    gene_selection : str
        Gene selection method: "hvg" for highly variable genes,
        or "custom" to use genes from adata.uns["selected_genes"].

    Returns
    -------
    dict
        Dictionary containing:
        - raw_counts: Raw count matrix
        - normal_counts: Normalized count matrix
        - librarysize: Library size per cell
        - adata: Processed AnnData object

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.read("data.h5ad")
    >>> data_dict = prepare_data_dict(adata, n_top_genes=2000)
    """
    if sc is None:
        raise ImportError("scanpy is required for data preparation")

    adata = adata.copy()
    adata.raw = adata

    # Gene selection
    if gene_selection == "hvg":
        if n_top_genes is not None:
            sc.pp.filter_genes_dispersion(adata, n_top_genes=n_top_genes)

    elif gene_selection == "custom":
        if "selected_genes" in adata.uns:
            selected_genes = list(
                set(adata.uns["selected_genes"]) & set(adata.var.index)
            )
            if len(selected_genes) == 0:
                raise ValueError(
                    "No matching genes found in adata.var.index for the custom gene list."
                )
            adata = adata[:, selected_genes].copy()
        else:
            raise KeyError("No 'selected_genes' found in adata.uns.")

    # Keep raw counts
    raw_counts = adata.X.copy()

    # Compute library size
    if hasattr(adata.X, "toarray"):
        adata.obs["librarysize"] = np.array(adata.X.sum(axis=1)).flatten()
    else:
        adata.obs["librarysize"] = adata.X.sum(axis=1)

    if normalize:
        # Normalize counts
        sc.pp.normalize_total(adata, target_sum=target_sum)

        # Log transformation
        if log:
            sc.pp.log1p(adata)

    # Compute PCA if not already present
    if "X_pca" not in adata.obsm.keys():
        sc.pp.pca(adata, n_comps=n_pcs)

    # Compute neighbors if not already present
    if "connectivities" not in adata.obsp.keys():
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)

    # Remove cells with zero connectivity
    row_sums = np.array(adata.obsp["connectivities"].sum(axis=1)).flatten()
    valid_cells = row_sums > 0
    n_invalid = np.sum(~valid_cells)
    if n_invalid > 0:
        print(f"[LSD] Removing {n_invalid} cells with zero connectivity")
        adata = adata[valid_cells].copy()

    # Convert to dense arrays
    if hasattr(raw_counts, "toarray"):
        raw_counts_dense = raw_counts.toarray()
    else:
        raw_counts_dense = np.array(raw_counts)

    if hasattr(adata.X, "toarray"):
        normal_counts_dense = adata.X.toarray()
    else:
        normal_counts_dense = np.array(adata.X)

    # Prepare output dictionary
    data_dict = {
        "raw_counts": raw_counts_dense,
        "normal_counts": normal_counts_dense,
        "librarysize": adata.obs["librarysize"].values.copy(),
        "adata": adata.copy(),
    }

    return data_dict


# Alias for backward compatibility
Prepare_DataDict = prepare_data_dict
