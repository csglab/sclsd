"""Random walk visualization utilities.

This module provides functions for visualizing random walks
on cell embeddings like UMAP and t-SNE.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    import scanpy as sc
    from anndata import AnnData
except ImportError:
    sns = None
    sc = None
    AnnData = None


def plot_random_walks(
    adata: "AnnData",
    walks: np.ndarray,
    rep: str = "X_umap",
    n_neighbors: int = 64,
    figsize: tuple = (10, 8),
    linewidth: float = 0.8,
    alpha: float = 0.6,
    markersize: int = 3,
) -> None:
    """Plot random walks on a cell embedding.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    walks : np.ndarray
        Array of random walks with cell indices.
    rep : str
        Embedding key in adata.obsm (e.g., "X_umap", "X_tsne").
    n_neighbors : int
        Number of neighbors (unused, kept for compatibility).
    figsize : tuple
        Figure size.
    linewidth : float
        Width of walk lines.
    alpha : float
        Transparency of walks.
    markersize : int
        Size of point markers.
    """
    if sns is None or sc is None:
        raise ImportError("seaborn and scanpy are required for plotting")

    coords = adata.obsm[rep]

    fig, ax = plt.subplots(figsize=figsize)

    if rep == "X_tsne":
        sc.pl.tsne(adata, ax=ax, show=False, title="Random Walks on t-SNE")
    elif rep == "X_umap":
        sc.pl.umap(adata, ax=ax, show=False, title="Random Walks on UMAP")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.3, c="gray")
        ax.set_title(f"Random Walks on {rep}")

    colors = sns.color_palette("husl", len(walks))

    labeled_start_end = False

    for idx, walk in enumerate(walks):
        walk_coords = coords[walk]

        ax.plot(
            walk_coords[:, 0],
            walk_coords[:, 1],
            "-",
            color=colors[idx],
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.plot(
            walk_coords[:, 0],
            walk_coords[:, 1],
            ".",
            color=colors[idx],
            markersize=markersize,
            alpha=alpha,
        )

        if not labeled_start_end:
            ax.plot(
                walk_coords[0, 0],
                walk_coords[0, 1],
                "o",
                color="green",
                markersize=10,
                label="Start",
            )
            ax.plot(
                walk_coords[-1, 0],
                walk_coords[-1, 1],
                "o",
                color="magenta",
                markersize=10,
                label="End",
            )
            labeled_start_end = True
        else:
            ax.plot(walk_coords[0, 0], walk_coords[0, 1], "o", color="green", markersize=10)
            ax.plot(
                walk_coords[-1, 0], walk_coords[-1, 1], "o", color="magenta", markersize=10
            )

    ax.legend()
    plt.tight_layout()
    plt.show()


def visualize_random_walks_on_umap(
    dyn_adata: "AnnData",
    paths,
    target_clusters: Union[str, List[str]],
    cluster_key: str = "clusters",
    n_walks: int = 10,
    figsize: tuple = (12, 8),
    alpha_walk: float = 0.7,
    linewidth: float = 1.5,
    seed: int = 42,
    rep: str = "X_umap",
    filename: Optional[str] = None,
    rasterize: bool = True,
    rasterization_zorder: int = 1,
    raster_dpi: int = 300,
) -> None:
    """Visualize random walks on UMAP starting from specific clusters.

    Parameters
    ----------
    dyn_adata : AnnData
        Annotated data object with embeddings.
    paths : torch.Tensor
        Paths tensor of shape (time_steps, n_trajectories).
    target_clusters : str or list
        Cluster(s) to visualize walks from.
    cluster_key : str
        Key in adata.obs for cluster labels.
    n_walks : int
        Number of walks to visualize.
    figsize : tuple
        Figure size.
    alpha_walk : float
        Transparency of walk lines.
    linewidth : float
        Width of walk lines.
    seed : int
        Random seed for reproducibility.
    rep : str
        Embedding key in adata.obsm.
    filename : str, optional
        Path to save figure.
    rasterize : bool
        Whether to rasterize plot elements.
    rasterization_zorder : int
        Z-order threshold for rasterization.
    raster_dpi : int
        DPI for rasterized elements.
    """
    if rep not in dyn_adata.obsm:
        raise KeyError(f"Embedding '{rep}' not found in dyn_adata.obsm")
    if cluster_key not in dyn_adata.obs:
        raise KeyError(f"Cluster key '{cluster_key}' not found in dyn_adata.obs")

    np.random.seed(seed)

    coords = dyn_adata.obsm[rep]
    if not isinstance(target_clusters, list):
        target_clusters = [target_clusters]
    clusters = dyn_adata.obs[cluster_key].astype(str)

    # Colors
    if f"{cluster_key}_colors" in dyn_adata.uns:
        cluster_colors = dyn_adata.uns[f"{cluster_key}_colors"]
        unique_clusters = dyn_adata.obs[cluster_key].cat.categories.astype(str)
    else:
        unique_clusters = clusters.unique()
        cluster_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    # Target cells
    target_cells = np.concatenate(
        [np.where(clusters == str(tc))[0] for tc in target_clusters]
    ) if len(target_clusters) else np.array([], dtype=int)

    if target_cells.size == 0:
        raise ValueError(f"No cells found in clusters {target_clusters}")

    # Walks that start in target cells
    valid_walks = [
        j for j in range(paths.shape[1]) if paths[0, j].item() in target_cells
    ]
    if not valid_walks:
        raise ValueError(f"No walks start from clusters {target_clusters}")

    n_walks = min(n_walks, len(valid_walks))
    selected_walks = np.random.choice(valid_walks, size=n_walks, replace=False)

    fig, ax = plt.subplots(figsize=figsize)

    if rasterize:
        ax.set_rasterization_zorder(rasterization_zorder)

    legend_handles, legend_labels = [], []

    for i, cl in enumerate(unique_clusters):
        mask = clusters == cl
        color = cluster_colors[i] if isinstance(cluster_colors[i], str) else cluster_colors[i]
        scatter = ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color],
            s=10,
            alpha=0.6,
            label=f"{cl}",
            zorder=1,
        )
        if rasterize:
            scatter.set_rasterized(True)
        legend_handles.append(scatter)
        legend_labels.append(cl)

    walk_colors = plt.cm.plasma(np.linspace(0, 1, n_walks))
    for i, widx in enumerate(selected_walks):
        walk_path = paths[:, widx].detach().cpu().numpy()
        walk_xy = coords[walk_path]

        line = ax.plot(
            walk_xy[:, 0],
            walk_xy[:, 1],
            color=walk_colors[i],
            alpha=alpha_walk,
            linewidth=linewidth,
            zorder=10,
        )[0]
        if rasterize:
            line.set_rasterized(True)

        ax.scatter(
            walk_xy[0, 0],
            walk_xy[0, 1],
            color="black",
            s=80,
            marker="o",
            edgecolor="white",
            linewidth=1,
            zorder=1.5,
        )
        ax.scatter(
            walk_xy[-1, 0],
            walk_xy[-1, 1],
            color="gold",
            s=80,
            marker="s",
            edgecolor="black",
            linewidth=1,
            zorder=1.5,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Sample Discretized Cell State Evolution", fontsize=14)

    ax.legend(
        legend_handles,
        legend_labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=raster_dpi, bbox_inches="tight", transparent=True)
    plt.show()

    print("Visualization Summary:")
    print(f"- Total cells in dataset: {dyn_adata.n_obs}")
    print(f"- Cells in target clusters {target_clusters}: {len(target_cells)}")
    print(f"- Total walks available: {paths.shape[1]}")
    print(f"- Walks starting from target clusters: {len(valid_walks)}")
    print(f"- Walks visualized: {n_walks}")
    print(f"- Steps per walk: {paths.shape[0]}")
