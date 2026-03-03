"""Streamline visualization for vector fields.

This module provides functions for visualizing vector fields
as streamlines on cell embeddings.

.. deprecated::
    The `plot_streamlines` function is deprecated and will be removed in a future version.
    Use :meth:`sclsd.LSD.stream_lines` instead for proper CellRank-based streamline visualization.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from anndata import AnnData
except ImportError:
    AnnData = None


def plot_streamlines(
    adata: "AnnData",
    velocity_key: str = "velocity",
    embedding_key: str = "X_umap",
    color_key: Optional[str] = None,
    density: float = 1.0,
    linewidth: float = 1.0,
    arrowsize: float = 1.0,
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 150,
    n_grid: int = 50,
    smooth: float = 0.5,
    min_mass: float = 1.0,
) -> None:
    """Plot velocity streamlines on a cell embedding.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with velocity and embedding.
    velocity_key : str
        Key prefix for velocity in adata.obsm (e.g., "velocity" looks for "velocity_umap").
    embedding_key : str
        Key for embedding in adata.obsm.
    color_key : str, optional
        Key in adata.obs for coloring points.
    density : float
        Density of streamlines.
    linewidth : float
        Width of streamlines.
    arrowsize : float
        Size of arrowheads.
    figsize : tuple
        Figure size.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.
    dpi : int
        DPI for saved figure.
    n_grid : int
        Grid resolution for streamlines.
    smooth : float
        Smoothing factor for velocity field.
    min_mass : float
        Minimum cell density for streamlines.

    Examples
    --------
    >>> plot_streamlines(adata, velocity_key="velocity", embedding_key="X_umap")

    .. deprecated::
        `plot_streamlines` is deprecated and will be removed in a future version.
        Use `lsd.stream_lines()` instead for proper CellRank-based visualization.
    """
    warnings.warn(
        "plot_streamlines is deprecated and will be removed in a future version. "
        "Use lsd.stream_lines() instead for proper CellRank-based streamline visualization.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Get embedding coordinates
    if embedding_key not in adata.obsm:
        raise KeyError(f"Embedding '{embedding_key}' not found in adata.obsm")

    X_emb = adata.obsm[embedding_key]

    # Determine velocity embedding key
    if embedding_key == "X_umap":
        vel_emb_key = f"{velocity_key}_umap"
    elif embedding_key == "X_tsne":
        vel_emb_key = f"{velocity_key}_tsne"
    else:
        vel_emb_key = f"{velocity_key}_{embedding_key.replace('X_', '')}"

    if vel_emb_key not in adata.obsm:
        raise KeyError(
            f"Velocity embedding '{vel_emb_key}' not found in adata.obsm. "
            "Please project velocities to the embedding first."
        )

    V_emb = adata.obsm[vel_emb_key]

    # Create grid for streamlines
    x_min, x_max = X_emb[:, 0].min(), X_emb[:, 0].max()
    y_min, y_max = X_emb[:, 1].min(), X_emb[:, 1].max()

    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range

    # Grid coordinates
    xi = np.linspace(x_min, x_max, n_grid)
    yi = np.linspace(y_min, y_max, n_grid)
    Xi, Yi = np.meshgrid(xi, yi)

    # Interpolate velocities to grid
    from scipy.interpolate import griddata

    # Grid velocities using weighted average
    Vx = griddata(X_emb, V_emb[:, 0], (Xi, Yi), method="linear", fill_value=0)
    Vy = griddata(X_emb, V_emb[:, 1], (Xi, Yi), method="linear", fill_value=0)

    # Calculate cell density on grid for masking
    from scipy.stats import gaussian_kde

    try:
        kernel = gaussian_kde(X_emb.T)
        positions = np.vstack([Xi.ravel(), Yi.ravel()])
        density_grid = kernel(positions).reshape(Xi.shape)
    except np.linalg.LinAlgError:
        density_grid = np.ones_like(Xi)

    # Mask low-density regions
    mask = density_grid < (density_grid.max() * 0.01 * min_mass)
    Vx[mask] = np.nan
    Vy[mask] = np.nan

    # Apply smoothing
    if smooth > 0:
        from scipy.ndimage import gaussian_filter

        Vx = gaussian_filter(Vx, sigma=smooth, mode="constant", cval=0)
        Vy = gaussian_filter(Vy, sigma=smooth, mode="constant", cval=0)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot cells as background
    if color_key is not None and color_key in adata.obs:
        categories = adata.obs[color_key]
        if hasattr(categories, "cat"):
            for cat in categories.cat.categories:
                mask = categories == cat
                ax.scatter(
                    X_emb[mask, 0],
                    X_emb[mask, 1],
                    s=10,
                    alpha=0.5,
                    label=cat,
                )
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            scatter = ax.scatter(
                X_emb[:, 0], X_emb[:, 1], c=categories, s=10, alpha=0.5, cmap="viridis"
            )
            plt.colorbar(scatter, ax=ax, label=color_key)
    else:
        ax.scatter(X_emb[:, 0], X_emb[:, 1], s=10, alpha=0.3, c="gray")

    # Plot streamlines
    speed = np.sqrt(Vx**2 + Vy**2)
    lw = linewidth * speed / speed[~np.isnan(speed)].max()

    ax.streamplot(
        xi,
        yi,
        Vx,
        Vy,
        density=density,
        linewidth=lw,
        arrowsize=arrowsize,
        color="black",
        arrowstyle="->",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title("Velocity Streamlines", fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
