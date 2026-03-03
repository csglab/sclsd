"""Latent component visualization utilities.

This module provides functions for visualizing ODE solution components
and latent space trajectories.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_z_components(
    z_sol_subset: Union[torch.Tensor, np.ndarray],
    t_max: float,
    save_path: Optional[str] = None,
    title: str = "ODE Solution Components in Cell State Space",
    subtitle: str = "(Selected Trajectories)",
    n_cols: int = 2,
    fig_size: tuple = (12, 14),
    cmap: str = "tab20",
    xlabel: str = "ODE Time Unit",
    ylabel: str = "Value",
    dpi: int = 300,
    bbox_inches: str = "tight",
    facecolor: str = "white",
    edgecolor: str = "none",
) -> None:
    """Plot each latent component across multiple trajectories.

    Parameters
    ----------
    z_sol_subset : torch.Tensor or np.ndarray
        Array of shape (T, N, D): T time points, N trajectories, D components.
    t_max : float
        Upper bound of the time axis.
    save_path : str, optional
        Path to save figure.
    title : str
        Supertitle for the figure.
    subtitle : str
        Subtitle shown under the supertitle.
    n_cols : int
        Number of subplot columns.
    fig_size : tuple
        Figure size (width, height).
    cmap : str
        Matplotlib colormap name.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    dpi : int
        DPI for saved figure.
    bbox_inches : str
        Bounding box for saved figure.
    facecolor : str
        Figure background color.
    edgecolor : str
        Figure edge color.

    Examples
    --------
    >>> plot_z_components(z_sol[:, :10, :], t_max=5.0, save_path="components.png")
    """
    # Convert to NumPy
    if torch.is_tensor(z_sol_subset):
        z_arr = z_sol_subset.cpu().numpy()
    else:
        z_arr = np.array(z_sol_subset)

    T, N, D = z_arr.shape
    t = np.linspace(0, t_max, T)

    n_rows = int(np.ceil(D / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size, sharex="col")
    axes = axes.flatten()

    colors = plt.get_cmap(cmap)(np.linspace(0, 1, N))

    for i in range(D):
        ax = axes[i]
        for j in range(N):
            ax.plot(t, z_arr[:, j, i], color=colors[j], alpha=1.0, linewidth=2)
        ax.set_title(f"Component {i+1}", fontsize=14, pad=10)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle="--")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(0.8)

        row_idx = i // n_cols
        if row_idx == n_rows - 1:
            ax.set_xlabel(xlabel, fontsize=12)
            ax.tick_params(axis="x", labelsize=10)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(axis="y", labelsize=10)

    # Turn off unused subplots
    for empty_ax in axes[D:]:
        fig.delaxes(empty_ax)

    plt.suptitle(title, fontsize=16, y=0.98)
    plt.title(subtitle, fontsize=12, y=0.94)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    fig.patch.set_facecolor(facecolor)

    if save_path:
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches=bbox_inches,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )
    plt.show()
