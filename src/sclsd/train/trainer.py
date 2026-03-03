"""Main LSD trainer class for model training and inference.

This module provides the LSD class which is the main entry point
for training and using the Latent State Dynamics model.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset
from torchdiffeq import odeint
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

import pyro
from pyro.optim import CosineAnnealingWarmRestarts
from pyro.infer import SVI, TraceEnum_ELBO

from sclsd.core.config import LSDConfig, WalkConfig
from sclsd.core.model import LSDModel
from sclsd.utils.seed import set_all_seeds, clear_pyro_state, enable_pyro_validation

try:
    import scanpy as sc
    from anndata import AnnData
except ImportError:
    AnnData = None


class LSD:
    """Latent State Dynamics (LSD) model for single-cell trajectory inference.

    This class provides the main interface for training and using the LSD model
    to infer cell differentiation trajectories from single-cell data.

    Parameters
    ----------
    adata : AnnData
        Preprocessed single-cell AnnData object. Should contain log-normalized
        counts in `adata.X`.
    config : LSDConfig, optional
        Model configuration. If None, uses default configuration.
    device : torch.device
        Training device (CPU or CUDA).
    lib_size_key : str
        Key for library size column in `adata.obs`.
    raw_count_key : str
        Key for raw count data in `adata.layers`.

    Attributes
    ----------
    lsd : LSDModel
        The underlying neural network model.
    adata : AnnData
        The single-cell data.
    walks : torch.Tensor
        Generated random walks for training.
    P : torch.Tensor
        Cell-cell transition probability matrix.

    Examples
    --------
    >>> from sclsd import LSD, LSDConfig
    >>> cfg = LSDConfig()
    >>> cfg.walks.path_len = 50
    >>> lsd = LSD(adata, cfg, device=torch.device("cuda"))
    >>> lsd.set_prior_transition(prior_time_key="pseudotime")
    >>> lsd.prepare_walks()
    >>> lsd.train(num_epochs=100)
    >>> result = lsd.get_adata()
    """

    def __init__(
        self,
        adata: "AnnData",
        config: Optional[LSDConfig] = None,
        *,
        device: torch.device = torch.device("cuda"),
        lib_size_key: str = "librarysize",
        raw_count_key: str = "raw",
    ):
        self.config = config if isinstance(config, LSDConfig) else LSDConfig()
        model_cfg = self.config.model
        walk_cfg = replace(self.config.walks)
        opt_cfg = self.config.optimizer

        layer_dims = model_cfg.layer_dims.as_dict()
        z_dim = model_cfg.z_dim
        B_dim = model_cfg.B_dim
        batch_size = walk_cfg.batch_size
        path_len = walk_cfg.path_len
        V_coeff = model_cfg.V_coeff

        self.optim_args = opt_cfg.adam.as_dict()
        self.KL_args = opt_cfg.kl_schedule.as_dict()
        self.W_args = opt_cfg.wasserstein_schedule.as_dict()

        self.walk_config = walk_cfg

        # AnnData parameters
        num_genes = len(adata.var)
        lib_size = adata.obs[lib_size_key]
        self.xl_loc = lib_size.mean()
        self.xl_scale = lib_size.std()

        self.adata = adata.copy()
        self.lib_size_key = lib_size_key
        self.raw_count_key = raw_count_key
        self.cluster_key = None

        # Model parameters
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.B_dim = B_dim
        self.device = device
        self.epoch = 0
        self.path_len = path_len

        # Initialization params
        self.P = None
        self.walks = None
        self.phylogeny = None

        # Cell fates
        self.paths = None
        self.z_sol = None
        self.fates = None

        # Create model
        self.lsd = LSDModel(
            B_dim=B_dim,
            z_dim=z_dim,
            num_genes=num_genes,
            layer_dims=layer_dims,
            batch_size=batch_size,
            path_len=path_len,
            device=device,
            scale_factor=1.0 / (batch_size * path_len * num_genes),
            V_coeff=V_coeff,
            xl_loc=self.xl_loc,
            xl_scale=self.xl_scale,
        )
        self.lsd.to(self.device)

    def prepare_datadict(self) -> Dict[str, Any]:
        """Prepare dictionary with arrays and metadata for training.

        Returns
        -------
        dict
            Dictionary containing raw_counts, normal_counts, librarysize, and adata.
        """
        data_dict = {
            "raw_counts": self.adata.layers[self.raw_count_key].toarray(),
            "normal_counts": self.adata.X.copy().toarray(),
            "librarysize": self.adata.obs[self.lib_size_key].copy().values,
            "adata": self.adata.copy(),
        }
        return data_dict

    def _calculate_annealing_factor(self) -> float:
        """Calculate annealing factor for KL term."""
        min_af = self.KL_args["min_af"]
        max_af = self.KL_args["max_af"]
        max_epoch = self.KL_args["max_epoch"]
        if self.epoch < max_epoch:
            af = min_af + (max_af - min_af) * self.epoch / max_epoch
        else:
            af = max_af
        return af

    def _calculate_W_factor(self) -> float:
        """Calculate Wasserstein regularization factor."""
        min_W = self.W_args["min_W"]
        max_W = self.W_args["max_W"]
        max_epoch = self.W_args["max_epoch"]
        if self.epoch < max_epoch:
            W = max_W + (min_W - max_W) * self.epoch / max_epoch
        else:
            W = min_W
        return W

    def _V(self, x: torch.Tensor) -> float:
        """Monitor potential monotonicity."""
        self.lsd.eval()
        with torch.no_grad():
            loc, _ = self.lsd.x_encoder(x)
            V = self.lsd.potential(loc)
            dim = V.shape[-1]
            V = V.reshape(int(len(x) / self.path_len), self.path_len, dim)
            S = 0
            for i in range(1, self.path_len):
                term1 = V[:, i, :]
                term2 = V[:, i - 1, :]
                gate = F.relu(term1 - term2)
                S += torch.mean(gate) / self.path_len
        return S.detach().cpu().numpy()

    def _H(self, x: torch.Tensor) -> float:
        """Monitor entropy."""
        self.lsd.eval()
        with torch.no_grad():
            loc, _ = self.lsd.x_encoder(x)
            _, sigma = self.lsd.z_encoder(loc)
            dim = sigma.shape[-1]
            sigma = sigma.reshape(int(len(x) / self.path_len), self.path_len, dim)
            S = 0
            for i in range(1, self.path_len):
                term1 = 0.5 * torch.log(sigma[:, i, :] ** 2).sum(axis=-1)
                term2 = 0.5 * torch.log(sigma[:, i - 1, :] ** 2).sum(axis=-1)
                gate = F.relu(term1 - term2)
                S += torch.mean(gate) / self.path_len
        return S.detach().cpu().numpy()

    def z_rec_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction loss of z from B."""
        self.lsd.eval()
        with torch.no_grad():
            z, _ = self.lsd.x_encoder(x)
            b, _ = self.lsd.z_encoder(z)
            z_hat, var = self.lsd.B_decoder(b)
            loss = torch.norm((z_hat - z), p=2, dim=-1)
        return torch.mean(loss)

    def prepare_dataset(
        self,
        walks: torch.Tensor,
        data_dict: Dict[str, Any],
    ) -> TensorDataset:
        """Prepare batches from random walks.

        Parameters
        ----------
        walks : torch.Tensor
            Random walk indices of shape (n_walks, path_len).
        data_dict : dict
            Data dictionary from prepare_datadict().

        Returns
        -------
        TensorDataset
            Dataset containing (x_raw, x, xl) for each walk.
        """
        x = torch.from_numpy(data_dict["normal_counts"]).type(torch.float32)
        x_raw = torch.from_numpy(data_dict["raw_counts"]).type(torch.float32)
        xl = torch.from_numpy(data_dict["librarysize"]).unsqueeze(-1)
        x = x[walks]
        x_raw = x_raw[walks]
        xl = xl[walks].type(torch.float32)
        dataset = TensorDataset(x_raw, x, xl)
        return dataset

    def train(
        self,
        num_epochs: int = 60,
        save_dir: Optional[str] = None,
        save_interval: int = 50,
        plot_loss: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """Train the LSD model.

        Parameters
        ----------
        num_epochs : int
            Number of training epochs.
        save_dir : str, optional
            Directory to save checkpoints.
        save_interval : int
            Save checkpoint every N epochs.
        plot_loss : bool
            Whether to plot loss curves.
        random_state : int, optional
            Random seed for reproducibility.
        """
        data_dict = self.prepare_datadict()
        adata = data_dict["adata"].copy()

        if save_dir is not None:
            adata_dir = save_dir + "/adata"
            os.makedirs(adata_dir, exist_ok=True)
            adata_name = "training_adata.h5ad"
            adata.write(os.path.join(adata_dir, adata_name))

        if random_state is None:
            random_state = getattr(self.walk_config, "random_state", 42)

        # Set all seeds for reproducibility
        set_all_seeds(random_state)
        enable_pyro_validation(True)

        self.lsd = self.lsd.to(self.device)
        scheduler = CosineAnnealingWarmRestarts(
            {
                "optimizer": Adam,
                "optim_args": {"lr": self.optim_args["lr"]},
                "T_0": self.optim_args["T_0"],
                "eta_min": self.optim_args["eta_min"],
                "T_mult": self.optim_args["T_mult"],
            },
            {"clip_norm": 10.0},
        )
        elbo = TraceEnum_ELBO(strict_enumeration_warning=False)
        svi = SVI(self.lsd.model, self.lsd.guide, scheduler, elbo)

        ELBO_losses = []
        V_losses = []
        H_losses = []

        epoch_bar = trange(num_epochs, desc="Training Epochs")
        for epoch in epoch_bar:
            epoch_losses = []
            V_loss = []
            H_loss = []

            # Shuffle walks each epoch
            shuffled_walks = self.walks[torch.randperm(self.walks.size(0))]
            dataset = self.prepare_dataset(shuffled_walks, data_dict)

            for i in tqdm(
                range(0, len(dataset), self.batch_size),
                desc=f"Epoch {epoch}",
                leave=False,
            ):
                self.lsd.train()
                batch = dataset[i : (i + self.batch_size)]

                x_raw, x, xl = batch
                x_raw = x_raw.to(self.device)
                x = x.to(self.device)
                xl = xl.to(self.device).unsqueeze(-1).reshape(-1, 1)

                batch_size, path_len, x_dim = x.shape
                assert self.path_len == path_len, (
                    f"path_len mismatch: LSD was initialized with path_len={self.path_len}, "
                    f"but got path_len={path_len} in the dataloader."
                )
                x_raw, x = x_raw.reshape(-1, x_dim), x.reshape(-1, x_dim)

                annealing_factor = self._calculate_annealing_factor()
                W_coeff = self._calculate_W_factor()

                loss = svi.step(
                    x_raw, x, xl=None, annealing_factor=annealing_factor, W_coeff=W_coeff
                )
                epoch_losses.append(loss)
                V_loss.append(self._V(x))
                H_loss.append(self._H(x))

                epoch_bar.set_postfix(
                    {
                        "-ELBO": f"{epoch_losses[-1]:.4f}",
                        "Potential Metric": f"{V_loss[-1]:.4f}",
                        "Entropy Metric": f"{H_loss[-1]:.4f}",
                    }
                )

            epoch_loss_mean = np.mean(epoch_losses)
            ELBO_losses.append(epoch_loss_mean)
            V_loss_mean = np.mean(V_loss)
            V_losses.append(V_loss_mean)
            H_loss_mean = np.mean(H_loss)
            H_losses.append(H_loss_mean)

            if save_dir is not None:
                if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
                    checkpoint_name = f"lsd_model_epoch{epoch+1:04d}.pth"
                    self.save(dir_path=save_dir, file_name=checkpoint_name)

            self.epoch += 1

        if plot_loss and save_dir is not None:
            fig, axs = plt.subplots(1, 3, figsize=(12, 5))

            axs[0].plot(ELBO_losses, label="ELBO Loss")
            axs[0].set_title("ELBO Loss over Epochs")
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("ELBO Loss")
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(V_losses, label="V Loss", color="orange")
            axs[1].set_title("V Loss over Epochs")
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("V Loss")
            axs[1].legend()
            axs[1].grid(True)

            axs[2].plot(H_losses, label="H Loss", color="red")
            axs[2].set_title("H difference over Epochs")
            axs[2].set_xlabel("Epoch")
            axs[2].set_ylabel("H difference")
            axs[2].legend()
            axs[2].grid(True)

            plt.tight_layout()
            out_path = os.path.join(save_dir, "loss_curves.png")
            fig.savefig(out_path)
            plt.show()

    def save(
        self,
        dir_path: str = "checkpoints",
        file_name: str = "model_and_params.pth",
    ) -> None:
        """Save model checkpoint.

        Parameters
        ----------
        dir_path : str
            Directory to save checkpoint.
        file_name : str
            Filename for checkpoint.
        """
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, file_name)
        torch.save(
            {
                "model_state_dict": self.lsd.state_dict(),
                "pyro_params": pyro.get_param_store().get_state(),
            },
            file_path,
        )

    def load(
        self,
        dir_path: str = "checkpoints",
        file_name: str = "model_and_params.pth",
    ) -> "LSD":
        """Load model checkpoint.

        Parameters
        ----------
        dir_path : str
            Directory containing checkpoint.
        file_name : str
            Filename of checkpoint.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        file_path = os.path.join(dir_path, file_name)
        # weights_only=False required for PyTorch 2.6+ to load Pyro state
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)

        self.lsd.load_state_dict(checkpoint["model_state_dict"])
        pyro.get_param_store().set_state(checkpoint["pyro_params"])

        print(f"[LSD] Model and Pyro parameters loaded from {file_path}")
        return self

    def get_variables(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute latent variables and metrics.

        Parameters
        ----------
        x : torch.Tensor
            Input expression data.

        Returns
        -------
        B_loc : torch.Tensor
            Differentiation state representation.
        z_loc : torch.Tensor
            Cell state representation.
        entropy : torch.Tensor
            Entropy of cell state.
        potential : torch.Tensor
            Potential value for each cell.
        pseudotime : torch.Tensor
            Normalized pseudotime based on potential.
        """
        x = x.to(self.device)
        self.lsd.eval()
        with torch.no_grad():
            z_loc, _ = self.lsd.x_encoder(x)
            B_loc, B_scale = self.lsd.z_encoder(z_loc)
            entropy = torch.log(B_scale).sum(dim=1)
            potential = self.lsd.potential(z_loc)
            max_potential = torch.max(potential)
            pseudotime = max_potential - potential
            pseudotime = (pseudotime - pseudotime.min()) / (
                pseudotime.max() - pseudotime.min() + 1e-8
            )
        return B_loc, z_loc, entropy, potential, pseudotime

    def calculate_transition_probs(
        self,
        potential: np.ndarray,
        connectivity_matrix: np.ndarray,
        beta: float = 1.0,
    ) -> np.ndarray:
        """Compute cell-cell transition probabilities using Boltzmann weights.

        Parameters
        ----------
        potential : np.ndarray
            Potential values of shape (n_cells,).
        connectivity_matrix : np.ndarray
            Binary connectivity matrix of shape (n_cells, n_cells).
        beta : float
            Boltzmann scaling factor.

        Returns
        -------
        np.ndarray
            Transition probability matrix, row-normalized.
        """
        potential = potential.astype(float)
        energy_diff = potential[None, :] - potential[:, None]
        boltzmann_weights = np.exp(-beta * energy_diff)
        boltzmann_weights *= connectivity_matrix
        row_sums = boltzmann_weights.sum(axis=1, keepdims=True) + 1e-12
        transition_matrix = boltzmann_weights / row_sums
        return transition_matrix

    def set_adata(self, adata: "AnnData") -> None:
        """Set the AnnData object."""
        self.adata = adata

    def get_adata(self) -> "AnnData":
        """Get annotated AnnData with LSD results.

        Returns
        -------
        AnnData
            AnnData with added observations and embeddings.
        """
        adata = self.adata.copy()
        x = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        B_loc, z_loc, entropy, potential, pseudotime = self.get_variables(
            torch.from_numpy(x).float()
        )

        adata.obsm["X_cell_state"] = z_loc.cpu().numpy()
        adata.obsm["X_diff_state"] = B_loc.cpu().numpy()
        adata.obs["entropy"] = entropy.cpu().numpy()
        adata.obs["potential"] = potential.cpu().numpy()
        adata.obs["lsd_pseudotime"] = pseudotime.cpu().numpy()

        if "connectivities" not in adata.obsp:
            raise KeyError(
                "'connectivities' matrix not found in adata.obsp. "
                "Run neighbors graph computation."
            )
        connectivity = adata.obsp["connectivities"]
        if not isinstance(connectivity, np.ndarray):
            connectivity = connectivity.toarray()
        binary_connectivity = (connectivity > 0).astype(float)

        transition_matrix = self.calculate_transition_probs(
            potential=potential.squeeze(-1).cpu().numpy(),
            connectivity_matrix=binary_connectivity,
            beta=1.0,
        )

        adata.obsp["transitions"] = transition_matrix
        return adata

    def _random_walks(self, n_trajectories: int) -> torch.Tensor:
        """Generate random walks in parallel on GPU."""
        n_cells = self.P.shape[0]
        current_states = torch.randint(
            0, n_cells, (n_trajectories,), device=self.device, dtype=torch.long
        )
        walks = torch.empty(
            (n_trajectories, self.path_len), dtype=torch.int, device=self.device
        )
        walks[:, 0] = current_states

        for step in range(1, self.path_len):
            next_states = torch.multinomial(
                self.P[current_states.to(self.device)], num_samples=1
            ).squeeze(1)
            walks[:, step] = next_states
            current_states = next_states

        return walks

    def prepare_walks(self, n_trajectories: Optional[int] = None) -> None:
        """Generate random walks for training.

        Parameters
        ----------
        n_trajectories : int, optional
            Number of walks to generate. Uses config if not provided.
        """
        if n_trajectories is None:
            if self.walk_config is None:
                raise ValueError(
                    "Number of trajectories must be provided when no walk config is set."
                )
            n_trajectories = self.walk_config.num_walks

        self.P = self.P.to(self.device)
        walks = self._random_walks(n_trajectories)
        self.P = self.P.cpu()
        self.walks = walks.cpu()

    def set_prior_transition(
        self,
        prior_time_key: Optional[str] = None,
        prior_transition: Optional[np.ndarray] = None,
        random_state: int = 42,
    ) -> None:
        """Set the prior cell-cell transition matrix.

        Parameters
        ----------
        prior_time_key : str, optional
            Name of pseudotime key in adata.obs.
        prior_transition : np.ndarray, optional
            Precomputed prior transition matrix.
        random_state : int
            Random seed.
        """
        n_cells = len(self.adata)

        def _get_connectivity_matrix():
            if "connectivities" not in self.adata.obsp:
                raise KeyError(
                    "'connectivities' matrix not found in adata.obsp. "
                    "Run neighbors graph computation (e.g. sc.pp.neighbors)."
                )
            mat = self.adata.obsp["connectivities"]
            if not isinstance(mat, np.ndarray):
                mat = mat.toarray()
            return (mat > 0).astype(float)

        if prior_transition is not None:
            if not isinstance(prior_transition, np.ndarray):
                prior_transition = prior_transition.toarray()
            if prior_transition.shape != (n_cells, n_cells):
                raise ValueError(
                    f"Shape mismatch: prior_transition has shape {prior_transition.shape}, "
                    f"but expected ({n_cells}, {n_cells}) from adata."
                )
            self.P = torch.from_numpy(prior_transition).float()
            print("[LSD] Prior transition matrix set from user input.")
            return

        if self.phylogeny is not None:
            A = self._create_phylogeny_matrix()
            if not isinstance(A, np.ndarray):
                A = A.toarray()
            connectivity = _get_connectivity_matrix()
            A *= connectivity

            self.adata.obsp["phylogeny_matrix"] = A
            row_sums = A.sum(axis=1)
            valid_cells = row_sums > 0
            if len(self.adata[~valid_cells]) != 0:
                print(
                    f"[LSD] Removing {np.sum(~valid_cells)} cells with no transitions:"
                )
            self.adata = self.adata[valid_cells]
            if prior_time_key is not None:
                P = self._get_transition_from_pseudotime(
                    prior_time_key, self.adata.obsp["phylogeny_matrix"].toarray()
                )
                print("[LSD] Prior transition matrix set from phylogeny and pseudotime.")
            else:
                raise KeyError("Run the function get_prior_transition first")
            self.P = torch.from_numpy(P).float()
            return

        if prior_time_key is not None:
            connectivity = _get_connectivity_matrix()
            P = self._get_transition_from_pseudotime(prior_time_key, connectivity)
            self.P = torch.from_numpy(P).float()
            print("[LSD] Prior transition matrix set from pseudotime and connectivities.")
            return

        raise KeyError(
            "LSD requires either a prior pseudotime, a phylogeny, "
            "or a prior transition matrix for initialization."
        )

    def _get_transition_from_pseudotime(
        self,
        time_key: str,
        connectivity: np.ndarray,
    ) -> np.ndarray:
        """Compute transition matrix from pseudotime."""
        potential = -self.adata.obs[time_key].values
        P = self.calculate_transition_probs(potential, connectivity, beta=50)
        return P

    def _get_all_descendants(
        self,
        cluster: str,
        descendants: Optional[set] = None,
    ) -> set:
        """Recursively find all descendants of a cluster."""
        if descendants is None:
            descendants = set()
        direct_children = self.phylogeny.get(cluster, [])
        descendants.update(direct_children)
        return descendants

    def set_phylogeny(self, phylogeny: Dict[str, List[str]], cluster_key: str) -> None:
        """Set the phylogeny for the model.

        Parameters
        ----------
        phylogeny : dict
            Dictionary with format {parent: [child1, child2, ...]}.
        cluster_key : str
            Key in adata.obs for cluster labels.
        """
        self.phylogeny = phylogeny
        self.cluster_key = cluster_key

    def _create_phylogeny_matrix(self) -> sp.csr_matrix:
        """Create phylogeny-based adjacency matrix."""
        clusters = self.adata.obs[self.cluster_key].unique().tolist()
        cell_to_cluster = dict(
            zip(self.adata.obs_names, self.adata.obs[self.cluster_key])
        )

        all_descendants = {}
        for cluster in clusters:
            all_descendants[cluster] = self._get_all_descendants(cluster)

        n_cells = self.adata.shape[0]
        phylo_matrix = np.zeros((n_cells, n_cells))

        for i, cell_i in enumerate(self.adata.obs_names):
            cluster_i = cell_to_cluster[cell_i]
            for j, cell_j in enumerate(self.adata.obs_names):
                cluster_j = cell_to_cluster[cell_j]
                if cluster_i == cluster_j:
                    phylo_matrix[i, j] = 1
                elif cluster_j in all_descendants.get(cluster_i, set()):
                    phylo_matrix[i, j] = 1

        return sp.csr_matrix(phylo_matrix)

    def ode_solve(
        self,
        z0: torch.Tensor,
        time_range: float,
        num_points: Optional[int] = None,
    ) -> torch.Tensor:
        """Solve ODE for trajectory propagation.

        Parameters
        ----------
        z0 : torch.Tensor
            Initial latent state.
        time_range : float
            Time range for integration.
        num_points : int, optional
            Number of time points.

        Returns
        -------
        torch.Tensor
            ODE solution trajectory.
        """
        t = torch.linspace(0, time_range, num_points).to(self.device)
        z = odeint(self.lsd.gradnet, z0, t)
        return z.detach()

    def propagate(
        self,
        x: torch.Tensor,
        time_range: float,
        num_points: Optional[int] = None,
    ) -> torch.Tensor:
        """Propagate cells through the potential landscape.

        Parameters
        ----------
        x : torch.Tensor
            Input expression data.
        time_range : float
            Time range for propagation.
        num_points : int, optional
            Number of time points.

        Returns
        -------
        torch.Tensor
            Trajectory through latent space.
        """
        x0 = x.to(self.device)
        z0, _ = self.lsd.x_encoder(x0)
        if num_points is None:
            num_points = 2 * int(time_range)
        z = self.ode_solve(z0, time_range, num_points=num_points)
        return z

    def project_z(
        self,
        z: torch.Tensor,
        adata: "AnnData",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project latent points to nearest cells.

        Parameters
        ----------
        z : torch.Tensor
            Latent positions.
        adata : AnnData
            Reference data.

        Returns
        -------
        z_end : torch.Tensor
            Projected latent positions.
        x_proj : torch.Tensor
            Corresponding expression profiles.
        nn : torch.Tensor
            Nearest neighbor indices.
        """
        latent_rep = torch.tensor(
            adata.obsm["X_cell_state"], dtype=torch.float32
        ).to(self.device)
        X = torch.tensor(adata.X.toarray(), dtype=torch.float32).to(self.device)
        nn = torch.argmin(torch.cdist(z, latent_rep), dim=-1).to(self.device)
        x_proj = X[nn, :]
        z_end = latent_rep[nn, :]
        return z_end, x_proj, nn

    def get_cell_fates(
        self,
        adata: "AnnData",
        time_range: float,
        dt: float = 0.5,
        cluster_key: Optional[str] = None,
        batch_size: int = 512,
        return_paths: bool = False,
    ) -> "AnnData":
        """Predict cell fates via ODE propagation.

        Parameters
        ----------
        adata : AnnData
            Input data with X_cell_state in obsm.
        time_range : float
            Time range for propagation.
        dt : float
            Time step.
        cluster_key : str, optional
            Key for cluster labels.
        batch_size : int
            Batch size for propagation.
        return_paths : bool
            Whether to store full paths.

        Returns
        -------
        AnnData
            Data with predicted fates in obs["fate"].
        """
        adata = adata.copy()
        IC = torch.from_numpy(adata.X.toarray()).to(self.device)
        z_sol_batches = []
        nn_list = []
        paths = []

        for batch in tqdm(
            torch.split(IC, batch_size), desc="Batch Propagation", leave=False
        ):
            z_sol = self.propagate(batch, time_range=time_range, num_points=int(time_range / dt))
            z_final = z_sol[-1, :, :]
            _, __, nn = self.project_z(z_final.to(self.device), adata)
            nn_list.append(nn)
            z_sol_batches.append(z_sol)
            if return_paths:
                _, __, path = self.project_z(z_sol.to(self.device), adata)
                paths.append(path)

        z_sol = torch.cat(z_sol_batches, dim=1)
        nn = torch.cat(nn_list, dim=0)
        if return_paths:
            paths = torch.cat(paths, dim=1)

        adata.obs["fate"] = "Other"
        predicted_fates = adata.obs[cluster_key][nn.cpu().numpy()].values
        adata.obs["fate"] = predicted_fates

        if return_paths:
            self.paths = paths
            self.z_sol = z_sol.detach()
            self.fates = nn
        else:
            self.z_sol = z_sol.detach()
            self.fates = nn

        return adata

    def stream_lines(
        self,
        embedding: str,
        save: bool = False,
        file_name: Optional[str] = None,
        *,
        color: str = "clusters",
        cmap: Optional[str] = None,
        palette: Optional[Dict] = None,
        size: float = 6.0,
        alpha: float = 0.9,
        title: Optional[str] = None,
        legend_loc: str = "right",
        frameon: bool = False,
        bg_color: str = "white",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = True,
        show: bool = True,
    ) -> plt.Axes:
        """Plot CellRank streamlines in embedding space.

        Uses CellRank's ConnectivityKernel and plot_projection for proper
        velocity field visualization with streamlines.

        Parameters
        ----------
        embedding : str
            Embedding key in adata.obsm (e.g., "X_umap", "X_tsne", "X_diff_state").
        color : str
            Column in adata.obs to color by (categorical or continuous).
        palette : dict or list, optional
            For categorical coloring. If None, tries adata.uns[f"{color}_colors"].
        cmap : str, optional
            Colormap for continuous variables (e.g., "viridis", "plasma").
        vmin, vmax : float, optional
            Clamp continuous colormap range.
        legend_loc : str or None
            Legend placement for categorical coloring ("right", "on data", None).
        colorbar : bool
            Show colorbar for continuous variables.
        size : float
            Marker size.
        alpha : float
            Marker opacity.
        frameon : bool
            Show frame around plot.
        bg_color : str
            Background color.
        title : str, optional
            Plot title.
        save : bool
            If True, save figure to file_name.
        file_name : str, optional
            Output file path (if save=True).
        show : bool
            If True, display the figure.

        Returns
        -------
        plt.Axes
            The matplotlib axes object.

        Examples
        --------
        >>> lsd.stream_lines("X_umap", color="clusters")
        >>> lsd.stream_lines("X_umap", color="lsd_pseudotime", cmap="viridis")
        """
        from cellrank.kernels import ConnectivityKernel

        adata = self.get_adata()

        # Build kernel with precomputed transitions
        ck = ConnectivityKernel(adata)
        ck.transition_matrix = adata.obsp["transitions"]

        # Determine if color is categorical or continuous
        if color not in adata.obs.columns:
            raise KeyError(f"`color`='{color}' not found in adata.obs.")
        dtype = adata.obs[color].dtype
        is_categorical = (str(dtype) == "category") or (
            getattr(dtype, "name", "") == "category"
        )

        # Derive palette if needed (for categorical)
        if is_categorical and palette is None:
            uns_key = f"{color}_colors"
            if uns_key in adata.uns:
                cats = list(adata.obs[color].astype("category").cat.categories)
                cols = list(adata.uns[uns_key])
                if len(cols) == len(cats):
                    palette = dict(zip(cats, cols))

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_facecolor(bg_color)

        # Common kwargs passed to CellRank's plotter
        common_kwargs = dict(
            basis=embedding,
            recompute=True,
            ax=ax,
            show=False,  # we handle showing/saving ourselves
            frameon=frameon,
            size=size,
            alpha=alpha,
            title=title,
        )

        # Coloring options
        if is_categorical:
            # Categorical: pass palette and legend settings
            plot_ax = ck.plot_projection(
                color=color, palette=palette, legend_loc=legend_loc, **common_kwargs
            )
        else:
            # Continuous: pass cmap, vmin/vmax, and colorbar toggle
            plot_ax = ck.plot_projection(
                color=color,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                colorbar=colorbar,
                legend_loc=None,  # no categorical legend
                **common_kwargs,
            )

        # Ensure we have an axis
        ax = plot_ax if plot_ax is not None else ax

        # Rasterize heavy artists (streamlines, scatter, etc.)
        for coll in ax.collections:
            try:
                coll.set_rasterized(True)
            except Exception:
                pass
        for ln in ax.lines:
            try:
                ln.set_rasterized(True)
            except Exception:
                pass

        # Save or show
        if save and file_name is not None:
            fig.savefig(file_name, format="svg", bbox_inches="tight")
            plt.close(fig)
        else:
            if show:
                plt.show()
            else:
                plt.close(fig)

        return ax

    def _perturb(
        self,
        adata: "AnnData",
        x: torch.Tensor,
        gene_name: str,
        perturbation_level: float = 0,
        dt_pert: float = 0.1,
        t_unpert: float = 10,
        max_perturbations: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper method for gene perturbation analysis.

        Iteratively perturbs gene expression and propagates cells through
        the learned potential landscape to find stable states.

        Parameters
        ----------
        adata : AnnData
            Reference data with cell states.
        x : torch.Tensor
            Initial expression values (batch of cells).
        gene_name : str
            Gene to perturb.
        perturbation_level : float
            Expression level after perturbation (0 for knockout).
        dt_pert : float
            Time step for intermediate propagations.
        t_unpert : float
            Final propagation time to find stable fate.
        max_perturbations : int
            Number of iterative perturbation steps.

        Returns
        -------
        fate_perturb : torch.Tensor
            Nearest neighbor indices for perturbed fates.
        fate_unperturb : torch.Tensor
            Nearest neighbor indices for unperturbed fates.
        """
        gene_idx = adata.var_names.get_loc(gene_name)
        X = torch.tensor(adata.X.toarray(), dtype=torch.float32).to(self.device)
        x0_pert = x.clone()
        x0_pert[:, gene_idx] = perturbation_level
        x0_unpert = x.clone()

        n_samples = x.shape[0]
        prev_nn_perturb = torch.full((n_samples,), -1, dtype=torch.long, device=self.device)
        prev_nn_unperturb = torch.full((n_samples,), -1, dtype=torch.long, device=self.device)

        # Iterative perturbation until convergence
        for i in range(max_perturbations):
            # Perturbed trajectory
            z_perturb = self.propagate(x0_pert, time_range=dt_pert, num_points=10)
            z_perturb_final = z_perturb[-1, :, :]

            # Project to data manifold
            _, x_proj, nn_perturb = self.project_z(z_perturb_final.to(self.device), adata)
            nn_perturb = nn_perturb.to(self.device)
            prev_nn_perturb = nn_perturb.clone()

            # Unperturbed trajectory
            z_unperturb = self.propagate(x0_unpert, time_range=dt_pert, num_points=10)
            z_unperturb_final = z_unperturb[-1, :, :]

            # Project to data manifold
            _, __, nn_unperturb = self.project_z(z_unperturb_final.to(self.device), adata)
            nn_unperturb = nn_unperturb.to(self.device)
            prev_nn_unperturb = nn_unperturb.clone()

            # Update initial conditions with projected data + reapply perturbation
            x0_pert = X[nn_perturb, :].clone()
            x0_unpert = X[nn_unperturb, :].clone()
            x0_pert[:, gene_idx] = perturbation_level

        # Final propagation to find stable fates
        final_nn_pert = prev_nn_perturb.clone()
        final_nn_unpert = prev_nn_unperturb.clone()
        x0_pert = X[final_nn_pert, :].clone()
        x0_unpert = X[final_nn_unpert, :].clone()

        z_unperturb = self.propagate(x0_unpert, time_range=t_unpert, num_points=10)
        z_unperturb_final = z_unperturb[-1, :, :]
        _, __, fate_unperturb = self.project_z(z_unperturb_final.to(self.device), adata)

        z_perturb = self.propagate(x0_pert, time_range=t_unpert, num_points=10)
        z_perturb_final = z_perturb[-1, :, :]
        _, __, fate_perturb = self.project_z(z_perturb_final.to(self.device), adata)

        return fate_perturb, fate_unperturb

    def perturb(
        self,
        adata: "AnnData",
        x: torch.Tensor,
        gene_name: str,
        cluster_key: str,
        perturbation_level: float = 0,
        dt_pert: float = 0.2,
        t_unpert: float = 15,
        max_perturbations: int = 10,
        batch_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform in-silico gene perturbation analysis.

        Simulates knockout or overexpression of specified genes and predicts
        resulting cell state changes and fate shifts via ODE propagation.

        Parameters
        ----------
        adata : AnnData
            Reference data with cell states in obsm['X_cell_state'].
        x : torch.Tensor
            Initial expression values for cells to perturb.
        gene_name : str
            Gene to perturb (must be in adata.var_names).
        cluster_key : str
            Key in adata.obs for cluster/cell type labels.
        perturbation_level : float
            Expression level after perturbation (default 0 for knockout).
            For overexpression, use positive value (e.g., log2(fold_change)).
        dt_pert : float
            Time step for intermediate propagations.
        t_unpert : float
            Final propagation time to find stable fate.
        max_perturbations : int
            Number of iterative perturbation steps before final propagation.
        batch_size : int
            Batch size for processing cells.

        Returns
        -------
        perturbed_fates : np.ndarray
            Predicted cell type labels after perturbation.
        unperturbed_fates : np.ndarray
            Predicted cell type labels without perturbation.

        Examples
        --------
        >>> # Knockout Sox2
        >>> adata = lsd.get_adata()
        >>> X = torch.from_numpy(adata.X.toarray()).float()
        >>> pert_fates, unpert_fates = lsd.perturb(
        ...     adata, X, "Sox2", "clusters", perturbation_level=0
        ... )
        >>> # Compare fate distributions
        >>> print("Perturbed:", np.unique(pert_fates, return_counts=True))
        >>> print("Unperturbed:", np.unique(unpert_fates, return_counts=True))
        """
        final_nn_pert_list = []
        final_nn_unpert_list = []

        # Process in batches
        for batch in torch.split(x, batch_size):
            fate_perturb, fate_unperturb = self._perturb(
                adata=adata,
                x=batch,
                gene_name=gene_name,
                perturbation_level=perturbation_level,
                dt_pert=dt_pert,
                t_unpert=t_unpert,
                max_perturbations=max_perturbations,
            )
            final_nn_pert_list.append(fate_perturb)
            final_nn_unpert_list.append(fate_unperturb)

        # Concatenate results from all batches
        final_nn_pert_all = torch.cat(final_nn_pert_list, dim=0).cpu().numpy()
        final_nn_unpert_all = torch.cat(final_nn_unpert_list, dim=0).cpu().numpy()

        # Map indices to cluster labels
        perturbed_fates = adata.obs[cluster_key].iloc[final_nn_pert_all].values
        unperturbed_fates = adata.obs[cluster_key].iloc[final_nn_unpert_all].values

        return perturbed_fates, unperturbed_fates
