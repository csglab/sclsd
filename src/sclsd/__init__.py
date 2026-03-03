"""LSDpy: Latent State Dynamics for single-cell trajectory inference.

LSDpy is a Python package for inferring cell differentiation trajectories
from single-cell RNA sequencing data using neural ODEs and variational inference.

Main Classes
------------
LSD : Main model class for training and inference
LSDConfig : Configuration dataclass for model hyperparameters

Example
-------
>>> from sclsd import LSD, LSDConfig
>>> import scanpy as sc
>>>
>>> # Load data
>>> adata = sc.read("my_data.h5ad")
>>>
>>> # Configure model
>>> cfg = LSDConfig()
>>> cfg.walks.path_len = 50
>>> cfg.model.z_dim = 10
>>>
>>> # Create and train model
>>> lsd = LSD(adata, cfg, device=torch.device("cuda"))
>>> lsd.set_prior_transition(prior_time_key="pseudotime")
>>> lsd.prepare_walks()
>>> lsd.train(num_epochs=100, random_state=42)
>>>
>>> # Get results
>>> result = lsd.get_adata()
>>> print(result.obs["lsd_pseudotime"])
"""

from sclsd._version import __version__

# Core configuration
from sclsd.core.config import (
    LSDConfig,
    ModelConfig,
    OptimizerConfig,
    WalkConfig,
    LayerDims,
    AdamConfig,
    KLScheduleConfig,
    WassersteinScheduleConfig,
)

# Core model
from sclsd.core.model import LSDModel
from sclsd.core.networks import (
    StateDecoder,
    ZDecoder,
    XEncoder,
    ZEncoder,
    LEncoder,
    PotentialNet,
    GradientNet,
)

# Training
from sclsd.train.trainer import LSD
from sclsd.train.walks import prepare_walks, random_walks_gpu

# Preprocessing
from sclsd.preprocessing.data import prepare_data_dict
from sclsd.preprocessing.prior import (
    get_prior_pseudotime,
    infer_prior_time,
    infer_phylo,
    get_prior_transition,
    prior_transition_matrix,
)

# Analysis
from sclsd.analysis.metrics import (
    cross_boundary_correctness,
    inner_cluster_coh,
    summary_scores,
    evaluate,
)

# Plotting
from sclsd.plotting import (
    plot_random_walks,
    visualize_random_walks_on_umap,
    plot_z_components,
    plot_streamlines,
)

# Utilities
from sclsd.utils.seed import set_all_seeds, clear_pyro_state

__all__ = [
    # Version
    "__version__",
    # Main API
    "LSD",
    "LSDConfig",
    "LSDModel",
    # Configuration
    "ModelConfig",
    "OptimizerConfig",
    "WalkConfig",
    "LayerDims",
    "AdamConfig",
    "KLScheduleConfig",
    "WassersteinScheduleConfig",
    # Networks
    "StateDecoder",
    "ZDecoder",
    "XEncoder",
    "ZEncoder",
    "LEncoder",
    "PotentialNet",
    "GradientNet",
    # Training
    "prepare_walks",
    "random_walks_gpu",
    # Preprocessing
    "prepare_data_dict",
    "get_prior_pseudotime",
    "infer_prior_time",
    "infer_phylo",
    "get_prior_transition",
    "prior_transition_matrix",
    # Analysis
    "cross_boundary_correctness",
    "inner_cluster_coh",
    "summary_scores",
    "evaluate",
    # Plotting
    "plot_random_walks",
    "visualize_random_walks_on_umap",
    "plot_z_components",
    "plot_streamlines",
    # Utilities
    "set_all_seeds",
    "clear_pyro_state",
]
