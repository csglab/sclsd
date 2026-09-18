# sclsd: Latent Space Dynamics for Single-Cell Trajectory Inference

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/sclsd)](https://pypi.org/project/sclsd/)


**sclsd** implements Latent Space Dynamics (LSD), a thermodynamic framework for modeling cell differentiation from single-cell RNA sequencing data.

Notebooks for reproducing manuscript figures and analyses are available at [csglab/sclsd-manuscript](https://github.com/csglab/sclsd-manuscript).

This README is the maintained documentation for the `sclsd` package. It covers
installation, input data, configuration, training, inference, reproducibility,
and citation information.

## Overview

LSD reinterprets Waddington's epigenetic landscape as an energy landscape in a learned latent cell state space. Cell differentiation is modeled as a stochastic dynamical system governed by a gradient flow down this potential surface, combined with noise representing gene expression variability.

The model jointly infers:

- **Cell state**: A latent representation of each cell's gene expression profile
- **Differentiation state**: A 2D embedding capturing developmental progression
- **Waddington potential**: An energy function whose gradient defines differentiation dynamics
- **Developmental entropy**: A measure of cellular plasticity derived from the uncertainty in differentiation state

## Installation

```bash
pip install sclsd
```

Or from source:

```bash
git clone https://github.com/csglab/sclsd.git
cd sclsd
pip install -e .
```

### Dependencies

The package requires Python 3.9 or later. Runtime dependencies, including
PyTorch, Pyro, torchdiffeq, Scanpy, AnnData, and CellRank, are installed by
`pip`. The complete dependency specification is maintained in
[`pyproject.toml`](https://github.com/csglab/sclsd/blob/main/pyproject.toml).

## Quick Start

```python
import scanpy as sc
import torch
from sclsd import LSD, LSDConfig

# Load preprocessed AnnData (log-normalized, with neighbors computed)
adata = sc.read("data.h5ad")

# Configure model
cfg = LSDConfig()
cfg.model.z_dim = 10           # Cell state dimensionality
cfg.walks.path_len = 50        # Trajectory length for training
cfg.walks.num_walks = 4096     # Number of training trajectories

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lsd = LSD(adata, cfg, device=device)

# Set prior transition matrix from pseudotime
lsd.set_prior_transition(prior_time_key="dpt_pseudotime")

# Generate training trajectories
lsd.prepare_walks()

# Train
lsd.train(num_epochs=100)

# Get results
result = lsd.get_adata()
```

## Output

After training, `lsd.get_adata()` returns an AnnData object with:

| Key | Location | Description |
|-----|----------|-------------|
| `X_cell_state` | `obsm` | Latent cell state representation |
| `X_diff_state` | `obsm` | 2D differentiation state embedding |
| `potential` | `obs` | Waddington potential value |
| `entropy` | `obs` | Developmental entropy (plasticity) |
| `lsd_pseudotime` | `obs` | Pseudotime derived from potential |
| `transitions` | `obsp` | Cell-cell transition probability matrix |

## Key Methods

### Cell Fate Prediction

Propagate cells through the learned landscape to predict terminal fates:

```python
result = lsd.get_cell_fates(
    adata=result,
    time_range=15.0,
    cluster_key="clusters",
    return_paths=True
)
# Predicted fates stored in result.obs["fate"]
```

### Velocity Streamlines

Visualize differentiation flow fields:

```python
lsd.stream_lines("X_umap", color="clusters")
```

### In Silico Gene Perturbation

Simulate gene knockouts and predict fate changes:

```python
X = torch.from_numpy(result.X.toarray()).float()
perturbed_fates, unperturbed_fates = lsd.perturb(
    adata=result,
    x=X,
    gene_name="Noto",
    cluster_key="clusters",
    perturbation_level=0  # Knockout
)
```

## Configuration

Key parameters in `LSDConfig`:

```python
cfg = LSDConfig()

# Model architecture
cfg.model.z_dim = 10              # Cell state dimensions
cfg.model.B_dim = 2               # Differentiation state dimensions (default: 2)
cfg.model.V_coeff = 0.01          # Potential regularization

# Training trajectories
cfg.walks.path_len = 50           # Steps per trajectory
cfg.walks.num_walks = 4096        # Number of trajectories
cfg.walks.batch_size = 256        # Batch size

# Optimizer
cfg.optimizer.adam.lr = 1e-3      # Learning rate
```

## Data Requirements

Input AnnData should contain:

- Log-normalized expression in `adata.X`
- Raw counts in `adata.layers["raw"]`
- Library sizes in `adata.obs["librarysize"]`
- Precomputed neighbor graph in `adata.obsp["connectivities"]`
- Pseudotime values in `adata.obs` when initializing transitions from a pseudotime key; alternatively, users may supply a transition matrix directly

## Method

LSD models cell state dynamics via the stochastic differential equation:

$$dm_t = -\nabla V(m_t) \ dt,  z_t \sim N(m_t, C_t)$$

where $V(m)$ is the Waddington potential parameterized by a neural network, and the gradient defines a neural ODE. The model is trained by variational inference, reconstructing gene expression through a zero-inflated negative binomial likelihood.

Training trajectories are generated by random walks on a k-nearest neighbor graph, biased by pseudotime to follow developmental progression.

## Reproducibility

Dataset-specific training and postprocessing notebooks are available in the
[`sclsd-manuscript`](https://github.com/csglab/sclsd-manuscript) repository.
The preprocessed datasets used by those notebooks are available from
[Zenodo record 18331587](https://zenodo.org/records/18331587).

## Citation

If you use `sclsd` or the accompanying analyses, please cite:

> Poursina, A., Hajhashemi, S., Mikaeili Namini, A., Saberi, A., Emad, A., &
> Najafabadi, H. S. (2026). A Latent Space Thermodynamic Model of Cell
> Differentiation. *bioRxiv*, 2026.03.04.709512.
> https://doi.org/10.64898/2026.03.04.709512

[View version 1 on bioRxiv](https://www.biorxiv.org/content/10.64898/2026.03.04.709512v1)

### BibTeX

```bibtex
@article{poursina2026latent,
  title     = {A Latent Space Thermodynamic Model of Cell Differentiation},
  author    = {Poursina, Ali and Hajhashemi, Shayan and
               {Mikaeili Namini}, Arsham and Saberi, Ali and
               Emad, Amin and Najafabadi, Hamed S.},
  journal   = {bioRxiv},
  pages     = {2026.03.04.709512},
  year      = {2026},
  publisher = {Cold Spring Harbor Laboratory},
  doi       = {10.64898/2026.03.04.709512},
  url       = {https://www.biorxiv.org/content/10.64898/2026.03.04.709512v1}
}
```

## Contact

For questions about the sclsd software, contact Ali Poursina at
[ali.poursina@mail.mcgill.ca](mailto:ali.poursina@mail.mcgill.ca). Bug reports and
feature requests can also be submitted through the
[GitHub issue tracker](https://github.com/csglab/sclsd/issues).

## License

MIT License
