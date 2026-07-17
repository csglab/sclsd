import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from anndata import AnnData

from sclsd import LSD, LSDConfig, clear_pyro_state, set_all_seeds


def test_toy_training_smoke():
    """Exercise the principal LSD training path on a tiny CPU dataset."""
    raw = np.array(
        [
            [4, 1, 0, 0],
            [3, 2, 0, 0],
            [2, 3, 1, 0],
            [1, 4, 1, 0],
            [0, 3, 2, 1],
            [0, 2, 3, 1],
            [0, 1, 4, 2],
            [0, 0, 4, 3],
        ],
        dtype=np.float32,
    )
    library_size = raw.sum(axis=1)
    normalized = np.log1p(raw / library_size[:, None] * 1e4)

    rows = []
    cols = []
    for cell in range(raw.shape[0]):
        rows.append(cell)
        cols.append(cell)
        if cell > 0:
            rows.append(cell)
            cols.append(cell - 1)
        if cell + 1 < raw.shape[0]:
            rows.append(cell)
            cols.append(cell + 1)

    connectivities = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(raw.shape[0], raw.shape[0]),
    )
    adata = AnnData(
        X=sp.csr_matrix(normalized),
        obs=pd.DataFrame(
            {
                "librarysize": library_size,
                "pseudotime": np.linspace(0, 1, raw.shape[0]),
            },
            index=[f"cell-{i}" for i in range(raw.shape[0])],
        ),
        var=pd.DataFrame(index=[f"gene-{i}" for i in range(raw.shape[1])]),
    )
    adata.layers["raw"] = sp.csr_matrix(raw)
    adata.obsp["connectivities"] = connectivities

    cfg = LSDConfig()
    cfg.model.z_dim = 2
    cfg.model.layer_dims.B_decoder = [4]
    cfg.model.layer_dims.z_decoder = [4]
    cfg.model.layer_dims.x_encoder = [4]
    cfg.model.layer_dims.z_encoder = [4]
    cfg.model.layer_dims.xl_encoder = [4]
    cfg.model.layer_dims.potential = [4]
    cfg.walks.batch_size = 2
    cfg.walks.path_len = 2
    cfg.walks.num_walks = 4

    clear_pyro_state()
    set_all_seeds(0)
    lsd = LSD(adata, cfg, device=torch.device("cpu"))
    lsd.set_prior_transition(prior_time_key="pseudotime", random_state=0)
    lsd.prepare_walks()
    lsd.train(num_epochs=1, plot_loss=False, random_state=0)
    result = lsd.get_adata()

    assert result.obsm["X_cell_state"].shape == (adata.n_obs, cfg.model.z_dim)
    assert result.obsm["X_diff_state"].shape == (adata.n_obs, cfg.model.B_dim)
    assert result.obsp["transitions"].shape == (adata.n_obs, adata.n_obs)
    assert np.isfinite(result.obs["potential"]).all()
    assert np.isfinite(result.obs["entropy"]).all()
