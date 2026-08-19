from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
from anndata import AnnData

from sclsd.preprocessing.prior import _create_phylogeny_matrix
from sclsd.train.trainer import LSD
from sclsd.train.walks import random_walks_sparse

BASELINE_PATH = (
    __file__.replace(
        "test_sparse_memory.py", "fixtures/dentate_gyrus_comment7_baseline.npz"
    )
)

DENTATE_PHYLOGENY = {
    "nIPC": ["Neuroblast", "Radial Glia-like"],
    "Neuroblast": ["Granule immature"],
    "Granule immature": ["Granule mature"],
    "Granule mature": [],
    "Radial Glia-like": ["Astrocytes"],
    "Astrocytes": [],
}


def _load_csr(baseline, prefix):
    return sp.csr_matrix(
        (
            baseline[f"{prefix}_data"],
            baseline[f"{prefix}_indices"],
            baseline[f"{prefix}_indptr"],
        ),
        shape=tuple(baseline[f"{prefix}_shape"]),
    )


@pytest.fixture(scope="module")
def dentate_graph():
    with np.load(BASELINE_PATH, allow_pickle=False) as baseline:
        connectivity = _load_csr(baseline, "connectivity")
        expected_phylogeny = _load_csr(baseline, "effective_phylogeny")
        expected_transition = _load_csr(baseline, "transition")
        clusters = baseline["clusters"].copy()
        pseudotime = baseline["pseudotime"].copy()

    obs = pd.DataFrame(
        {"clusters": clusters, "prior_pseudotime": pseudotime},
        index=[f"dentate-cell-{i}" for i in range(len(clusters))],
    )
    adata = AnnData(
        X=sp.csr_matrix((len(clusters), 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["placeholder-gene"]),
    )
    adata.obsp["connectivities"] = connectivity
    return adata, expected_phylogeny, expected_transition


def _assert_sparse_equal(actual, expected):
    actual = actual.tocsr()
    expected = expected.tocsr()
    actual.sort_indices()
    expected.sort_indices()
    difference = actual - expected
    difference.eliminate_zeros()
    assert difference.nnz == 0


def test_preprocessing_phylogeny_matches_frozen_effective_graph(dentate_graph):
    adata, expected, _ = dentate_graph
    mask = _create_phylogeny_matrix(adata, DENTATE_PHYLOGENY, "clusters")
    actual = adata.obsp["connectivities"].multiply(mask).tocsr()

    _assert_sparse_equal(actual, expected)
    assert sp.isspmatrix_csr(mask)
    assert mask.nnz <= adata.obsp["connectivities"].nnz


def test_trainer_phylogeny_matches_frozen_effective_graph(dentate_graph):
    adata, expected, _ = dentate_graph
    lsd = object.__new__(LSD)
    lsd.adata = adata
    lsd.phylogeny = DENTATE_PHYLOGENY
    lsd.cluster_key = "clusters"

    mask = lsd._create_phylogeny_matrix()
    actual = adata.obsp["connectivities"].multiply(mask).tocsr()

    _assert_sparse_equal(actual, expected)
    assert mask.nnz <= adata.obsp["connectivities"].nnz


def test_transition_probabilities_match_frozen_dense_result(dentate_graph):
    adata, _, expected = dentate_graph
    connectivity = adata.obsp["connectivities"].copy().tocsr()
    connectivity.data = np.ones_like(connectivity.data, dtype=float)
    potential = -adata.obs["prior_pseudotime"].to_numpy()

    actual = LSD.calculate_transition_probs(None, potential, connectivity, beta=50)
    actual = actual.tocsr()
    actual.sort_indices()
    expected = expected.copy().tocsr()
    expected.sort_indices()

    assert np.array_equal(actual.indptr, expected.indptr)
    assert np.array_equal(actual.indices, expected.indices)
    np.testing.assert_allclose(actual.data, expected.data, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(actual.sum(axis=1)).ravel(),
        np.asarray(expected.sum(axis=1)).ravel(),
        rtol=1e-12,
        atol=1e-14,
    )


def test_supplied_transition_remains_sparse(dentate_graph):
    adata, _, transition = dentate_graph
    lsd = object.__new__(LSD)
    lsd.adata = adata

    lsd.set_prior_transition(prior_transition=transition)

    assert sp.isspmatrix_csr(lsd.P)
    assert np.array_equal(lsd.P.data, transition.data.astype(np.float32))


def test_sparse_walks_are_reproducible_and_follow_graph(dentate_graph):
    _, _, transition = dentate_graph
    first = random_walks_sparse(transition, 12, 2048, random_state=42)
    second = random_walks_sparse(transition, 12, 2048, random_state=42)

    assert first.dtype == torch.int32
    assert first.shape == (2048, 12)
    assert np.array_equal(first.numpy(), second.numpy())

    walks = first.numpy()
    for step in range(1, walks.shape[1]):
        weights = np.asarray(transition[walks[:, step - 1], walks[:, step]]).ravel()
        assert np.all(weights > 0)


def test_sparse_walk_empirical_frequencies_match_probabilities():
    probabilities = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    transition = sp.csr_matrix(np.tile(probabilities, (4, 1)))

    walks = random_walks_sparse(
        transition, n_steps=2, n_trajectories=100_000, random_state=7
    ).numpy()
    observed = np.bincount(walks[:, 1], minlength=4) / len(walks)

    np.testing.assert_allclose(observed, probabilities, atol=0.01, rtol=0)


def test_sparse_and_dense_inverse_cdf_agree_for_controlled_uniforms():
    transition = sp.csr_matrix(
        [
            [0.0, 0.25, 0.0, 0.75],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.1, 0.2, 0.7],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    states = np.array([0, 1, 2, 3, 0, 2])
    uniforms = np.array([0.1, 0.6, 0.05, 0.9, 0.9, 0.25])

    dense_choices = []
    sparse_choices = []
    dense = transition.toarray()
    for state, uniform in zip(states, uniforms):
        dense_choices.append(np.searchsorted(np.cumsum(dense[state]), uniform, side="right"))

        start = transition.indptr[state]
        end = transition.indptr[state + 1]
        neighbors = transition.indices[start:end]
        probabilities = transition.data[start:end]
        offset = np.searchsorted(np.cumsum(probabilities), uniform, side="right")
        sparse_choices.append(neighbors[offset])

    assert np.array_equal(dense_choices, sparse_choices)


def test_sparse_walks_reject_empty_rows():
    transition = sp.csr_matrix((3, 3), dtype=float)

    with pytest.raises(ValueError, match="no outgoing transitions"):
        random_walks_sparse(transition, n_steps=2, n_trajectories=3, random_state=0)


def test_prepare_walks_keeps_transition_sparse(dentate_graph):
    _, _, transition = dentate_graph
    lsd = object.__new__(LSD)
    lsd.P = transition.copy()
    lsd.path_len = 3
    lsd.walk_config = SimpleNamespace(num_walks=16, random_state=11)

    lsd.prepare_walks()

    assert sp.isspmatrix_csr(lsd.P)
    assert lsd.walks.shape == (16, 3)
