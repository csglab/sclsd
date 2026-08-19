"""Freeze realistic legacy outputs before the comment-7 source update."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch

from benchmarks.comment7_legacy import (
    calculate_transition_probs,
    create_phylogeny_matrix,
    random_walks_dense,
)

DENTATE_PHYLOGENY = {
    "nIPC": ["Neuroblast", "Radial Glia-like"],
    "Neuroblast": ["Granule immature"],
    "Granule immature": ["Granule mature"],
    "Granule mature": [],
    "Radial Glia-like": ["Astrocytes"],
    "Astrocytes": [],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csr_payload(prefix: str, matrix: sp.spmatrix) -> dict:
    matrix = matrix.tocsr()
    return {
        f"{prefix}_data": matrix.data,
        f"{prefix}_indices": matrix.indices,
        f"{prefix}_indptr": matrix.indptr,
        f"{prefix}_shape": np.asarray(matrix.shape, dtype=np.int64),
    }


def _csr_hash(matrix: sp.spmatrix) -> str:
    matrix = matrix.tocsr()
    digest = hashlib.sha256()
    for array in (matrix.data, matrix.indices, matrix.indptr):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    digest.update(np.asarray(matrix.shape, dtype=np.int64).view(np.uint8))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    adata = sc.read(args.data)
    connectivity = adata.obsp["connectivities"].tocsr()
    pseudotime = adata.obs["prior_pseudotime"].to_numpy()
    clusters = adata.obs["clusters"].astype(str).to_numpy(dtype="U")

    legacy_mask = create_phylogeny_matrix(adata, DENTATE_PHYLOGENY, "clusters")
    effective_connectivity = connectivity.multiply(legacy_mask).tocsr()

    binary_connectivity = (connectivity.toarray() > 0).astype(float)
    dense_transition = calculate_transition_probs(
        -pseudotime, binary_connectivity, beta=50
    )
    sparse_transition = sp.csr_matrix(dense_transition)
    walks = random_walks_dense(
        torch.from_numpy(dense_transition).float(),
        n_trajectories=2048,
        path_len=12,
        random_state=42,
    ).numpy()

    payload = {
        **_csr_payload("connectivity", connectivity),
        **_csr_payload("effective_phylogeny", effective_connectivity),
        **_csr_payload("transition", sparse_transition),
        "clusters": clusters,
        "pseudotime": pseudotime,
        "legacy_walks": walks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)

    repo = Path(__file__).resolve().parents[1]
    package_names = ["numpy", "scipy", "pandas", "anndata", "scanpy", "torch", "pyro-ppl"]
    package_versions = {name: metadata.version(name) for name in package_names}
    metadata_payload = {
        "created_before_source_update": True,
        "python": platform.python_version(),
        "packages": package_versions,
        "source_sha256": {
            "prior.py": _sha256(repo / "src/sclsd/preprocessing/prior.py"),
            "trainer.py": _sha256(repo / "src/sclsd/train/trainer.py"),
            "walks.py": _sha256(repo / "src/sclsd/train/walks.py"),
        },
        "dataset": {
            "path": str(args.data),
            "sha256": _sha256(args.data),
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "connectivity_nnz": int(connectivity.nnz),
            "cluster_count": int(adata.obs["clusters"].nunique()),
        },
        "configuration": {
            "walk_count": 2048,
            "path_len": 12,
            "random_state": 42,
            "transition_beta": 50,
        },
        "legacy_outputs": {
            "raw_phylogeny_mask_nnz": int(legacy_mask.nnz),
            "effective_phylogeny_nnz": int(effective_connectivity.nnz),
            "effective_phylogeny_sha256": _csr_hash(effective_connectivity),
            "transition_nnz": int(sparse_transition.nnz),
            "transition_sha256": _csr_hash(sparse_transition),
            "walks_sha256": hashlib.sha256(walks.tobytes()).hexdigest(),
            "walks_dtype": str(walks.dtype),
            "walks_shape": list(walks.shape),
        },
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata_payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
