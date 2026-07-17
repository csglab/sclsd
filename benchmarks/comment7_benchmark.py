"""Fresh-process old-versus-new benchmarks for comment 7."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch

from benchmarks.comment7_legacy import (
    calculate_transition_probs as legacy_transition_probs,
)
from benchmarks.comment7_legacy import (
    create_phylogeny_matrix as legacy_phylogeny_matrix,
)
from benchmarks.comment7_legacy import (
    random_walks_dense,
)
from benchmarks.freeze_comment7_baseline import DENTATE_PHYLOGENY

BONE_MARROW_PHYLOGENY = {
    "HSC_1": ["HSC_2", "Ery_1", "Mega"],
    "Ery_1": ["Ery_2"],
    "HSC_2": ["Precursors"],
    "Precursors": ["Mono_1", "Mono_2", "DCs"],
    "Ery_2": [],
    "Mega": [],
    "Mono_1": [],
    "Mono_2": [],
    "DCs": [],
}

MOUSE_CORTEX_PHYLOGENY = {
    "Apical progenitors": ["IPC", "IN nonMGE"],
    "IPC": ["ULPN", "Migrating neurons"],
    "IN nonMGE": ["IN MGE"],
    "ULPN": ["DLPN"],
    "Migrating neurons": ["DLPN"],
    "DLPN": [],
    "IN MGE": [],
}

DATASETS = {
    "dentate": (DENTATE_PHYLOGENY, "prior_pseudotime", 2048, 12),
    "bone_marrow": (BONE_MARROW_PHYLOGENY, "palantir_pseudotime", 2048, 12),
    "mouse_cortex": (MOUSE_CORTEX_PHYLOGENY, "prior_pseudotime", 8192, 30),
}


def _current_rss_bytes() -> int:
    statm = Path("/proc/self/statm").read_text().split()
    return int(statm[1]) * os.sysconf("SC_PAGE_SIZE")


def _peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _csr_hash(matrix: sp.spmatrix) -> str:
    matrix = matrix.tocsr()
    digest = hashlib.sha256()
    for array in (matrix.data, matrix.indices, matrix.indptr):
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    digest.update(np.asarray(matrix.shape, dtype=np.int64).view(np.uint8))
    return digest.hexdigest()


def _run_worker(args: argparse.Namespace) -> dict:
    phylogeny, time_key, n_walks, path_len = DATASETS[args.dataset]
    adata = sc.read(args.data)
    connectivity = adata.obsp["connectivities"].tocsr()
    pseudotime = adata.obs[time_key].to_numpy()

    if args.operation == "walks":
        from sclsd.preprocessing.prior import prior_transition_matrix

        transition = prior_transition_matrix(adata, time_key, beta_t=50)
    else:
        transition = None

    current_phylogeny = None
    current_transition = None
    current_walks = None
    if args.implementation == "current":
        if args.operation == "phylogeny":
            from sclsd.preprocessing.prior import _create_phylogeny_matrix

            current_phylogeny = _create_phylogeny_matrix
        elif args.operation == "transition":
            from sclsd.train.trainer import LSD

            current_transition = LSD.calculate_transition_probs
        else:
            from sclsd.train.walks import random_walks_sparse

            current_walks = random_walks_sparse

    gc.collect()
    rss_before = _current_rss_bytes()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    if args.operation == "phylogeny":
        if args.implementation == "legacy":
            mask = legacy_phylogeny_matrix(adata, phylogeny, "clusters")
        else:
            mask = current_phylogeny(adata, phylogeny, "clusters")
        result = connectivity.multiply(mask).tocsr()
    elif args.operation == "transition":
        if args.implementation == "legacy":
            binary = (connectivity.toarray() > 0).astype(float)
            result = legacy_transition_probs(-pseudotime, binary, beta=50)
        else:
            binary = connectivity.copy()
            binary.data = np.ones_like(binary.data, dtype=float)
            result = current_transition(None, -pseudotime, binary, beta=50)
    else:
        if args.implementation == "legacy":
            dense_transition = torch.from_numpy(transition.toarray()).float()
            result = random_walks_dense(
                dense_transition, n_walks, path_len, random_state=42
            )
        else:
            result = current_walks(
                transition,
                n_steps=path_len,
                n_trajectories=n_walks,
                random_state=42,
            )

    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    peak_rss = _peak_rss_bytes()

    if sp.issparse(result):
        output = {
            "output_type": type(result).__name__,
            "output_nnz": int(result.nnz),
            "output_hash": _csr_hash(result),
        }
    elif isinstance(result, torch.Tensor):
        array = result.cpu().numpy()
        output = {
            "output_type": str(result.dtype),
            "output_nnz": None,
            "output_hash": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    else:
        array = np.asarray(result)
        output = {
            "output_type": str(array.dtype),
            "output_nnz": int(np.count_nonzero(array)),
            "output_hash": hashlib.sha256(array.tobytes()).hexdigest(),
        }

    return {
        "dataset": args.dataset,
        "implementation": args.implementation,
        "operation": args.operation,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "connectivity_nnz": int(connectivity.nnz),
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "rss_before_bytes": rss_before,
        "peak_rss_bytes": peak_rss,
        "peak_rss_delta_bytes": max(0, peak_rss - rss_before),
        **output,
    }


def _summarize(records: list[dict]) -> dict:
    summary = {}
    for key in ("wall_seconds", "cpu_seconds", "peak_rss_bytes", "peak_rss_delta_bytes"):
        values = np.asarray([record[key] for record in records], dtype=float)
        summary[key] = {
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--operation", choices=["phylogeny", "transition", "walks"], required=True)
    parser.add_argument("--implementation", choices=["legacy", "current"], required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(_run_worker(args)))
        return

    records = []
    for _ in range(args.repeats):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--data",
            str(args.data),
            "--dataset",
            args.dataset,
            "--operation",
            args.operation,
            "--implementation",
            args.implementation,
            "--worker",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        records.append(json.loads(completed.stdout.strip().splitlines()[-1]))

    payload = {
        "python": platform.python_version(),
        "repeats": args.repeats,
        "records": records,
        "summary": _summarize(records),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
