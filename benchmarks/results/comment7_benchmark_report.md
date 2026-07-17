# Comment 7 sparse-memory benchmark

Date: 2026-07-17

## Method

- Environment: Python 3.10.12, NumPy 1.26.4, SciPy 1.15.3, PyTorch 2.4.1+cu121.
- Each implementation ran five times in a fresh process.
- Imports and dataset loading occurred before the timed section.
- Times below are medians; detailed minima, maxima, input dimensions, output hashes,
  and total process RSS are stored in the adjacent JSON files.
- Incremental peak RSS is Linux `ru_maxrss` minus RSS immediately before the measured
  operation. It includes allocations retained by the operation but excludes dataset loading.
- CUDA was not available in the execution session, so no direct GPU measurement was made.

## Frozen baseline

The legacy implementations were copied and executed before production-source edits.
The source hashes, dataset hash, environment, configuration, and output hashes are in
[`comment7_baseline_metadata.json`](comment7_baseline_metadata.json). The frozen
Dentate Gyrus graph and effective outputs are stored in
[`tests/fixtures/dentate_gyrus_comment7_baseline.npz`](../../tests/fixtures/dentate_gyrus_comment7_baseline.npz).

## Old-versus-new results

| Dataset | Operation | Legacy wall (s) | Sparse wall (s) | Legacy CPU (s) | Sparse CPU (s) | Legacy incremental peak RSS (MiB) | Sparse incremental peak RSS (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Dentate Gyrus (2,460 cells; 53,472 edges) | Phylogeny | 2.0818 | 0.0197 | 2.0816 | 0.0197 | 141.86 | 3.13 |
| Dentate Gyrus | Transition probabilities | 0.0877 | 0.0029 | 0.0877 | 0.0029 | 184.97 | 3.62 |
| Dentate Gyrus (2,048 walks; length 12) | Random walks | 0.1571 | 0.2764 | 3.3273 | 0.2764 | 68.59 | 1.70 |
| Bone Marrow (5,292 cells; 114,718 edges) | Phylogeny | 9.4167 | 0.0436 | 9.4163 | 0.0436 | 455.42 | 5.93 |
| Bone Marrow | Transition probabilities | 0.4338 | 0.0052 | 0.4337 | 0.0052 | 855.12 | 6.71 |

The sparse CPU walker reduced measured incremental peak RSS by approximately 40-fold on
Dentate Gyrus. Its wall time increased by approximately 0.12 seconds for the manuscript
configuration, while measured CPU time decreased because the legacy PyTorch operation used
multiple CPU threads.

## New-only scaling check

Mouse Cortex contains 12,814 cells and 250,236 connectivity edges. The manuscript walk
configuration uses 8,192 walks of length 30.

| Operation | Median wall (s) | Median CPU (s) | Median incremental peak RSS (MiB) |
|---|---:|---:|---:|
| Phylogeny | 0.0768 | 0.0768 | 7.06 |
| Transition probabilities | 0.0096 | 0.0096 | 8.47 |
| Sparse CPU random walks | 2.8805 | 2.8805 | 19.10 |

The corresponding dense float32 transition matrix would contain 164,198,596 values and
occupy approximately 626 MiB before accounting for the per-step indexed probability rows or
other model allocations. The sparse CPU path does not allocate this matrix on the GPU.

## Equivalence and behavior

- Dentate Gyrus effective phylogeny connectivity has the exact frozen legacy hash.
- Bone Marrow effective phylogeny connectivity has the exact legacy hash.
- Dentate Gyrus transition support is identical and values agree with the frozen dense
  result at `rtol=1e-12`, `atol=1e-14`.
- Bone Marrow transition values agree at the same tolerance; the observed maximum absolute
  difference was `6.66e-16`.
- Sparse walks are exactly reproducible for a fixed seed, use only nonzero transition edges,
  match controlled dense inverse-CDF choices, and reproduce expected empirical frequencies.
- Walk indices intentionally differ from legacy `torch.multinomial` output because the CPU
  and PyTorch samplers use different random-number streams.
- The existing public `random_walks_gpu` function remains unchanged.

## Real workflow verification

The complete Dentate Gyrus phylogeny, prior-transition, and walk-generation path passed with
the manuscript configuration. As in the legacy effective graph, three cells had no permitted
outgoing transition and were removed. The retained 2,457-cell CSR transition matrix contained
48,455 nonzero entries, and 2,048 length-12 int32 walks were generated successfully.

## Limitations

This change removes the dense cell-by-cell transition matrix from the main random-walk GPU
path and makes phylogeny/posterior transitions edge-sparse. It does not address other possible
memory hotspots, including dense expression materialization, fate projection, perturbation,
or arbitrary direct use of the retained dense GPU utility functions.
