# econhdfe 0.4.5 PPML execution/separation performance note

The externally supplied 2026-09-12 real-machine PPML result is retained as a **pre-fix baseline**: six-year optimized 213.681 s vs `ppmlhdfe` 800.036 s (3.74x) and replica 357.788 s (2.24x). The companion diagnosis established that the intended four-thread / 2048 MiB `ExecutionConfig` was not actually controlling the internal weighted projector, so these figures must not be relabelled as strict configured-resource measurements.

v0.4.5 repairs that plumbing and removes replica-only routing from optimized simplex/ReLU separation. Same-container fixed-four-thread development A/B checks preserved coefficient/sample/separation/iteration parity and measured about 1.14x versus v0.4.4.4 on a 1M high-zero two-FE synthetic PPML case and about 1.07x on a 300k three-FE case. These are regression guards, not target-hardware claims. Machine-readable evidence is in `benchmarks/ppml/v045_execution_fix_synthetic.json`.

The release does **not** claim a post-fix real-machine speedup. Certification requires a same-host rerun of the archived workload with `projection_resources`, per-method separation timings, peak RSS, and a separately labelled FE-only sensitivity run.

<!-- econhdfe architecture note -->
> **Lineage note:** this document records validation/benchmark work from the pre-rename `pyreghdfe` lineage. The `econhdfe 0.2.0` package preserves the numerical implementations and regression tests but did not rerun every historical external/performance matrix. Treat the figures below as inherited evidence until the corresponding `econhdfe` benchmark is rerun.


## 0.1.0a3 PPML performance gate

The a3 PPML pass deliberately leaves the 3+ FE numerical method unchanged and optimizes only execution/memory behavior. Local release gates completed:

- 1M two-way PPML same-run replica/optimized: 15.09 s / 4.52 s (3.34x), coefficient difference at machine precision.
- 5M optimized two-way robust PPML: 19.23 s, ~1.36 GiB max RSS.
- 10M optimized two-way robust PPML: 45.75 s, ~2.50 GiB max RSS under the 4 GiB cgroup.
- Repeated 10M split exp/deviance state update stable across eight calls (0.166--0.182 s).
- 5M streamed covariance: robust 0.44 s, one-way cluster 0.59 s, two-way cluster 3.71 s.
- 144k true-three-FE gravity remains numerically identical across engines; no a3 claim is made about a new 3+ FE algorithm.

The remaining performance certification items are target-hardware NUMA/multi-socket scaling and external Stata `ppmlhdfe` parity; neither is inferred from these container results.
# pyreghdfe 0.8.0 performance release check

## Passed locally

- Functional/regression suite: 115 passed.
- 10M complex hierarchical interaction-FE + event-study + two-way cluster: ~19.60 s, ~2.82 GiB peak RSS, 12 two-way iterations, max identified-parameter error ~2.96e-4.
- 1M generic four-FE / 12-control / two-way-cluster sanity: ~5.93 s, 16 iterations, ~606 MiB peak RSS.
- Runtime/execution planning is isolated from the econometric solver layer and records its resolved workspace plan in result metadata.

## External target-hardware gate

The generic 10M four-FE / 12-control sustained-memory workload is not certified from this 4 GiB cgroup run. Repeated 10M stress phases exhibited environment-sensitive sustained OpenMP/Numba memory-bound throughput at high resident working sets. No container-specific econometric/kernel patch was added. Validate this case on the production-memory/NUMA host before making universal 10M generic-runtime claims.


## 0.2.0 IV-PPML note

No new 3+FE algorithm is introduced. Two-way IV-PPML inherits the shared optimized weighted-HDFE path; true Class-C 3FE remains close to replica speed. SPJ/bootstrap adds repeated-estimation workloads and therefore defaults to serial execution unless `n_jobs` is explicitly requested. Production parallel-bootstrap claims require memory/thread-budget validation on the deployment host.
