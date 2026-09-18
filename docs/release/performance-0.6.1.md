# 0.6.1 correctness/performance trade-off

A same-host deterministic microbenchmark compares the original 0.6.0 source with
0.6.1 (50,000 observations; OLS has 12 columns; one warm-up and five measurements;
BLAS/OMP threads=1). Kernel medians, not end-to-end FE pipeline timings:

| Kernel | 0.6.0 (s) | 0.6.1 (s) | Ratio |
|---|---:|---:|---:|
| OLS_IID | 0.021743 | 0.055876 | 2.570 |
| OLS_robust | 0.022842 | 0.051562 | 2.257 |
| IV_2SLS_robust | 0.083127 | 0.209959 | 2.526 |

The stable QR/SVD and role-equilibration safeguards have a real cost here.
Do not claim no performance regression: the measured kernels are roughly
2.3–2.6 times slower in this small probe, with absolute medians about 0.05–0.21s.
No old 10M benchmark was rerun. This probe is not a full pipeline or memory
benchmark and is not a universal hardware forecast. Regular-scale coefficient
and covariance agreement stays at round-off (see the JSON). Correctness is the
priority of this patch; further speed work must preserve the unit-invariance
and failure-state tests rather than restoring unscaled Gram inverses.

Evidence: `benchmarks/release/v061_correctness/same_host_dense_probe.json`; raw
measurements and the executable probe are in the separate revision-evidence ZIP.
