# Same-host bounded performance probe — 0.6.1 to 0.6.2

Date: 2026-09-14. Single-thread BLAS/OMP/Numba; fresh process for each version and
case; identical seed/input; one warmup and five timed repetitions. Profiling is
performed separately. This is not a historical 10M or cross-platform benchmark.

| Case | N | 0.6.1 median seconds | 0.6.2 median seconds | Ratio | 0.6.2 process peak MiB |
|---|---:|---:|---:|---:|---:|
| iv_kernel | 50000 | 0.009059 | 0.008612 | 0.951 | 202.3 |
| ivppml_2fe | 10000 | 0.058689 | 0.054456 | 0.928 | 226.9 |
| ols_3fe | 200000 | 0.352916 | 0.321639 | 0.911 | 351.9 |
| ols_3fe | 50000 | 0.102358 | 0.095588 | 0.934 | 252.0 |
| ols_kernel | 50000 | 0.015232 | 0.006702 | 0.440 | 199.5 |
| ppml_2fe | 10000 | 0.032641 | 0.032278 | 0.989 | 223.0 |

`ols_kernel` is robust OLS estimate+VCE, not just matrix multiplication.
`iv_kernel` is weighted 2SLS solve+bread, not the full IV inference pipeline.
Process peak RSS includes imports, data generation, warmup, repeats and the
separate profile; it is not isolated solver allocation. Repetitions share one
process per version/case; small ratio differences should not be overinterpreted.
The largest observed candidate process peak is below 352 MiB for these cases.
All full-model cases converged. Coefficients differ by at most 1.67e-16 and
covariance/bread by at most 5.70e-19 on these ordinary-scale fixtures.

The OLS kernel median falls from 15.23 to 6.70 ms. End-to-end gains are smaller:
3-FE OLS 0.93/0.91 ratios at 50k/200k, 2-FE PPML 0.99 and IV-PPML 0.93 at 10k.
This does not establish universal speedups or restoration of every historical
0.6.0 performance claim. The unmodified 0.6.1 source is the measured baseline.

Reproduce with `scripts/benchmark_release_local.py --help`, using explicit
source directories and identical thread environment settings. Raw JSON includes
all timings, beta, covariance/bread, RSS, and the top 25 profiled package functions.
It lives in the accompanying evidence archive under `benchmarks/`.
