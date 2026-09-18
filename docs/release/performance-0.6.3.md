# 0.6.3 PERF-01: measured local integration

Baseline: unchanged 0.6.2 source. Candidate: actual 0.6.3 source, not a runtime monkeypatch. Only the NumPy weighted/unweighted column-reduction fusion is integrated. `_sym_projection`, QR/SVD and the GPU path are unchanged.

## Measurement design

Two fresh-process rounds for each case and version; reverse baseline/candidate order in round 2. One warmup and seven timed repetitions per process (14 hot timings per version/case); medians below pool the repetitions. NumPy/BLAS/OMP/Numba use one thread. Profiling is run separately from the timed repetitions. Nonlinear fixtures have separation=(). See the outer evidence archive for all timings, input seed, call options and environment.

Local predeclared retention criterion: no case regresses more than 10% in the pooled median, at least two cases improve by more than 5%, and numerical/stopping guards pass. This is a local experimental criterion, not a universal CI timing threshold.

| Case | Rows | 0.6.2 ms | 0.6.3 ms | Time reduction |
|---|---:|---:|---:|---:|
| ols_3fe | 50,000 | 92.92 | 77.26 | 16.9% |
| ols_3fe | 200,000 | 308.48 | 270.56 | 12.3% |
| ppml_2fe | 10,000 | 32.63 | 22.69 | 30.5% |
| ivppml_2fe | 10,000 | 56.05 | 48.44 | 13.6% |

All four comparisons have observed maximum absolute coefficient and VCOV differences of zero, with the same reported convergence/iteration fields. This does NOT prove bitwise equality for every input/layout/platform. Targeted tests independently check summation error, explicit weighted projection, dynamic and zero weights, and stopping boundaries.

## Limits

This is one Linux / Python 3.13.5 / NumPy 2.3.5 / SciPy 1.17.0 / Numba 0.65.1 environment, maximum 200k rows. It is not a GPU, specialized Schur, cold-start, separation-diagnostic or historical 10M benchmark. Process/order noise affects absolute times and percentages; do not add speedup percentages from different experiments.

PERF-02 buffer reuse is NOT included; PERF-03–06 remain measurement candidates in TODO.md. Statistical formulas, precision, tolerances, failure checks and requested diagnostics were not removed for speed.

## Reproduction

Use `scripts/benchmark_release_local.py --source <unpacked-version> --case <case> --n <rows> --repeat 7 --out <json>` in a new single-thread process per version/case/round. Repeat with reversed ordering; do not run other numerical workloads concurrently. The source ZIP for 0.6.2 is needed for a real baseline.

NumPy API semantics: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html . `optimize=False` avoids choosing a path with a large intermediate; it is not a precision relaxation. The measurements are local data, not a performance guarantee from that documentation.

## Temporary allocation probe

For 200,000-by-12 preallocated input arrays, tracemalloc observes about
19,201,640–19,266,680 temporary bytes in the old reduction versus 1,561–1,803
bytes in the fused reduction, in both C and Fortran layouts. This is NumPy/Python
tracked allocation inside the reduction only, NOT full-model or process RSS.
Input arrays themselves use 38,400,000 bytes and are outside that measurement.

The large Fortran-layout dot products differ by up to 6.14e-12 from the old
summation order; this is why no global bitwise-identity promise is made.
The C-layout probe has zero observed difference. Independent accuracy guards
remain the acceptance criterion, not mandatory last-bit agreement.
