# Benchmarks — 0.1.0a1

The benchmark harness is `benchmarks/bench_ppml.py`. It uses deterministic
synthetic data and disables separation so timings isolate PPML IRLS, HDFE
absorption, the final exact solve and VCE. Replica and optimized timings in
one row come from the same process/run; speedups are never mixed across runs.

## Latest local results

| Scenario | N | Replica | Optimized | Speedup | coefficient max abs diff |
|---|---:|---:|---:|---:|---:|
| two-way crossed FE | 200,000 | 1.537 s | 0.642 s | 2.393x | 8.33e-17 |
| two-way crossed FE | 1,000,000 | 7.337 s | 3.433 s | 2.137x | 1.80e-16 |
| gravity 3FE | 144,000 | 0.978 s | 0.812 s | 1.205x | 0 |
| hierarchical 4FE | 240,000 | 1.593 s | 0.718 s | 2.218x | 8.33e-17 |

The hierarchical DGP requests `firm + year + province#year + city#year` and
constructs an exact `city -> province` mapping. Canonicalization proves that
`year` and `province#year` are spanned by `city#year`, leaving `firm +
city#year` for the numerical solve.

The gravity DGP is a genuinely crossed `pair + exporter#year + importer#year`
problem. It cannot use the specialized two-way Schur solver, so its smaller
speedup is a useful control: most of the remaining gain comes from compiled FE
codes/indexes/workspaces rather than structural dimensionality reduction.

Peak RSS in the JSON files is process high-water RSS, not clean incremental
memory, because replica and optimized runs occur in the same process. It must
not be used as an engine-to-engine memory comparison.

**No number above is a Stata speedup.** Licensed-Stata same-machine benchmarking
is a separate external certification gate.
