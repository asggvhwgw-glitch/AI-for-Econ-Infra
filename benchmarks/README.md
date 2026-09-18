# Benchmark layout

Benchmark evidence is grouped when a stable domain boundary exists:

- `hdfe/`: HDFE numerical-solver and exact-rank evidence.
- `ppml/`: PPML benchmark programs and results.
- `repeated/`: repeated-specification/session benchmarks.
- `real_world/`: externally executed, user-supplied real-data benchmark records; preserve original runs immutably and append follow-ups.

Some inherited linear/HDFE benchmark scripts remain at this directory level to avoid path churn in historical tooling. New benchmark artifacts should be placed in the appropriate domain subdirectory rather than adding more root-level files.

## Real-machine benchmark return format

Use `real_world/BENCHMARK_REPORT_TEMPLATE.md` for new user-authorized real-data benchmarks. Preserve prior dated records; never overwrite historical benchmark evidence. A benchmark report must separate parity from speed, state timing scope/cold-vs-warm behavior, record machine/package versions and resource settings, and avoid embedding raw/private data.
