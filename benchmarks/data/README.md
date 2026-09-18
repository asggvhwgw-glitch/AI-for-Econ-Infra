# Data-layer benchmarks

These benchmarks measure source-format behavior before estimator execution. They are development evidence, not universal I/O claims.

- `ingestion_projection.py`: wide CSV full read versus specification-projected read.
- `ingestion_projection_stata.py`: wide Stata full read versus selected-column pandas fallback.
- `identifier_encoding.py`: memory/time effect of converting identifier-only string FE columns to stable int32 codes.

Each benchmark uses separate child processes for full/projected reads so peak RSS is not contaminated by a previous read in the same process. Import/startup time is excluded from the timed I/O region.

The Stata benchmark intentionally records that pandas selected-column reading may be slower and use more parser peak RSS; this is why `StataSource` supports an optional ReadStat/pyreadstat backend rather than assuming pandas `columns=` is a performance optimization.

- `bench_persistent_session.py`: cross-process exploratory-session benchmark. It compares a fresh source-backed OLS-HDFE run with persistent cache build, changed-y reuse, add-control reuse, and an exact completed-result resume. `persistent_session_resume_strict.json` records conservative full-file hashing; `persistent_session_resume_metadata.json` records opt-in filesystem-metadata validation. The latter is a local research-cache optimization, not a cryptographic source certificate.
