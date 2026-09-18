# Real-world benchmark record — 2026-09-12

This directory archives a user-supplied local-machine benchmark run and its PPML diagnosis.
The records are archived verbatim in the release bundle. The copies in this public repository are
redacted for local paths and confidential dataset labels — see `provenance.json`. Do not overwrite
them when a new build is rerun; add a dated follow-up artifact instead.

## Evidence status

- The OLS/IV and complex three-way-FE rows are retained as external real-world performance/parity evidence.
- The PPML rows are a **pre-v0.4.5 baseline**. They are valid observations of the installed code path,
  but the supplied diagnosis found that `ExecutionConfig.threads` and `memory_budget_mb` did not
  reach the direct PPML projector. Consequently they are not a strict configured-4-thread resource-policy claim.
- The PPML default comparison remains estimator-comparable: both sides used the default
  `fe + simplex + relu` separation family according to the supplied diagnosis.
- A post-fix real-machine rerun should be stored alongside these files rather than replacing them.

See `provenance.json` for hashes and classification.
