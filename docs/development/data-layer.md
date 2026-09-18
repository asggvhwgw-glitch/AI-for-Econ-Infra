# Econometric data layer — repeated-workflow optimization

## Economic purpose

The data layer is not primarily an out-of-core system. Its main target is the ordinary empirical workflow in which a researcher repeatedly estimates related specifications on a **medium-to-large, wide dataset**: firm panels, customs/product data, worker-firm panels, scanner data, or similarly structured microdata.

The economic problem is upstream of the estimator. A regression table may use only 10–20 variables from a 100+ column source, while the same firm/year/cluster identifiers and numeric controls are parsed, validated, factorized and converted repeatedly across dozens of specifications. Once HDFE/OLS/PPML kernels become fast, those repeated preparation costs can become a material fraction of total wall time and peak memory.

The data layer therefore answers two questions:

1. **Which raw columns are needed anywhere in this empirical workflow?**
2. **Which validated/encoded column state can be safely reused across specifications?**

It does not change the estimand, choose instruments, define fixed effects, perform PPML separation, or compute covariance formulas.

## Current execution contract

```text
Repeated economic specifications
            |
            v
frontend.columns
  union raw-column + role requirements
            |
            v
data.source
  projected materialization
            |
            v
data.dataset
  immutable EncodedEconometricDataset
  - compact identifier codes
  - one-time column signatures
  - one-time validity metadata
  - reusable numeric arrays
            |
            v
sessions
  FE/sample cache + within-column cache
            |
            v
existing HDFE / estimator / inference
```

The ordinary single-estimator DataFrame path remains available and unchanged. The encoded dataset is most useful for repeated-specification sessions and wide-source projection; it is not forced onto every regression.

## EncodedEconometricDataset

`EncodedEconometricDataset` is an immutable in-process snapshot of the columns retained for estimation. It is deliberately narrower than a generic DataFrame replacement.

It provides:

- one-time snapshotting of retained columns;
- stable `int32` encoding for **identifier-only** FE/cluster/group variables;
- preservation of original values for variables that enter explicit regressors/factors, so reference-level and coefficient-label semantics are unchanged;
- one-time content signatures used by repeated-session cache validation;
- packed per-column validity metadata;
- lazy cached float64 numeric arrays;
- a dataset fingerprint and privacy-minimized source metadata.

Because the snapshot is immutable, repeated-session cache validation can consult stored signatures in O(1) time instead of hashing an N-row pandas column during every interactive fit.

## Repeated-specification cache contract

The cache is intentionally layered rather than global.

### L0 — source projection

The union of raw columns and roles is computed across all specifications supplied to `fit_many()`. Wide CSV/DTA/Parquet sources are projected before the estimator sees them.

### L1 — encoded columns

Identifier-only FE/cluster columns are encoded once. Numeric conversion and validity metadata are reused across specifications.

### L2 — FE component cache

Stable encoded FE identifiers seed the existing HDFE component cache. Changing the outcome or controls does not require re-factorizing firm/year IDs.

### L3 — FE-specification cache

Each distinct absorbed-FE specification has its own sample/absorber/DoF entry. A change in FE structure creates a separate exact entry; transformed variables are never reused across incompatible FE systems.

### L4 — within-column cache

Within a fixed FE entry, only newly requested y/x/endogenous/instrument columns need to be residualized. Adding one control does not re-transform all previously cached controls.

## Cache invalidation rules

Correctness takes precedence over reuse.

- **Change y only:** reuse encoded identifiers, FE topology and already transformed x columns; only a new y is transformed if necessary.
- **Add/drop x:** reuse FE state; residualize only newly requested columns.
- **Change FE:** reuse raw encoded columns but create/reuse a different FE entry and corresponding within cache.
- **Change source requirements in a DataSource-backed interactive session:** rebuild the immutable dataset snapshot using the union of all roles seen so far, then invalidate all downstream FE/within state.
- **A column changes role from identifier-only to explicit factor/regressor:** rebuild from the source rather than reusing integer codes that could alter reference-level semantics.
- **Mutable DataFrame sessions:** existing signature validation remains the correctness guard.
- **Encoded datasets:** mutation is not part of the contract; create a new snapshot when source values change.

## DataSource-backed sessions

`OLSHDFESession` and `IVHDFESession` can now accept a supported DataSource/file path without changing their call signatures.

`fit_many()` is the preferred source-backed path because all specifications are visible before materialization:

```text
40 requested regressions
       |
       v
union required columns once
       |
       v
project/encode once
       |
       v
one multi-RHS HDFE preparation
       |
       v
40 model solves
```

Sequential `fit()` calls remain correct. If a later call introduces a new raw column or changes a column's semantic role, the source snapshot is rebuilt and dependent caches are invalidated. This is intentionally more expensive than giving the whole table to `fit_many()`.

## Development evidence

These are development benchmarks, not release guarantees.

### Wide CSV + regression-table workflow

Synthetic workload: 120k observations, 64 raw columns, 40 OLS-HDFE specifications, 14 columns used anywhere in the table.

A full CSV read followed by the existing batched `fit_many()` path took about 5.28 s in one current-environment run. A DataSource-backed session that first projected/encoded the 14-column union took about 3.00 s, about **1.76x faster** end-to-end. The retained encoded column payload was about 12.7 MB versus 61.4 MB for the full DataFrame, a reduction of roughly **79%**.

The purpose of this benchmark is workflow-level cost, not parser microseconds.

A separate-process RSS benchmark (100k observations, 64 raw columns, 30 specifications) measured about 2.92 s / 397 MB peak RSS for full-width CSV + `fit_many()` versus about 2.19 s / 358 MB for the projected encoded source session: roughly **1.34x** total speedup and **9.9%** lower process peak RSS in that run. The much smaller RSS improvement than retained-payload improvement is expected for row-text CSV parsing and is explicitly not presented as an out-of-core result.

### Repeated specification patterns

On a 150k-observation in-memory panel with 30 sequential fits per pattern, the immutable encoded dataset reduced repeated preparation/validation cost materially in the current environment:

- changing outcomes with fixed x/FE: roughly 2–3x in sequential-fit runs;
- adding/dropping controls: roughly 2x;
- alternating FE sets: roughly 2x, helped by pre-encoded FE IDs.

However, when the researcher already passes the whole specification table to `fit_many()`, the existing Session implementation already batches signature checks and residualization efficiently. In repeated warm runs, the incremental encoded-dataset gain can shrink to only a few percent. This is an important stop line: the data layer should complement `fit_many()`, not duplicate its optimization.

## Deliberate non-goals

- No fully streaming HDFE/PPML execution is claimed.
- No distributed dataframe/Spark/Dask execution is introduced.
- No GPU ingestion layer is introduced.
- No custom Stata parser is introduced.
- Missing-value/drop-sample semantics remain unchanged.
- The data layer does not force an encoded cache for ordinary one-off in-memory regressions.

## Backend boundary

CSV projection can reduce retained width substantially but the parser still scans row text, so peak parser RSS may fall much less than final payload. Stata performance is backend-dependent: pandas selected-column reading is retained as a compatibility path, while optional pyreadstat/ReadStat is preferred when available and compatible with requested options. Parquet remains naturally aligned with projection but is optional.

## Privacy boundary

Default profiles report source type, counts, timings, retained bytes, cache counts and strategy. Raw file paths and raw column names are not emitted in ordinary profiles.

## Next decision point

The next data-layer feature should be justified by **workflow-level evidence**. A persistent on-disk encoded cache should only be added if repeated parsing across separate research runs remains a dominant cost after projected DataSource sessions. Fully out-of-core HDFE should remain lower priority unless real workloads regularly exceed practical workstation memory.

## Experimental persistent research sessions

The phase-3 experiment adds an explicit, opt-in cross-process cache for the exploratory workflow in which a researcher runs a specification, inspects the result, exits or edits the script, and later changes the outcome/control set while keeping much of the same data and FE structure.

It is enabled on a linear repeated-specification session rather than through estimator constructors:

```python
session = OLSHDFESession(CSVSource("panel.csv"))
session.enable_persistent_cache(".econhdfe-cache")
result = session.fit(y="y", x=["x1", "x2"], absorb=["firm", "year"])
```

The cache is **disposable**. Removing the directory must only remove reuse; it must never change an estimate.

### What is persisted

The prototype persists three kinds of reusable state using versioned JSON manifests plus `.npy` arrays; it does not use pickle.

1. **Encoded source columns.** Numeric columns and identifier-only FE/cluster codes can be restored without reparsing every previously used column. If a later specification introduces a new outcome/control, only the missing representation is read from the source. A column that changes from identifier-only to an explicit factor/regressor receives a distinct representation and is reread rather than reusing semantically unsafe integer codes.
2. **Linear within-transformed columns.** OLS/linear-IV sessions may restore a previously residualized x/y/endogenous/instrument column only when the FE/sample/weight/HDFE signature and the column content signature are identical. This cache is deliberately not generalized to PPML weighted-IRLS states.
3. **Completed OLS/linear-IV results.** A deterministic result key includes the specification, all relevant variable/FE/weight/cluster identities, HDFE/inference configuration, collinearity policy and estimator-specific IV options. An exact hit returns the completed result without rerunning HDFE or the model solve. `fit_many()` checks completed specifications before prewarming the unfinished part of a table, so a partially completed robustness suite can resume rather than starting from column 1.

The FE absorber object itself is **not** serialized. Benchmarking showed that, once Numba/runtime state is warm, persisting the full two-way/generic absorber topology would add substantially more cache-format/version coupling than the measured construction cost justifies.

### Source validation modes

Persistent reuse introduces a source-identity problem that does not exist for an in-process immutable dataset. The prototype therefore makes the trade-off explicit.

- `source_validation="strict"` (default) hashes the complete file bytes plus parser options on every new process before reusing persistent state. This is the conservative correctness mode.
- `source_validation="metadata"` fingerprints the resolved file identity plus device/inode, byte size, nanosecond mtime/ctime and parser options. Normal file rewrites invalidate the cache, but this is not a cryptographic content certificate: deliberately restoring filesystem metadata could defeat it. It is intended only for a user's local disposable research cache.

Example:

```python
session.enable_persistent_cache(
    ".econhdfe-cache",
    source_validation="metadata",
)
```

### Measured stop line

The experiment is useful, but its benefit is narrower than an in-process `Session` cache.

With **strict full-file validation**, a 150k-row, 32-column CSV benchmark showed essentially no end-to-end benefit when the specification changed: changing only y was approximately parity and adding one control was slightly slower in that run. The disk/cache work and complete source hash offset the saved HDFE transformation. Exact-result resume remained useful: the estimator-side work was about 4.4x faster and full fresh-process wall time about 1.36x faster.

With **metadata validation**, a 300k-row, 16-column benchmark showed a clearer crossover: changing y with the same x/FE was about 1.21x faster inside the estimation call and about 1.18x faster including process startup; adding one control was about 1.22x faster inside the call but only about 1.06x end-to-end. Exact-result resume skipped estimation almost entirely (roughly 214x estimator-side in that run), while full-process speedup was about 1.58x because Python/import startup is irreducible. The resulting cache after the small exploratory sequence was about 36 MB.

Completed-result entries currently store the full serializable `RegressionResult`. When fitted/residual arrays are present, disk consumption is therefore O(N) per cached completed specification rather than O(K). This is acceptable for an explicit development checkpoint, but it is a real storage trade-off: there is no automatic eviction policy yet, and large robustness suites should treat the cache as disposable acceleration rather than archival output.

These are development measurements, not release promises. They imply three design decisions:

- persistent caching remains **opt-in**, not a default execution path;
- completed-specification resume is the strongest and most generally defensible use case;
- cross-process within-column caching is worth retaining for larger/repeated linear-HDFE work, but it does not justify persistent FE-topology serialization or a broader out-of-core framework at present.

### Non-goals

- No PPML/IV-PPML iteration checkpoint is stored.
- No Krylov/MAP iteration vector is restored mid-solve.
- No absorber object is pickled.
- No cache entry is authoritative data; source + specification remain authoritative.
- No automatic persistent cache is created for ordinary one-off regressions.
