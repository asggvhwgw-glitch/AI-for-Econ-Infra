## 0.6.1 correctness boundaries

Do not accept one-cluster inference, invalid t degrees of freedom, non-finite
recovery inputs, or LSMR condition-limit stops as successful inference. For
mathematical claims separate proof assumptions, implementation evidence and
historical novelty. The three registered manuscripts have a 0.6.1 review, not a
priority certificate. Host-dependency target smoke is not a clean install;
remote CI configuration is not a record of remote execution.

# Advanced validation and testing

Use this guide for replication, numerical auditing, performance qualification and release-grade testing. Ordinary empirical users usually do not need the full matrix.

## Installed-package smoke checks

From the skill directory:

```bash
python scripts/check_environment.py
python scripts/smoke_test.py
```

These verify that an installed environment can import and run core estimators. They do not certify Stata parity, GPU performance or extreme-scale behavior.

## Source-tree internal gate

When the source tree is available, run the repository's controlled validation rather than reconstructing commands from memory:

```bash
python -m pytest -q
python scripts/version.py gate
```

For a release build, use the repository release script after the maintenance/compatibility gates are complete.

## Public-contract compatibility

The repository stores snapshots under `compatibility/baselines/`. The release gate compares the current package against the previous release for:

- `econhdfe.__all__` public symbols;
- callable signatures;
- result dataclass fields;
- public configuration fields/defaults;
- structured error classes/codes/stages;
- exported dataclass schemas.

Inspect pending differences with:

```bash
python scripts/compatibility.py diff
```

Any change must be explicitly approved with a reason before release:

```bash
python scripts/compatibility.py approve <change-id> --reason "why this public change is intentional"
python scripts/compatibility.py check
```

Do not approve a diff merely to make CI green. First decide whether the change is necessary, backward compatible, deprecated properly, and reflected in docs/tests.

## Consequential empirical validation

For a paper replication or a new estimator path, validate at the level relevant to the claim:

1. deterministic unit/regression tests;
2. known analytical/small-matrix references where possible;
3. Stata/R/upstream golden parity when claiming compatibility;
4. simulation coverage for identification/inference claims;
5. target-hardware performance only when making performance claims;
6. reproducibility from a clean install, not only the developer source tree.

Never infer parity from a handful of matching coefficients. Compare sample, omissions, DoF, covariance, finite-sample corrections, diagnostics and convergence semantics.

## Cluster-inference validation

For changes under `inference/cluster`, validate more than bootstrap reproducibility:

1. the observed scalar WCR/WCU statistic must match the fitted one-way CRV1 t statistic under the same finite-sample convention;
2. full Rademacher enumeration for small G should match an independent explicit-OLS construction;
3. serial and parallel runs must agree to numerical precision for identical seeds/draw partitions;
4. weighted aweight/pweight/fweight paths must preserve the fitted WLS metric;
5. multi-way WCB must fail loudly until a separately certified algorithm is implemented;
6. diagnostics flags must remain advisory rather than mutating the estimator specification.

Do not promote the current delete-cluster CRV3 prototype into the public API without profiling memory/state duplication and large-G HDFE refit cost.

## Stata/upstream parity

Consult the source tree's `docs/development/external-validation.md`, `docs/development/upstream-references.md`, validation scripts and test-status documents. Licensed Stata or official binary/reference corpora are external gates when the development environment does not contain them.

Record the exact Stata/package versions and commands used. Upstream-version-specific quirks should be classified rather than silently normalized away.

## Performance testing

Benchmark the changed code path, not an unrelated favorable workload. At minimum record:

- observation count and RHS width;
- FE count/cardinalities and interaction structure;
- solver/projection backend;
- thread count and effective CPU quota;
- memory limit/headroom and peak RSS if available;
- wall time after JIT warm-up where applicable;
- numerical parity/tolerance against the reference path.

For >=10M observations, use the target production host when drawing deployment conclusions. A constrained CI/container benchmark is useful for regressions but is not a universal hardware recommendation.

## Bootstrap validation

Check seed reproducibility, failed-replication accounting, worker/thread oversubscription and resampling-unit semantics. Model-specific bootstraps should reuse the shared resampling infrastructure rather than reimplementing generic parallel/seed/failure loops.

## Failure-path validation

Test invalid inputs and non-convergence as deliberately as successful fits. Public failures should be `EconHDFEError` subclasses with stable machine-readable metadata at the documented stage. Native `KeyError`/`TypeError` leakage at a public boundary is a compatibility bug unless explicitly part of the contract.
## Real-world PPML benchmark discipline

For PPML comparisons, preserve the exact separation policy in the formal apples-to-apples benchmark. Record `diagnostics["projection_resources"]`, `diagnostics["separation_seconds"]`, `diagnostics["separation_iterations"]`, and `diagnostics["separation_solvers"]` together with wall/fit time and peak RSS.

When comparing against the 2026-09-12 pre-v0.4.5 real-machine record, rerun on the same host with the intended thread count and memory policy and append a new dated artifact. Never overwrite the pre-fix record. An FE-only run is useful for decomposition but is not the formal default-separation result.


## User-authorized real-data benchmark

When a real dataset would materially improve validation, ask for explicit permission before using it for benchmarking. After permission, preserve source data and replication scripts read-only, run apples-to-apples reference/econhdfe specifications, and write the result using `benchmark-report-template.md`. See [benchmarking.md](benchmarking.md) for the required environment, parity, timing, diagnostics, failure and artifact fields.

Do not report a speedup for rows that fail specification/sample parity. Keep parity status and performance status separate.
