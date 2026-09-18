# econhdfe 0.6.3

> **0.6.3 local release candidate.** A narrow CPU weighted column-reduction
> optimization is integrated without changing estimator equations, tolerances,
> QR/SVD or the GPU path. The [unified TODO](TODO.md) is now in the source tree.
> [Execution acceptance](docs/release/acceptance.md) separates review completion
> from real validation; candidate validity is not formal release authorization.
> See [measured performance](docs/release/performance-0.6.3.md),
> [migration notes](docs/release/migration.md) and
> [actual validation status](docs/development/test-status.md).
> Clean installation, remote CI and unexecuted optional/reference environments
> are not certified by local tests. Originality status and mathematical scope
> remain unchanged.


`econhdfe` is a high-performance econometrics package built around a shared high-dimensional fixed-effect (HDFE) and numerical-computing core. OLS-HDFE, linear-IV-HDFE, PPML-HDFE, and IV-PPML-HDFE are estimator families that consume that core; none of them defines the package architecture.

The project was refactored from the earlier `pyreghdfe` namespace after PPML and IV became first-class components. Existing `pyreghdfe` imports remain supported through a compatibility namespace, but new code should use `econhdfe`.

## HDFE computation in 0.4.4

Version 0.4.0 integrates the exactdof2 multiway categorical-rank engine. The default `dof_method="pairwise"` remains unchanged for compatibility and predictable cost; advanced users can opt into `dof_method="exact"` for certified characteristic-zero absorbed DoF with three or more intercept-only categorical FE partitions. The exact path deduplicates FE tuples, performs exact degree-one peeling, decomposes residual hypergraph cores, uses a structural proper-connectivity certificate and bounded GF(2) certificate, then falls back to optional SymPy/FLINT or native exact arithmetic only when required. Numerical FE absorption now adds the solver-opt2 multiway path: exact degree-one numerical core reduction for eligible 3+ pure-intercept systems, an opt-in adaptive MAP/CG planner, and bounded fused weighted multi-RHS projection. The direct `HDFEAbsorber` default remains CG for backward compatibility, while optimized `FEPlan` uses the adaptive planner. The specialized two-way solver remains unchanged.

Solver canonicalization and inference topology are now explicitly separated: dropping a redundant FE from the numerical solver cannot change requested-design DoF or cluster-nesting accounting.

## Primary API

```python
from econhdfe import olshdfe, ppmlhdfe, ivhdfe, ivppmlhdfe
```

OLS-HDFE:

```python
r = olshdfe(
    df,
    y="y",
    x=["x1", "x2"],
    absorb=["firm", "year"],
    cluster=["firm", "year"],
    vce="cluster",
)
```

PPML-HDFE:

```python
from econhdfe import ppmlhdfe, PPMLConfig

r = ppmlhdfe(
    y_count,
    X,
    absorb=[firm, year],
    clusters=[firm, year],
    vce="cluster",
    config=PPMLConfig(engine="optimized"),
)
```

IV-HDFE:

```python
r = ivhdfe(
    df,
    y="y",
    exog=["control"],
    endog=["price"],
    instruments=["z1", "z2"],
    absorb=["firm", "year"],
    estimator="liml",
    vce="robust",
)
```


Repeated PPML specifications can reuse a previous converged predictor:

```python
model = PPMLHDFE(df, absorb=["firm", "year"], config=PPMLConfig(engine="optimized"))
r1 = model.fit("trade", ["distance", "fta"])
r2 = model.fit("trade", ["distance", "fta", "tariff"], warm_start=r1)
```

Warm starts change only the numerical starting point; the final estimating equations and convergence tolerances are unchanged.

Compatibility aliases remain available:

```python
from econhdfe import reghdfe, ivreghdfe
# and legacy: import pyreghdfe
```


## Identified categorical fixed effects

Version 0.6.0 adds a separate post-estimation recovery layer for additive categorical/indicator fixed effects. It does not change the estimator APIs.

```python
from econhdfe.effects import recover_linear_result, NormalizationSpec

fe = recover_linear_result(
    fitted_result,
    df[["x1", "x2"]].to_numpy(),
    [df["worker_id"].to_numpy(), df["firm_id"].to_numpy()],
    x_names=["x1", "x2"],
    fe_names=["worker", "firm"],
    normalization=NormalizationSpec("weighted_mean_zero", baseline="worker"),
)
firm_effects = fe.term("firm")
```

The recovery layer reports realized-sample rank/nullity and connected components. If an independent component has additional unidentified directions beyond ordinary additive shifts, v0.6.0 salvages every fully identified component and reports the unidentified component's FE levels as unavailable (`NaN`) with a structured diagnosis; a specification whose every component is unidentified still raises `FixedEffectIdentificationError`. Normalization can choose a reference or mean-zero convention but cannot create identification. PPML/IV-PPML adapters recover effects only on the final finite-MLE estimation sample after singleton/separation handling. Continuous/varying-slope FE recovery is intentionally outside the v0.6.0 scope.

## Agent Skill package

The release ships a separate portable Agent Skill, `econhdfe-skill-v0.6.1.zip`. Its canonical source is `skills/econhdfe/` and follows progressive disclosure: `SKILL.md` is only the routing/core-safety layer, while installation, configuration, empirical workflows, advanced validation, and third-party development live in focused `references/` files. Deterministic environment and smoke checks live under the skill's `scripts/`, and OpenAI UI metadata lives in `agents/openai.yaml`.

The skill is intentionally not duplicated at repository root. This avoids the version/content drift that occurred when two full `SKILL.md` copies had to be maintained manually.

## Architecture

The substantive layers are intentionally one-way:

```text
models ───────────────► hdfe ───────────────► compute
  │                      │                       │
  └──────────────► iv ───┘                       ▼
                         data ───────────────► planner
                           ▲                    ▲
                           └──── frontend       └── hdfe
```

- `models`: outcome-model equations and orchestration for OLS, linear IV, PPML and IV-PPML.
- `hdfe`: FE specification, encoding, canonicalization, absorption, specialized solvers, DoF, group/individual FE and iterative weighted projection.
- `iv`: outcome-agnostic instrument roles, additive moment primitives and weighted-2SLS inner solves. Linear-IV estimators/diagnostics live under `models.linear_iv`.
- `planner`: estimator-agnostic exactness/resource policy. It separates legal execution candidates from memory/representation/thread cost decisions, calibrates `threads="auto"` on sufficiently large HDFE jobs against the actual runtime, and imports no estimator/HDFE/data implementation.
- `compute`: low-level kernels, linear algebra, covariance, weights, and execution primitives; runtime resource discovery is shared through the planner layer.
- `resampling`: model-agnostic bootstrap/resampling execution, seeds, cluster-group compilation, parallel scheduling, and failure aggregation.
- `frontend`: role-aware input preflight and deterministic schema validation.
- `data`: projected/encoded estimation-ready data and repeated-workflow reuse; it reports resource needs to the planner but does not define estimator semantics.

A thin root API provides common design objects and result containers. The architecture is documented economics-first: `docs/development/architecture.md` explains which empirical problem each layer solves, while `docs/development/economic-module-map.md` maps every runtime module to its economic/econometric purpose and its computational responsibility. The generated `docs/development/architecture-map/` remains the source of truth for code dependencies. The documentation index is `docs/README.md`; HDFE theory/solver notes are under `docs/technical/hdfe/`.

Technical originality is governed separately from implementation complexity. `docs/technical/innovation-audit.md` is the package-wide novelty audit and `docs/technical/innovation-registry.json` is the machine-readable release contract. Registration requires a formal LaTeX manuscript and compiled PDF plus implementation/test/evidence mappings, but those artifacts do not establish novelty or historical priority. The 0.6.1 registry classifies the entries as theorem-backed frameworks/applications and explicitly records originality as not independently established. Current registered contributions are limited to exact multiway categorical FE rank/DoF, exact arbitrary-G numerical residual-core reduction, and exact partition-refinement HDFE structural design reduction.

## Shared HDFE infrastructure

OLS, PPML and IV share the same:

- dense FE encoding and interaction coding;
- exact FE canonicalization where partition refinement can be certified;
- MAP/indexed projection and specialized two-way Schur/PCG;
- dynamic `update_weights()` lifecycle for iterative WLS/PPML;
- absorbed DoF machinery;
- structural/post-absorption collinearity handling;
- IID, robust, multiway-cluster, HAC and Driscoll–Kraay covariance infrastructure;
- cluster-inference diagnostics plus one-way OLS WCR11/WCU11 wild-cluster tests for few/unbalanced-cluster sensitivity analysis;
- shared runtime/resource planning plus bounded memory/thread execution.

This means future work on exact 3+ FE rank, new multi-FE solvers, fused kernels or memory planning belongs in one shared layer rather than being reimplemented for each estimator.

## PPML status

The integrated PPML implementation includes ppmlhdfe-style IRLS, adaptive inner tolerances, FE/simplex/ReLU separation, standardization, robust/multiway-cluster VCE, persistent FE topology and optimized weighted HDFE projection. As of v0.4.5, direct and reusable PPML/IV-PPML APIs propagate `ExecutionConfig.threads` and `memory_budget_mb` into their weighted projectors; optimized simplex/ReLU separation also follows the selected optimized execution plan rather than forcing replica-only paths. PPML diagnostics expose the effective projector resources plus per-method separation time/iterations/solver labels. Licensed Stata golden parity and the upstream separation-suite execution remain external certification gates.

## IV and IV-PPML status

The generic `iv` layer remains outcome-model agnostic: it provides `IVDesign`, additive score/moment primitives and the shared weighted-2SLS inner solver. Linear IV-HDFE stays under `models.linear_iv` and owns 2SLS/LIML/k-class/GMM plus linear weak-ID diagnostics.

IV-PPML is a separate model family under `models.ppml_iv`. It solves the additive moment condition `E[q(y-mu)]=0` by iteratively reweighted 2SLS while reusing the same weighted HDFE projector as PPML. It supports multiple endogenous regressors, over-identification, robust/multiway-cluster VCE, offset/exposure, `fweight`/`pweight`, Stata-style `quadvariance` standardization, FE/simplex/ReLU plus optional in-loop `mu` separation, and loud nonconvergence/divergence guards.

The IV-PPML model layer also contains Class A/B/C split-panel jackknife bias correction and cluster bootstrap inference. SPJ/bias logic is deliberately not placed in generic `iv`. The bootstrap returns the usual draw SD, percentile CI and a CI-implied SE. No linear KP/SW/Stock-Yogo statistic is exposed as an IV-PPML diagnostic because the current upstream implementation/paper does not supply a justified IV-PPML analogue.

Example:

```python
from econhdfe import ivppmlhdfe, IVPPMLConfig

r = ivppmlhdfe(
    y, exog=controls, endog=endogenous, instruments=excluded_z,
    absorb=[exporter_year, importer_year], vce="cluster", clusters=[pair],
    config=IVPPMLConfig(engine="optimized"),
)
```

## Validation

The current release is validated by the full local regression suite; see `docs/development/test-status.md` for the release-exact count and source-archive rerun. A licensed-Stata feature-golden harness and an official Class A/B/C corpus comparator are included under `validation/ivppml/`; those remain explicit external certification gates until run on a host with licensed Stata and the upstream binary datasets.

## Naming

`pyhdfe` and `hdfe` are already established Python package names, so this refactor deliberately avoids both. `econhdfe` describes the broader econometric scope while keeping HDFE as the shared computational infrastructure.
## Frontend validation and structured errors

`econhdfe` 0.3.0 added a cheap role-aware preflight layer. Continuous outcomes/regressors/endogenous variables/instruments/weights/offsets/exposures must be numeric; FE, cluster and factor identifiers may remain strings or categoricals. The validator intentionally does not attempt collinearity, FE rank, separation, or weak-identification diagnostics.

```python
from econhdfe import preflight_dataframe

report = preflight_dataframe(df, {
    "trade": "outcome",
    "distance": "regressor",
    "firm": "fixed_effect",
})
report.raise_for_errors()
```

Public estimator failures expose stable structured metadata:

```python
from econhdfe import EconHDFEError

try:
    ...
except EconHDFEError as err:
    print(err.code, err.stage, err.details)
```

The hierarchy separates input, specification, identification, convergence, numerical, inference and resampling failures while retaining compatibility with the corresponding built-in `ValueError`, `RuntimeError`, or `numpy.linalg.LinAlgError` families.

## Python-safe factor-variable expressions

Linear OLS/IV DataFrame estimators accept the existing object API (`factor()` / `reg_interaction()`) and the compact `fv()` frontend. The DSL intentionally uses function-style markers rather than Stata's dot prefixes, so categorical/continuous roles are visually distinct from Python attribute access:

```python
from econhdfe import fv, olshdfe

fit = olshdfe(
    df,
    y="outcome",
    x=["control", fv("i(x)##i(y)##c(z)")],
    absorb=["firm", "year"],
)
```

`#` is interaction-only and `##` is full factorial. Thus `i(x)##i(y)##c(z)` expands to the three main effects, all three two-way interactions, and the three-way interaction. `+` joins terms inside a grouped expression, so `i(x)##(i(y)+c(z))` expands distributively without creating `i(y)#c(z)`. `i(x,y)##c(z)` is a shorthand for `(i(x)+i(y))##c(z)`. Continuous powers such as `c(age)#c(age)` are supported.

Reference levels use keyword syntax: `i(group, base=3)`, `base=first`, `base=last`, `base=freq`, or `base=none`. Quoted column/base names are allowed. The compatibility DSL supports interaction order up to eight. Dot-style forms such as `i.group` or `c.age` are rejected deliberately with a structured syntax error.

The same symbolic `fv()` expression can also be used in `absorb=` for the DataFrame linear OLS/IV path. In absorb context the compiler targets HDFE specifications rather than explicit dummy columns: `fv("i(firm)#i(year)")` is a joint categorical FE, `fv("i(firm)#c(trend)")` is a slope-only heterogeneous FE, and `fv("i(firm)##c(trend)")` is an intercept-plus-slope FE. Full-factorial lower-order absorbed terms are reduced symbolically to an equivalent minimal FE column space before numerical encoding, so `fv("i(firm)##i(year)##c(trend)")` compiles to the joint `firm#year` intercept-plus-trend-slope representation rather than duplicating redundant lower-order slope FEs.

Repeated-spec sessions accept categorical-only `fv()` absorb expressions but intentionally reject heterogeneous-slope `fv()` absorb terms; use direct `olshdfe()` / `ivhdfe()` for those. Low-level PPML/IV-PPML array APIs continue to take already constructed numeric FE arrays and do not gain heterogeneous-slope FE support in this release.

## Resampling infrastructure


### Cluster inference (0.4.6)

The default covariance layer continues to provide CRV1, including arbitrary multi-way Cameron–Gelbach–Miller inclusion/exclusion and existing nesting/small-sample conventions. For one-way clustered OLS where finite-cluster behavior is consequential, fit with `keep_state=True` and use:

```python
from econhdfe import cluster_diagnostics, wild_cluster_test_ols

fit = olshdfe(
    df, y="y", x=["x1", "x2"], absorb=["firm", "year"],
    cluster="state", vce="cluster", keep_state=True,
)

diag = cluster_diagnostics(fit)
wcr = wild_cluster_test_ols(fit, param="x1", reps=9_999, impose_null=True)
```

`cluster_diagnostics()` reports cluster-size and OLS score concentration diagnostics; its flags are review heuristics, not automatic rules for changing the clustering level. `wild_cluster_test_ols()` currently certifies one-way OLS-HDFE WCR11/WCU11 only. Multi-way CRV1 remains supported in the main estimator, but multi-way WCB and CRV3 are not claimed by this release. The legacy coefficient-draw `wild_bootstrap()` remains available and now respects the fitted WLS metric for weighted OLS.

Generic repeated-draw mechanics now live under `econhdfe.resampling`; model-specific bootstrap definitions remain with the model. `models.ppml_iv.bootstrap`, for example, owns the SPJ cluster-bootstrap statistic but uses shared cluster-group compilation and replicate execution. Replicate failures are summarized by structured error code rather than silently swallowing arbitrary exceptions.




## Repeated empirical specifications

For regression-table workflows that repeatedly change outcomes, controls, or FE combinations, use the linear HDFE session API instead of independent estimator calls:

```python
from econhdfe import OLSHDFESession, OLSSpec

session = OLSHDFESession(df, cluster="firm", vce="cluster")
results = session.fit_many([
    OLSSpec("exports", ("treat",), ("firm",)),
    OLSSpec("exports", ("treat", "size", "age"), ("firm", "year")),
    OLSSpec("sales", ("treat", "size", "age"), ("firm", "year")),
    OLSSpec("exports", ("treat", "size", "age"), ("firm",)),
])
```

The session caches exact FE-specific within transforms. Reusing the same FE combination reuses the compiled absorber and already-partialled columns; a genuinely new FE combination still receives its own joint HDFE projection and singleton sample. `fit_many_y()` provides a dedicated multi-outcome path. `IVHDFESession` provides the same lifecycle for linear IV specifications while recomputing specification-dependent first-stage/identification diagnostics.

The optimized session surface currently targets standard DataFrame numeric columns and intercept-only categorical FEs. Factor/continuous-interaction designs, heterogeneous-slope FEs, group/individual FEs, and nonlinear PPML families continue to use their full estimator APIs.

## Publication-ready results and advanced controls

> **Testing/beta status.** For consequential empirical work, randomly select at least one representative specification from every substantive run/table and compare its sample, coefficients, SEs, DoF, and major fit statistics against `reghdfe`, `ivreghdfe`/`ivreg2`, `ppmlhdfe`, or another mature reference when available. Real-user-data performance benchmarking should be run only after explicit user agreement and should use `benchmarks/real_world/BENCHMARK_REPORT_TEMPLATE.md`.

For crashes, material parity mismatches, convergence/numerical anomalies, regressions, or API/documentation defects, use `docs/development/ERROR_REPORT_TEMPLATE.md`. The standard report is privacy-minimized and parameter-only: return anonymous specification structure, counts/configuration, structured error fields, aggregate diagnostics, and parity-difference magnitudes; do not include or reconstruct observations, exact identifiers, paths, commands/scripts, raw logs, full tracebacks, or attachments. Benchmarking is a separate explicit-consent workflow.


Version 0.3.5 adds a common publication surface across OLS, linear IV, PPML and IV-PPML. The default output is deliberately paper-facing rather than solver-facing:

```python
out = result.publication_output()
out["coefficients"]   # estimate, SE, t/z, p-value, CI, stars
out["model"]          # N, DoF, VCE, clusters, FE and model-fit statistics
```

Linear IV additionally exposes reportable first-stage output and the main identification/over-identification tests commonly used in empirical economics. Full internal diagnostics and timing profiles are not included unless requested:

```python
result.publication_output(include_diagnostics=True, include_profile=True)
```

The common coefficient table is available directly through `result.coef_table()`. OLS model statistics expose the four `reghdfe`-style R-squared measures plus RSS/TSS/within-TSS/MSS, RMSE, Gaussian log likelihoods, model F/Prob>F, model DoF, fit residual DoF, and covariance rank. Linear IV exposes the convention-stable partialled-out subset (RSS, within TSS, RMSE, model F/Prob>F and DoF) while retaining econhdfe's existing overall R-squared as an explicitly documented extension rather than claiming strict `ivreghdfe` parity. The adjusted statistics use the fit residual degrees of freedom implied by the absorbed-FE design; they deliberately do **not** reuse the cluster reference-distribution DoF (for example `G-1`). `result.df_resid` remains the inference DoF used by t/F reference distributions. PPML reports original-scale deviance, log likelihood/null log likelihood, pseudo-R2, model Wald chi-square/p-value, model DoF, full pre-separation N, and covariance rank.

Strategy-level advanced controls are grouped rather than flattened into every estimator signature:

```python
from econhdfe import HDFEConfig, InferenceConfig, ExecutionConfig

hdfe = HDFEConfig(solver="auto", tolerance=1e-10, dof_method="pairwise")
inference = InferenceConfig(vce="cluster", confidence_level=0.95, diagnostics="off")
execution = ExecutionConfig(threads="auto", memory_budget_mb=4096, profile="summary")
```

`HDFEConfig` is consumed by OLS/linear-IV where the general HDFE solver is directly selected; PPML/IV-PPML retain their model-specific numerical configs and share `InferenceConfig`/`ExecutionConfig`. Implementation-detail knobs such as Krylov restart thresholds remain private. In PPML/IV-PPML, `ExecutionConfig` is an active resource policy: use `result.diagnostics["projection_resources"]` to audit the actual weighted-projector backend/thread count and `result.diagnostics["separation_seconds"]` to attribute separation cost. The memory budget controls projector/workspace planning; it is not a hard process-RSS limit. For sufficiently large HDFE workloads, `threads="auto"` uses a cached real-runtime synthetic calibration to select a near-saturation thread count; small HDFE jobs stay single-threaded to avoid calibration/parallel overhead. Explicit integer threads override calibration. Machine-specific planner anomalies can be returned through the privacy-minimized `docs/development/PLANNER_REPORT_TEMPLATE.md`; no telemetry is uploaded automatically.

Reusable PPML/IV-PPML model objects now maintain a content fingerprint for compiled FE topology. Under the default signature validation, changing the underlying FE source columns invalidates and rebuilds the cached plan instead of silently reusing stale topology.

Strict frontend preflight is advisory rather than coercive:

```python
report = preflight_dataframe(df, roles, model="ivppml", level="strict")
```

It can warn about few clusters, near-constant continuous variables/instruments, extreme weight ranges, or extreme zero shares. Such warnings do not block estimation; deterministic type/domain errors still do.


### Privacy-safe support reports

Installed builds include `econhdfe-report error` and `econhdfe-report benchmark` for writing the canonical Markdown templates. The error workflow is parameter-only by default. For in-process failures, `econhdfe.support_reports.write_error_report(...)` can extract a strict allowlist of scalar/counter metadata from result/error/config objects without serializing data, coefficient/residual arrays, variable names, paths, error messages, raw logs, or full tracebacks. Real-data benchmarking still requires separate explicit user consent.
