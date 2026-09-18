# reghdfe R-squared / DoF parity validation (v0.4.7)

Version 0.4.7 was checked against the current `reghdfe` Mata implementation (`current-code/Regression.mata`) for fit-statistic degrees-of-freedom semantics. The reference implementation distinguishes three quantities that must not be conflated:

1. the ordinary regression residual DoF before cluster reference-distribution adjustment;
2. the public/inference `df_r`, which clustered VCE may cap at `G-1`; and
3. `used_df_r = N - df_a - df_m - df_a_nested`, used by adjusted overall and adjusted within R-squared.

The package therefore keeps `RegressionResult.df_resid` as the inference DoF while computing adjusted fit measures from an internal fit DoF. The validation corpus covers one-way clustering, FE nested in cluster, linear IV, repeated-spec sessions, and fweight physical replication.

Strict adjusted-R2 parity additionally requires the same absorbed-DoF method. `dof_method="exact"` for 3+ categorical FEs is an econhdfe extension and may legitimately change the adjusted denominator relative to `reghdfe`'s approximate/default multiway DoF accounting.

---


## Cluster inference v0.4.6 external boundary

Local validation certifies the implemented one-way OLS WCR11/WCU11 algebra against independent explicit-OLS enumeration and internal CRV1 conventions. This release does **not** claim licensed Stata `boottest`/R `fwildclusterboot` golden parity, multi-way WCB, IV WCB, PPML score bootstrap, or large-G CRV3 performance. Those require separate external corpora/implementations and should remain labelled unverified until executed.

<!-- econhdfe architecture note -->
> **Lineage note:** this document records validation/benchmark work from the pre-rename `pyreghdfe` lineage. The `econhdfe 0.1.0a3` package preserves the numerical implementations and regression tests but did not rerun every historical external/performance matrix. Treat the figures below as inherited evidence until the corresponding `econhdfe` benchmark is rerun.

# pyreghdfe 0.8.0 — External Validation Plan

This document defines the final validation that should be run outside the current constrained container before claiming production-grade parity with Stata `reghdfe` / `ivreghdfe` or publishing performance numbers for a specific hardware platform.

## 1. Stata golden parity — mandatory for compatibility claims

Run on a machine with licensed Stata MP and current upstream packages pinned to the versions under test.

Recommended setup:

```stata
ssc install ftools, replace
ssc install reghdfe, replace
ssc install ivreg2, replace
ssc install ranktest, replace
net install ivreghdfe, from(https://raw.githubusercontent.com/sergiocorreia/ivreghdfe/master/src/) replace
```

Then run:

```bash
python validation/make_fixture.py
stata-mp -b do validation/stata/generate_golden.do
python validation/compare_golden.py
```

The 23-row v0.8 fixture covers ordinary HDFE OLS/IV, heterogeneous slopes, multiway cluster/DK cases, typed weights, group+individual FE models, hierarchy canonicalization, auto two-way routing, explicit user omission, structural factor collinearity, and event-study reference handling. Treat a difference as a release blocker until it is classified as one of:

- documented upstream normalization difference for saved FE coefficients;
- documented small-sample/DoF convention difference;
- floating-point tolerance only;
- an actual compatibility bug.

Store Stata version, `reghdfe` version, `ivreghdfe` version, `ftools` version and command output with the fixture results.

## 2. Cross-platform install matrix — mandatory

Test wheel install and the full pytest suite on:

- Linux x86_64: Python 3.10, 3.11, 3.12, 3.13;
- macOS Apple Silicon: Python 3.10–3.13;
- Windows x86_64: Python 3.10–3.13.

Commands:

```bash
python -m venv .venv
# activate the environment
python -m pip install -U pip
python -m pip install '.[test]'
python -m pytest -q
```

Also install the built wheel into a clean environment without the source tree on `PYTHONPATH` and run `scripts/smoke_install.py`.

## 3. CPU scaling — mandatory for performance claims

Run `benchmarks/bench_cpu_projection.py` on at least:

- 8-core workstation;
- 16–32 core server;
- >=64-core / dual-socket NUMA server if production deployment targets such systems.

For each host record:

- physical/logical cores;
- CPU model;
- memory channels / total RAM;
- NUMA topology;
- Python, NumPy, Numba, SciPy versions;
- BLAS implementation;
- `absorb_threads = 1, 2, 4, 8, 16, ...`;
- projection backend;
- rows, FE levels, RHS width;
- warm and cold timings;
- peak RSS.

Do not assume all cores are optimal. Group projection becomes memory-bandwidth bound; use the measured optimum.

For dual-socket hosts compare normal execution with NUMA pinning, for example:

```bash
numactl --cpunodebind=0 --membind=0 python benchmarks/bench_cpu_projection.py
numactl --interleave=all python benchmarks/bench_cpu_projection.py
```

## 4. 10M–100M row stress tests — mandatory before very-large-data claims

Run `benchmarks/bench_10m_multife.py` unchanged first, then scale to 25M, 50M and 100M rows as RAM permits.

Minimum production stress specification:

- 10M rows;
- >=12 continuous regressors;
- >=4 HDFE dimensions;
- 300k–500k levels per FE;
- two-way cluster with high-cardinality intersections;
- `tol=1e-8`;
- compare `pool_size` and `memory_budget_mb` settings;
- compare `projection_backend='fused'` and `'indexed'` where feasible;
- record encoding, absorption, DoF, regression/VCE and total times separately;
- record peak RSS.

Additional stress cases:

- highly unbalanced / Zipf-distributed FE sizes;
- nested FEs;
- disconnected FE graph;
- severe singleton pruning;
- many RHS columns (25, 50, 100 controls);
- 3–10 way clustering;
- fweight with effective N materially larger than physical N.

## 5. Numerical hard cases — mandatory

Create deterministic tests for:

- nearly collinear regressors after absorption;
- weakly connected FE graphs;
- heterogeneous slopes with near-zero within-group variance;
- large weight dispersion;
- weak instruments;
- nearly singular instrument matrices;
- many instruments;
- LIML/GMM2S under weak identification;
- singleton cascades.

Compare MAP, CG and LSMR where all are applicable. Report convergence iterations and condition warnings, not just coefficient differences.

## 6. Inference validation — mandatory

Against Stata / trusted independent implementations validate separately:

- IID;
- HC1 robust;
- one-way cluster;
- two-way cluster;
- 3+ way cluster;
- HAC/Newey-West;
- Driscoll-Kraay;
- fweight/aweight/pweight conventions;
- nested-FE small-sample adjustments;
- first-stage diagnostics;
- Cragg-Donald;
- Kleibergen-Paap;
- Sanderson-Windmeijer;
- Stock-Yogo lookup behavior;
- overidentification tests.

For each test compare coefficients, covariance matrix, standard errors, residual DoF, absorbed DoF, effective N and diagnostic statistics.

## 7. Group + individual FE — mandatory

Test patent–inventor / paper–author style long-form data with:

- `aggregation='mean'` and `'sum'`;
- LSMR and LSQR;
- CSR and matrix-free incidence backends;
- individual heterogeneous slopes;
- recursive graph singleton pruning;
- group-level IV;
- saved FE reconstruction;
- highly skewed team sizes;
- >=10M memberships.

Saved FE level coefficients are normalization-dependent. Validate the joint FE contribution and fitted-value reconstruction rather than requiring level-by-level equality where upstream coefficients are not uniquely identified.

## 8. GPU validation — mandatory before GPU performance claims

The current environment did not contain a CUDA GPU. Run on at least one NVIDIA Ampere/Ada/Hopper GPU with a matching CuPy build.

Validate:

```bash
python -m pip install '.[gpu,test]'
```

Then compare CPU vs GPU coefficients and absorbed matrices at tight tolerances. Benchmark host-to-device transfer separately from steady-state absorption. Test GPU memory exhaustion and verify that the documented CPU fallback remains available.

Do not claim that the new CPU `indexed` backend is used on CuPy; it is a CPU/Numba backend.

## 9. Bootstrap concurrency — mandatory

Test wild/bootstrap workloads at combinations such as:

- `n_jobs = 1, 2, 4, 8`;
- `absorb_threads = 1, 2, 4, 8`;
- fixed total core budgets.

Verify deterministic output for equal seeds. Monitor actual runnable threads and confirm the worker/FE-thread budget prevents nested oversubscription.

## 10. Memory/error handling — mandatory

Verify clean failures and useful messages for:

- insufficient index memory budget;
- forced indexed backend on unsupported GPU/solver configurations;
- invalid weights;
- fweight + unsupported IV HAC/DK combination;
- inconsistent long-form group variables;
- duplicated `(group, individual)` memberships;
- non-convergence at low `max_iter`.

## Release gate

A production release should have all of the following archived:

1. clean wheel install logs on the OS/Python matrix;
2. full pytest results;
3. Stata golden parity report with pinned upstream versions;
4. 10M benchmark results and at least one larger stress run;
5. CPU scaling results on the intended deployment hardware;
6. GPU results if GPU acceleration is advertised for that release;
7. SHA256 hashes for source archive and wheel.

## dev5 event-study / omission validation additions

For Stata/reference parity add specifications with an explicitly chosen event-study base period and verify: (1) the same estimation sample, (2) the same reference coefficient is omitted, (3) every automatic omission is separately identified, (4) coefficient/SE equivalence under an algebraically equivalent factor-base parameterization, and (5) no unmatched omission selector is silently accepted. Include a 10M interaction-heavy specification matching `bench_10m_complex_eventstudy.py` on a production-memory server and compare peak RSS plus two-way multi-RHS solver timing.



## v0.8.0 mandatory target-hardware runtime acceptance

For 10M+ generic 3+ FE workloads, record CPU affinity/quota, physical RAM or cgroup limit, current free headroom, NUMA topology, `absorb_info["execution_plan"]`, FE cardinalities, RHS width, iterations, wall time and peak RSS. The local 4 GiB release cgroup did not provide a stable sustained-memory result for the generic 10M four-FE stress case, so do not advertise a universal 10M generic runtime from the local build. The 10M complex interaction/event-study two-way path did pass locally (~19.60 s).


## v0.9.0a1 PPML external certification

The integrated package includes `validation/ppml/`. Run its deterministic fixture
through licensed Stata `ppmlhdfe`, then compare coefficients, standard errors, sample
selection, absorbed DoF and separation diagnostics. Separately execute the upstream
17-dataset nonexistence/separation corpus and require observation-level separation-mask
parity. These are certification gates; local tests do not substitute for them.

Performance certification should also include 1M/10M two-way PPML, irreducible
three-way gravity, and hierarchical FE specifications on the same host as Stata. Report
replica and optimized Python paths separately so speedups from shared structural
optimization are not confused with Stata-vs-Python comparisons.


## 0.2.0 IV-PPML external certification

`validation/ivppml/` contains two separate gates. The feature-golden gate covers robust/two-way-cluster covariance, standardization, fweight/pweight, exposure, multiple endogenous regressors/over-identification and `separation(all)`. The official-corpus gate consumes the upstream Class A/B/C `.dta` files and licensed-Stata outputs.

A missing Stata golden is a hard **UNVERIFIED** state, not a pass. For Class C, raw 3+FE `df_a` may be reported separately because Stata/reghdfe and econhdfe can use different DoF conventions; coefficients, sample, covariance and separation diagnostics remain parity gates under the selected compatibility convention.

SPJ/bootstrap validation should additionally record successful/failed replicate counts, random seed, bootstrap unit (individual for A; directed pair for B/C), SPJ point estimate, bootstrap SD, percentile interval and CI-implied SE. Large-B parallel runs require target-hardware memory/thread-budget validation.

## 2026-09-12 real-world benchmark record

The repository archives the supplied local-machine real-data benchmark and diagnosis at
`benchmarks/real_world/2026-09-12/`. Preserve these files as historical external evidence.
For PPML, the archived run is explicitly a pre-v0.4.5 baseline because the diagnosis
found that direct-API `ExecutionConfig` thread and memory settings were not propagated
into the internal weighted FE projector. A release claiming post-fix PPML performance
must add a new same-machine rerun, including the actual projector resource diagnostics,
full default separation timings, peak RSS, and a separate FE-only sensitivity run.
