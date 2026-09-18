# Unified execution planner calibration checkpoint

Machine-readable development artifacts are under `benchmarks/planner/`. `representation_operator_mix.json` compares dense and certified block operators across structural sparsity and BLAS thread counts; it shows that storage-byte savings materially reduce transpose/operator traffic but do not translate one-for-one into wall-time speedups. `planner_overhead.json` measures the pure composite planner at roughly 12 microseconds per call on the development container, which is negligible when planning occurs outside numerical hot loops. `heterogeneous_spec_planner_checkpoint.json` is a same-checkpoint regression/performance guard for the four structured model families. These are calibration artifacts, not release performance promises.

The resulting policy is intentionally conservative: exactness certificates are production-critical, while the generic byte-traffic score remains advisory until hardware/operator calibration is robust enough to replace the existing one-shot/repeated-pass representation rule.

## v0.4.5 PPML execution-fix development A/B

A fixed-4-thread same-container development A/B is stored in `benchmarks/ppml/v045_execution_fix_synthetic.json`. On the measured high-zero synthetic cases, the repaired candidate preserved coefficient/sample/separation/iteration parity and measured about 1.14x versus v0.4.4.4 at 1M/2FE and 1.07x at 300k/3FE. These numbers only guard against a local performance regression; they are not substituted for the externally supplied real-data benchmark.

# Real-world benchmark registry

A user-supplied real-machine run dated 2026-09-12 is archived under
`benchmarks/real_world/2026-09-12/`. The common-comparison trade workload reported
12 OLS fits at 7.79x aggregate speedup, 6 IV fits at 5.50x, and the combined 18-fit
comparison at 6.06x. The complex `firm + year + city#year` specifications reported
about 11.69x--12.07x versus `reghdfe` and 5.56x--6.49x versus the recorded
`reghdfejl` runs. These are hardware/workload-specific external measurements, not
universal performance guarantees.

The same run recorded six-year PPML speedups of 3.74x (`optimized`) and 2.24x
(`replica`), but its companion diagnosis identified a resource-policy propagation bug:
the direct PPML API validated `ExecutionConfig` without forwarding its requested thread
and memory policy into the weighted FE projector. Those PPML numbers are therefore
kept as a pre-v0.4.5 observed baseline and must be rerun before making a strict
configured-thread post-fix claim. The source records are immutable; follow-up runs are
added as new dated artifacts.

# HDFE solver integration benchmark — v0.4.4

The solver-opt2 integration was rechecked against v0.4.3 on the current container. These timings are environment-specific and are not hardware-independent claims. Machine-readable results are in `benchmarks/hdfe/solver_v044_integration.json`.

Direct absorber checks confirm the intended topology-dependent behavior: warmed random 3FE improved from a 0.214 s median to 0.124 s (~1.72x); a 200k leaf-rich 3FE case was ~8.69x faster; a weighted 1M leaf-rich case was ~2.91x faster; weighted 500k x 20 RHS was ~4.22x faster; the deliberately hard 10k cycle stayed near parity (~1.05x) and routed to `auto_cg`. A cold first random run can include one-time JIT/planner overhead, so warmed medians are recorded separately rather than hidden.

A 100k end-to-end complex-FE corpus with two repetitions per cell remained close to parity on irreducible mobility designs (roughly 0.98x–1.02x) and ranged from ~1.01x to ~1.16x on the tested stable interaction specifications. This is expected: the public estimator path retains its established direct-CG semantics, while the strongest gains occur in peelable cores and wide weighted multi-RHS projection.

<!-- econhdfe architecture note -->
> **Lineage note:** this document records validation/benchmark work from the pre-rename `pyreghdfe` lineage. The `econhdfe 0.2.0` package preserves the numerical implementations and regression tests but did not rerun every historical external/performance matrix. Treat the figures below as inherited evidence until the corresponding `econhdfe` benchmark is rerun.


## econhdfe 0.1.0a3 PPML engineering pass

These measurements were rerun on the current `econhdfe` namespace after the a3 PPML execution/memory changes. They are same-container development figures, not hardware-independent claims. The container exposes 5 CPU cores and a 4 GiB cgroup memory limit. Unless noted, separation was disabled to isolate estimation-engine performance; robust covariance is included in the completed-estimator timings.

| scenario | replica | optimized | speedup | numerical check |
|---|---:|---:|---:|---:|
| 1M, 2 FE, 4 regressors | 15.09 s | 4.52 s | **3.34x** | max coef diff `2.8e-16` |
| 144k true 3-FE gravity | 3.10 s | 2.44 s | **1.27x** | max coef diff `0` |
| 240k hierarchy, 4 requested FE -> 2 effective FE | 3.45 s | 1.48 s | **2.32x** | max coef diff `5.6e-17` |

The true-three-FE row is intentionally only a regression/performance check: a3 does not introduce a new 3+ FE solver. Its smaller speedup therefore comes from common PPML/HDFE engineering reuse rather than a changed multiway algorithm.

### Large-scale optimized PPML

- 5M observations, 2 regressors, two crossed FEs, robust VCE: **19.23 s**, 6 IRLS iterations, ~**1.36 GiB** process max RSS; coefficients `[0.11975, -0.07910]` for DGP `[0.12, -0.08]`.
- 10M observations, 2 regressors, two crossed FEs, robust VCE: **45.75 s**, 6 IRLS iterations, ~**2.50 GiB** process max RSS; coefficients `[0.12031, -0.08003]` for DGP `[0.12, -0.08]`.

A stage trace of the 10M run showed no nonlinear FE-solver stall after the a3 changes: the initial post-FE rank projection was ~6.2 s; IRLS weighted-FE projections were ~2.3--3.0 s each; strict WLS solves ~2.7 s; final robust VCE ~0.6 s.

### IRLS kernel stability

The previous fused parallel `exp + scalar deviance reduction` path showed rare long-tail stalls under repeated 10M calls. a3 separates the embarrassingly parallel elementwise exponential from a deterministic compiled deviance reduction. Across eight consecutive 10M calls the combined state update measured **0.166--0.182 s** (median ~0.178 s) with no stall.

### Streamed PPML covariance

On a 5M x 3 residualized synthetic block:

- robust: **0.44 s**;
- one-way cluster: **0.59 s**;
- two-way cluster: **3.71 s**.

The robust/cluster implementation no longer materializes a full `N x K` score matrix.

### Cross-specification warm start

On 300k observations with the same two FE structure, extending `[x1, x2]` to `[x1, x2, x3]` took 1.24 s cold versus 0.97 s from the previous converged predictor (**1.28x**), reducing IRLS iterations from 6 to 5 with maximum coefficient difference `2.8e-17`.

Machine-readable records are in `benchmarks/ppml/optimized_round_*.json`.
# Benchmarks — v0.8.0 (including retained v0.6 historical snapshots)

These are development-container measurements, not hardware-independent claims.
The CPU-only Linux container exposes 5 AMD EPYC cores and a 4 GiB process/cgroup
memory ceiling. Runtimes vary with memory bandwidth, FE connectivity, BLAS load
and virtualization.

## 10M-row multi-FE / two-way-cluster stress test

The main v0.6 stress test matches the intended large applied-economics workload:

- 10,000,000 observations;
- 12 continuous controls plus the dependent variable (13 RHS columns to absorb);
- four intercept FEs with 300k / 350k / 400k / 450k levels;
- two-way clustering on the first two high-dimensional FEs;
- symmetric MAP, tolerance `1e-8`;
- `projection_backend="indexed"`, 4 Numba absorption threads.

The DGP coefficient vector was
`[1, -.7, .5, .25, -.2, .15, .8, -1.2, .33, .1, -.45, .6]`.
The maximum absolute coefficient error was `5.25e-4`.

| RHS pool | Residualization | End-to-end estimation | MAP iterations | Max RSS |
|---:|---:|---:|---:|---:|
| auto / 6 (`6+6+1`) | 78.03 s | 100.85 s | 17 | 3.72 GiB |
| 7 (`7+6`) | 66.14 s | 87.67 s | 17 | 3.72 GiB |
| 13 (single block) | **58.37 s** | **79.78 s** | 17 | **3.72 GiB** |

The previous low-memory serial/fused branch measured about 160.87 s for FE
absorption and 177.35 s end-to-end on the same scale. The one-block v0.6 run is
therefore roughly 2.76x faster in residualization and 2.22x faster end-to-end.
The 13-column pool intentionally exceeds the default 512 MiB scratch budget;
it was tested because this particular 4 GiB container still had enough headroom.
Use `pool_size="auto"` for conservative memory behavior.

## Indexed projection CPU scaling

For one intercept FE with 450,000 levels and a `10,000,000 x 6` RHS block, after
the group index and Numba kernel are warm:

| Absorption threads | Projection time | Speedup vs 1 thread |
|---:|---:|---:|
| 1 | 0.656 s | 1.00x |
| 2 | 0.331 s | 1.98x |
| 4 | **0.177 s** | **3.70x** |
| 5 | 0.205 s | 3.20x |

The fifth core is already past the memory-bandwidth sweet spot on this machine,
so `absorb_threads=4` is faster than using all five cores. `"auto"` means use
the currently available Numba thread count; users seeking maximum throughput
should benchmark a few thread counts on their own hardware.

### Group-index construction

Dense FE codes are indexed with an O(N+G) counting sort rather than `argsort`.
For 10M observations / 450k levels the steady-state counting-sort build was
about 0.39 s in a microbenchmark (the first ever call also pays Numba JIT cost).
The resident index is about 41.6 MiB for that FE. Four indexes for the stress
test occupy about 164 MiB.

## Parallel convergence scan

The v0.6 MAP stopping metric no longer creates chunk-level NumPy temporaries.
For a `5,000,000 x 6` block:

| Method | Time per convergence scan |
|---|---:|
| previous chunked NumPy | 0.379 s |
| Numba, 1 thread | 0.066 s |
| Numba, 2 threads | 0.035 s |
| Numba, 5 threads | **0.032 s** |

The returned maximum relative update was identical in the benchmark.

## Group + individual FE regression check

The v0.5 incidence engine remains unchanged. A retained release check uses
400,000 group outcomes / 1,200,000 membership rows / 200,000 individuals,
1,000 time FE levels and four ordinary regressors:

| Workload | Result |
|---|---:|
| Solver / backend | LSMR / auto-selected CSR |
| Runtime | 9.261 s |
| LSMR iterations | 88 |
| Whole-process max RSS | 404.3 MiB |
| Estimated coefficients | `[1.00011, -0.50066, 0.25029, 2.00133]` |

## Reproduce

CPU projection scaling:

```bash
PYTHONPATH=. python benchmarks/bench_cpu_projection.py \
  --n 10000000 --levels 450000 --rhs 6 --threads 1 2 4 5
```

Full stress test (large RAM/CPU workload):

```bash
PYTHONPATH=. python benchmarks/bench_10m_multife.py \
  --n 10000000 --controls 12 --threads 4 --pool-size 13
```

Group/individual benchmark:

```bash
PYTHONPATH=. python benchmarks/bench_group_individual.py \
  --groups 400000 --individuals 200000 --incidence-backend auto
```

## 0.7 development: interaction-FE solver diagnostic

Scaled analogue of a common empirical specification with a redundant time FE inside an interaction FE:

- 500,000 observations
- 80,000 firm FE levels
- 16 year FE levels
- 347 cities × 16 years (`city#year`)
- 3 continuous controls
- 1% cross-city firm mobility
- 4 CPU absorption threads
- tolerance `1e-8`

| path | time | iterations | converged |
|---|---:|---:|---|
| plain symmetric MAP, redundant year retained | 10.81 s | 250 | no |
| symmetric MAP + CG, redundant year retained | 2.64 s | 34 | yes |
| symmetric MAP + CG + FE canonicalization | 1.89 s | 34 | yes |

The converged CG paths agree numerically; canonicalization removes `year` because it is exactly spanned by `city#year`.  The plain-MAP row is deliberately capped at 250 iterations and is a solver-diagnostic baseline, not a completed estimate.

Interaction encoding microbenchmark (2,000,000 rows, 400 cities × 16 years): mixed-radix encoding ~0.20 s versus pandas MultiIndex factorization ~0.37 s in the release container (~1.9× faster).  Treat these timings as hardware-specific.

A second structural stress test used 2,000,000 observations, 559,951 potential firm levels, 16 years, and 347×16 city-year cells. After singleton pruning, 1,944,278 observations remained. MAP+CG completed in ~7.14 s (23 iterations); `method="twoway"` completed in ~1.91 s (35 PCG iterations), about 3.7× faster, with coefficient vectors agreeing to floating-point precision. This is a scaled structural analogue, not the user's confidential dataset.


## 0.7.0.dev4 structural dependency DAG

`benchmarks/bench_dependency_dag.py` constructs a 2,000,000-row categorical
hierarchy `city -> province -> region -> country`. The planner certified only
three adjacent direct component mappings and inferred three broader refinement
relations transitively. On the release container the structural planning stage
took about 0.20 seconds. This benchmark does not materialize an N-by-K dummy
design; it isolates dependency-graph construction.

## 0.7.0.dev5 — 10M complex interaction-FE / event-study stress test

The dev5 stress DGP intentionally combines structures that commonly appear together in applied work:

- 10,000,000 observations; ~555,555 potential firm FE levels; 16 years; 347 cities, 31 provinces, 7 regions and 80 metros.
- Requested FEs: `firm`, `year`, `region#year`, `province#year`, `metro#year`, `city#year`. Exact hierarchy certification canonicalizes this to `firm + city#year`.
- Staggered event-study regressors with an explicit user-selected `event_time=-1` reference.
- Deliberate composite collinearity (`xsum=x1+x2`) with an explicit user omission.
- Large coarse factor/interaction control blocks that are structurally spanned by the finer absorbed FE; 1,904 columns were omitted before dense materialization.
- Two-way clustered covariance.

On the constrained release container the full estimation completed in about **21.06 s**, selected the specialized two-way solver, required 12 two-way iterations and peaked at about **2966 MiB RSS (~2.90 GiB)**. DGP construction took about 2.30 s and is not included in the estimator time. The maximum absolute error across identified simulated coefficients was about **2.96e-4**.

The test exposed and motivated three dev5 engineering changes: shared component-code caching for repeated interaction components, multi-RHS two-way PCG, and low-memory clustered-score aggregation. These figures are environment-specific and are not a cross-hardware performance guarantee.



## 0.8.0 final release performance check

Final local checks after the runtime/execution refactor:

- 10M complex hierarchical FE + event-study + two-way cluster: **19.60 s**, ~**2.82 GiB** peak RSS, 12 specialized two-way iterations, maximum identified-parameter error ~`2.96e-4`.
- 1M generic 4-FE / 12-control / two-way-cluster sanity: **5.93 s**, 16 iterations, ~**606 MiB** peak RSS.
- The generic 10M / 4-FE / 12-control sustained-memory run is intentionally not reported as a v0.8.0 pass on the 4 GiB cgroup. Repeated stress runs showed environment-sensitive memory-bound runtime at high resident working sets. This remains a target-hardware acceptance benchmark rather than a reason to add container-specific solver branches.

`absorb_info["execution_plan"]` should be recorded with future benchmarks so wall time is interpreted together with the actual pool width, threads and memory headroom selected on that host.


## 0.2.0 IV-PPML baseline

These are same-container development A/B timings, not Stata speed claims. Two-way IV-PPML naturally reuses the a3 weighted-HDFE topology/cache path: 100k observations measured about 1.45 s replica vs 0.25 s optimized (~5.76x); 1M measured 20.26 s vs 4.50 s (~4.51x), with coefficient/SE differences at numerical precision. A 144k true Class-C-style `pair + exporter#year + importer#year` case measured 2.82 s vs 2.74 s (~1.03x), confirming that irreducible 3+FE remains the shared bottleneck and was not patched in a4.

SPJ/bootstrap timings are workload-specific because each Class A/B/C point/replicate requires multiple IV-PPML refits. The bootstrap implementation precompiles cluster row groups once; do not benchmark the old per-cluster full-table-scan prototype.

## 0.4.0 exact multiway categorical-rank engine

Warm-path timings in the current release container (illustrative, not API guarantees):

| case | time | notes |
|---|---:|---|
| random 3FE, 5k rows / ~1.5k levels | 0.16 s | auto exact rank |
| random 3FE, 10k rows / ~3k levels | 0.44 s | auto exact rank |
| random 3FE, 20k rows / ~6k levels | 1.61 s | auto exact rank |
| leaf-rich 3FE, 1M rows | 0.63 s | all residual edges eliminated by exact peeling |

These timings include tuple deduplication and exact certification. They are slower than the dedicated exactdof2 prototype's recorded machine, but show the same scaling regime after integration. The hard case remains a large genuinely rank-deficient irreducible residual core; optional SymPy/FLINT backends or future sparse black-box exact algebra target that regime. Default `dof_method="pairwise"` is unchanged.

## v0.4.1 repeated-spec benchmark

Synthetic 300,000-observation OLS-HDFE table, cluster VCE by firm, default `cache_validation="signature"`, four paired rounds; see `benchmarks/repeated/repeated_spec_v041.json`.

| workload | independent median | session median | speedup |
|---|---:|---:|---:|
| three outcomes, same X/FE | 0.327 s | 0.238 s | 1.38x |
| four columns adding controls | 0.428 s | 0.222 s | 1.93x |
| three new FE combinations once each | 0.535 s | 0.470 s | 1.14x |
| three FE combinations toggled across eight columns | 1.136 s | 0.603 s | 1.88x |

All reported maximum coefficient differences were zero at the stored double-precision output. A separate 180k linear-IV clustered benchmark with three exogenous-control specifications showed ~1.25x median speedup; IV gains are smaller because first-stage and identification diagnostics are intentionally recomputed.
