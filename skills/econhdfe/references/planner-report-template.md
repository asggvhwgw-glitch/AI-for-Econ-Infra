# econhdfe planner / performance developer report

> Return this completed Markdown file to the econhdfe developer when `threads="auto"`, memory/representation planning, or execution performance looks wrong. This is a **parameter-only** report: include no raw or synthetic observations, variable identifiers, confidential paths, exact commands/scripts, hostnames, credentials, raw logs, or full tracebacks.

## 1. Report identity

- Report ID:
- econhdfe version/build identifier:
- Planner-report schema:
- Automatic thread-calibration ID:
- Calibration source: measured / cache / fallback / single_thread

## 2. Runtime resource summary

Report only aggregate execution capabilities. Do not return hostnames, usernames, machine serials, or raw environment dumps.

| Item | Value |
|---|---:|
| OS family / architecture | |
| Python | |
| NumPy | |
| Numba | |
| Effective CPU threads | |
| CPU quota threads | |
| Affinity threads | |
| Runtime memory limit bytes | |
| Runtime memory current bytes | |

## 3. Automatic thread calibration

The standard calibration uses deterministic package-generated data only; do **not** attach or return those synthetic arrays.

- Maximum eligible threads:
- Selected automatic threads:
- Selection rule: smallest measured candidate within 5% of the best median calibration time
- Synthetic working-set bytes:
- Passes / repetitions:

| Threads | Median seconds | Relative to best |
|---:|---:|---:|
| | | |

## 4. Anonymous workload summary

Use only counts and anonymous roles. Do not include exact variable names, formulas, commands, filters, or data values.

- Estimator: OLS / linear IV / PPML / IV-PPML / other
- N after estimator sample construction:
- Number of active regressors:
- FE dimensions:
- FE level counts (optional aggregate tuple):
- Cluster dimensions/counts:
- Heterogeneous/block design: yes / no
- Approximate logical design width:
- Approximate stored/physical design width or payload, if available:
- Repeated-specification count / reuse count, if relevant:

## 5. Planner decision

- Requested threads: auto / integer
- Selected inner threads:
- Selected outer workers:
- BLAS thread budget:
- Parallel reason:
- Representation selected: dense / block / other
- Representation reason:
- Memory pressure: low / normal / high / critical / unknown
- Effective memory budget bytes:
- Estimated peak bytes:
- Cache/reuse route, if relevant:

## 6. Aggregate observed performance

Only report aggregate timings/counters; performance benchmarking on user data remains a separate explicit-consent workflow.

- Fit time (seconds):
- HDFE/projector time (seconds), if available:
- HDFE iterations:
- PPML/IV-PPML outer iterations, if applicable:
- Peak RSS bytes, if available:
- Cold or warm process:
- JIT included: yes / no / unknown
- Calibration cache hit: yes / no

If testing explicit thread counts on the **same unchanged econometric specification**, report only aggregate timing ratios:

| Threads | Time (s) | Relative to `auto` | Same statistical result? |
|---:|---:|---:|---|
| auto | | 1.000x | yes |
| | | | |

## 7. Behavioral/performance stability

- Repeated run shows the same selected automatic thread count: yes / no / not tested
- Explicit thread count materially faster than auto: yes / no / not tested
- Planner chooses a different route after restart: yes / no / not tested
- Same behavior on another machine/runtime: yes / no / not tested
- Prior econhdfe version materially faster: yes / no / unknown

## 8. Developer-facing summary

- One-sentence performance/planner issue:
- Suspected layer: calibration / parallel planner / memory planner / representation planner / HDFE / estimator / inference / data / unknown
- Correctness mismatch observed: yes / no
- Largest performance concern (ratio or aggregate time only):
- Most informative planner counter/mismatch:
- Recommended next action using **additional aggregate metadata only**:

## Privacy rule

This report is parameter-only. It must **not request or contain** raw/sampled/synthetic observations, variable identifiers, exact formulas, confidential paths, exact commands/scripts, hostnames, credentials, raw logs, full tracebacks, or source data. The automatic calibration itself uses package-generated synthetic arrays but never writes or returns those arrays. If more evidence is needed, ask for narrower aggregate counters/timings first. Real-data performance benchmarking is a separate explicit-consent workflow governed by the benchmark-report policy.
