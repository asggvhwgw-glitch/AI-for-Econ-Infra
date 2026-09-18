# econhdfe planner / performance developer report

> Privacy-minimized, parameter-only report. It contains no raw observations, variable names, file paths, exact commands, hostnames, raw logs, or full tracebacks.

## 1. Report identity

- Report schema: 1
- Report ID: `34071f2e0b5025bf`
- econhdfe version: 0.4.10.1

## 2. Runtime resource summary

| Item | Value |
|---|---:|
| OS family / architecture | Linux / x86_64 |
| Python | 3.13.5 |
| NumPy | 2.3.5 |
| Numba | 0.65.1 |
| Effective CPU threads | 4 |
| CPU quota threads | 4 |
| Affinity threads | 5 |
| Runtime memory limit bytes | 4294967296 |
| Runtime memory current bytes | 1996795904 |

## 3. Automatic thread calibration

- Calibration ID: `caed077350a323ea`
- Calibration source: measured
- Maximum eligible threads: 4
- Selected automatic threads: **4**
- Near-best rule: smallest candidate within 5.0% of best median time
- Synthetic working-set bytes: 100663296
- Passes / repetitions: 4 / 3

| Threads | Median seconds | Relative to best |
|---:|---:|---:|
| 1 | 0.059966 | 2.635x |
| 2 | 0.040767 | 1.791x |
| 3 | 0.026560 | 1.167x |
| 4 | 0.022759 | 1.000x |

## 4. Anonymous run summary

- No estimator result was supplied; this is an environment/calibration-only report.

## 5. Execution plan

- No explicit `ExecutionPlan` was supplied.

## 6. Developer-facing assessment

- Suspected planner issue: not specified
- User-observed performance concern: not specified
- Recommended comparison: rerun the same econometric specification with explicit thread counts around the calibrated choice; do not change regressors, FE, instruments, weights, clustering, or sample solely for performance testing.

## Privacy rule

This report is parameter-only. Include no raw or synthetic observations. Do **not** add raw/sampled/synthetic observations, variable identifiers, confidential paths, exact commands/scripts, hostnames, credentials, raw logs, full tracebacks, or source data. If more evidence is needed, return aggregate timings/counters or use the separately authorized real-machine benchmark workflow.
