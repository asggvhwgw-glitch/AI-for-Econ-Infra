# econhdfe real-machine benchmark report

> Return this completed Markdown file to the econhdfe developer. Do **not** include raw observations, confidential file paths, credentials, or other identifying data.

## 1. Run identity and consent

- Report date/time:
- econhdfe version:
- econhdfe build/source identifier (wheel hash, source hash, or git commit if available):
- Reference package(s) and exact version(s):
- User explicitly authorized benchmarking on this data: yes / no
- Source data/scripts treated as read-only: yes / no
- Benchmark operator/agent:
- Notes on confidentiality/redaction:

## 2. Machine and software environment

| Item | Value |
|---|---|
| OS / version | |
| CPU model | |
| Physical cores / logical threads | |
| RAM | |
| Storage type (if relevant) | |
| Python | |
| NumPy / SciPy | |
| BLAS/LAPACK implementation | |
| Numba | |
| Stata / R / Julia version | |
| reghdfe / ivreghdfe / ppmlhdfe / other reference version | |
| econhdfe threads | |
| Reference threads | |
| econhdfe memory budget | |
| Other relevant environment variables | |

## 3. Dataset metadata

Use non-identifying labels where necessary.

| Dataset ID | Rows before filters | Columns | On-disk size | In-memory/peak size if known | Format | Notes |
|---|---:|---:|---:|---:|---|---|
| | | | | | | |

Data preparation performed before the timed estimator call:

- 

Data loading included in timing: yes / no

## 4. Timing protocol

- Timing unit: fit-only / full command / wall-clock workflow
- Cold-start timing reported: yes / no
- Warm-process timing reported: yes / no
- JIT compilation included: yes / no
- Number of repetitions per specification:
- Reported summary: median / minimum / mean / single run
- Garbage collection / process restart policy:
- Cache policy:
- Thread pinning policy:
- Peak RSS measurement method, if any:

## 5. Specification registry

Give every compared specification a stable ID.

| Spec ID | Estimator | Outcome | Regressors / controls | Endogenous | Excluded IVs | Absorbed FE | Cluster/VCE | Weights | Sample/filter | DoF method | Other options |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S01 | | | | | | | | | | | |

For event studies, state the exact omitted/reference category. For PPML, state the separation methods. For IV, state the estimator (`2sls`, `liml`, `gmm2s`, etc.).

## 6. Parity summary

### 6.1 Discrete/sample parity

| Spec ID | Reference N | econhdfe N | Sample/mask match | Singletons dropped match | Separated obs match | Omitted variables match | Cluster counts match | Status |
|---|---:|---:|---|---|---|---|---|---|
| S01 | | | | | | | | pass / fail / n.a. |

### 6.2 Coefficient and inference parity

| Spec ID | Max |coef abs diff| | Max |coef rel diff| | Max |SE abs diff| | Max |SE rel diff| | Inference df reference | Inference df econhdfe | Absorbed DoF reference | Absorbed DoF econhdfe | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S01 | | | | | | | | | pass / fail |

### 6.3 Major fit-statistic parity

Fill only quantities defined comparably by both implementations. Do not force an IV overall-R2 comparison when the reference reports only partialled-out/within R2.

| Spec ID | Statistic | Reference | econhdfe | Abs diff | Rel diff | Same definition? | Status |
|---|---|---:|---:|---:|---:|---|---|
| S01 | R2 | | | | | yes / no | |
| S01 | Adjusted R2 | | | | | yes / no | |
| S01 | Within R2 | | | | | yes / no | |
| S01 | Adjusted within R2 | | | | | yes / no | |
| S01 | RSS | | | | | yes / no | |
| S01 | TSS / within TSS | | | | | yes / no | |
| S01 | RMSE | | | | | yes / no | |
| S01 | Model F / Wald chi2 | | | | | yes / no | |
| S01 | Prob > F / Prob > chi2 | | | | | yes / no | |
| S01 | Log likelihood | | | | | yes / no | |
| S01 | Null log likelihood | | | | | yes / no | |
| S01 | Deviance (PPML) | | | | | yes / no | |
| S01 | Pseudo R2 (PPML) | | | | | yes / no | |

### 6.4 IV/identification diagnostics, if applicable

| Spec ID | Diagnostic | Reference | econhdfe | Difference / interpretation | Status |
|---|---|---:|---:|---|---|
| | First-stage F / SW / KP / CD / over-ID as applicable | | | | |

Do not compare linear-IV KP/SW/Stock-Yogo diagnostics to IV-PPML unless the statistic is explicitly defined for that model.

## 7. Performance summary

| Spec ID | Reference engine | Reference time (s) | econhdfe engine/config | econhdfe time (s) | Speedup | Reference peak RSS | econhdfe peak RSS | Cold/warm | Parity status | Performance status |
|---|---|---:|---|---:|---:|---:|---:|---|---|---|
| S01 | | | | | | | | | | |

Do not combine non-comparable fits into one speedup number. If the reference executed extra specifications or data preparation, separate those costs.

## 8. Estimator-specific execution diagnostics

### HDFE / linear models

| Spec ID | Solver route | HDFE iterations | Canonicalized FE topology | Cache/session used | Notes |
|---|---|---:|---|---|---|
| | | | | | |

### PPML

| Spec ID | Separation methods | FE sep time | Simplex sep time | ReLU/IR sep time | Total sep time | IRLS iterations | Absorb subiterations | Separated N | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| | | | | | | | | | |

## 9. Mismatches, warnings, and failures

Record every material mismatch or failed specification, even when the overall benchmark looks favorable.

### Issue 1

- Spec ID:
- Quantity/phase:
- Reference result/error:
- econhdfe result/error:
- Reproducible: yes / no
- Suspected cause:
- Minimal reproduction available without raw data: yes / no
- Relevant log/artifact:

## 10. Exact commands / pseudo-code

Reference command(s):

```text
<command here>
```

econhdfe call/config:

```python
# call here; redact private paths/data identifiers as needed
```

## 11. Attached benchmark artifacts

List filenames and hashes if available. Do not attach raw confidential data unless separately authorized.

| File | Purpose | SHA-256 | Contains raw data? |
|---|---|---|---|
| | | | no |

## 12. Developer-facing summary

- Number of comparable specifications:
- Parity pass / fail / convention-sensitive counts:
- Median or aggregate speedup under comparable timing:
- Largest observed speedup:
- Largest regression/performance concern:
- Any correctness blocker:
- Recommended follow-up:
