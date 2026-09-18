# econhdfe privacy-minimized error / mismatch report

> This is a **parameter-only** error report. Its default purpose is to let the developer diagnose implementation/configuration problems without receiving data or reconstructing the user's dataset. Do **not** include raw or synthetic observations, row-level values, file paths, filenames, actual variable/entity identifiers, exact commands/scripts, raw logs, stack locals, or a full traceback. Use anonymous aliases such as `y`, `x1`, `fe1`, `cluster1`, and report counts/configuration instead.

## 1. Report identity

- econhdfe version:
- econhdfe build/source identifier (wheel hash, source hash, or git commit if available):
- Issue type: crash / wrong result / reference-package mismatch / convergence / numerical instability / performance regression / API or documentation / other
- Severity: blocks analysis / material result risk / workaround available / cosmetic
- First version known to be affected:
- Last version known to work, if any:

## 2. Privacy checklist

Confirm the returned report contains **none** of the following:

- Raw, sampled, transformed, reconstructed, or synthetic observations: yes / no
- Row-level values or descriptive records that could reconstruct observations: yes / no
- Actual variable names, firm/person/place/product IDs, labels, or other domain identifiers: yes / no
- Local/cloud file paths, filenames, account/workspace names, credentials, license information, or URLs containing identifiers: yes / no
- Exact user commands, replication scripts, notebooks, or source-data code: yes / no
- Raw logs, full traceback text, stack locals, or memory dumps: yes / no

All six answers should be `no`. If any answer would be `yes`, redact or aggregate the information before returning this report.

Recommended aliases: outcome=`y`; regressors=`x1...xK`; endogenous variables=`endog1...`; instruments=`z1...`; absorbed effects=`fe1...`; cluster dimensions=`cluster1...`.

## 3. Runtime/environment parameters

Report only parameters relevant to numerical behavior. Omit machine/user identifiers.

| Parameter | Value |
|---|---|
| Python version | |
| econhdfe version | |
| NumPy / SciPy versions | |
| BLAS/LAPACK family | |
| Numba version / enabled | |
| OS family (optional; no hostname/user name) | |
| CPU architecture (optional; no device identifier) | |
| econhdfe threads | |
| econhdfe memory budget | |
| Reference package/version, if used | |

## 4. Anonymous specification fingerprint

Do not report real variable names or filter values. Describe only roles, counts, dimensions, and symbolic structure.

- Estimator: `olshdfe` / `ivhdfe` / `ppmlhdfe` / `ivppmlhdfe` / session / other
- Outcome count: usually 1
- Explicit regressor count before omission:
- Active regressor count after omission:
- Endogenous-variable count:
- Excluded-instrument count:
- Intercept included: yes / no / implicit
- Factor-variable structure using aliases only, if relevant (example: `i(fe1)##c(x3)`):
- Absorbed FE dimensions:
- FE level counts by anonymous dimension (`fe1`, `fe2`, ...):
- Heterogeneous-slope count by FE dimension, if any:
- Cluster dimensions:
- Cluster counts by anonymous dimension (`cluster1`, ...):
- Weight type: none / aweight / fweight / pweight / other
- Missing-data/sample rule category: listwise / explicit mask / other (do not report filter values)
- Singleton policy:
- Omitted/reference-category policy (type only; do not report labels):
- DoF method:
- VCE/inference method:
- Solver/backend:
- Solver tolerance / maximum iterations:
- Acceleration setting:
- Execution threads / memory budget:
- PPML separation methods, if applicable:
- Other non-identifying numerical/configuration options:

## 5. Sample and dimensional counts

Counts are sufficient; do not provide row hashes, example rows, quantiles tied to identifiers, or reconstructable summaries.

- Rows presented to estimator:
- Final estimation N:
- Reference-package final N, if applicable:
- Number of dropped missing observations:
- Number of singleton observations dropped:
- Number of PPML separated observations, if applicable:
- Number of omitted/collinear explicit regressors:
- Explicit design rank:
- Absorbed DoF:
- Inference residual DoF:
- Fit-statistics residual DoF, if relevant:

## 6. Error / warning fingerprint

For `EconHDFEError`, return structured parameters only. `details` may contain variable names or identifiers; include only detail **keys** and non-identifying scalar/count values.

- Exception class:
- `code`:
- `stage`:
- `details` keys:
- Non-identifying scalar/count details:
- `suggestion` category or sanitized text:
- Warning code(s), if any:

For a non-`EconHDFEError`, report only:

- Python exception type:
- Sanitized final error message with paths/identifiers/variable names removed:
- Stack signature as `module:function` names only (maximum 5 frames; no paths, line contents, locals, or full traceback):

## 7. Behavioral stability parameters

- Failure deterministic: yes / no / unknown
- Approximate occurrence rate, if nondeterministic:
- Random seed fixed: yes / no / not applicable (do not report secret/random-state contents)
- Changing thread count changes behavior: yes / no / not tested
- Changing documented solver/backend changes behavior: yes / no / not tested
- Same behavior on a smaller N chosen by the user: yes / no / not tested (do not return the subset)
- Prior econhdfe version behaves differently: yes / no / unknown

## 8. Reference-package parity summary, if applicable

Do not return exact commands or real variable names. Report whether the compared definitions and counts match, then return discrepancy magnitudes. Exact coefficient values are optional and should normally be omitted.

- Reference implementation/version:
- Same final N: yes / no / unknown
- Same number of active regressors: yes / no / unknown
- Same FE dimensions/level counts: yes / no / unknown
- Same cluster dimensions/counts: yes / no / unknown
- Same weight semantics: yes / no / unknown
- Same DoF convention: yes / no / unknown
- Same singleton/separation policy: yes / no / unknown / not applicable
- Same omitted/reference policy: yes / no / unknown

| Quantity class | Max abs difference | Max relative difference | Same definition? | Status |
|---|---:|---:|---|---|
| Coefficients | | | yes / no | |
| Standard errors | | | yes / no | |
| R2 / adjusted / within R2 | | | yes / no | |
| RSS / TSS / RMSE | | | yes / no | |
| F / Wald / p-value | | | yes / no | |
| Log likelihood / deviance / pseudo-R2 | | | yes / no | |
| Other diagnostic | | | yes / no | |

If one scalar is essential to diagnose the issue, report a rounded/normalized value only when it is non-identifying. Otherwise report the difference magnitude without the underlying estimate.

## 9. Numerical/execution diagnostics, if relevant

Only return aggregate counters/timings that do not expose data values.

- HDFE iterations:
- PPML IRLS iterations:
- Separation iterations by method:
- Solver route selected:
- FE canonicalization/core-reduction status:
- Fit time (seconds), if relevant:
- Separation time (seconds), if relevant:
- Peak RSS, if available:
- Converged: yes / no
- Termination/error reason code:

## 10. Developer-facing summary

- One-sentence issue summary using anonymous roles only:
- Correctness risk: none known / local / material / unknown
- Reference parity status: pass / fail / convention-sensitive / not tested
- Suspected layer: frontend / factorvars / HDFE / OLS / IV / PPML / IV-PPML / inference / reporting / runtime / unknown
- Regression from prior version: yes / no / unknown
- Most informative parameter mismatch/counter:
- Recommended next action using **additional metadata only**:

### Privacy rule

The standard econhdfe error-report workflow is parameter-only. It must **not request** raw data, sample rows, reconstructed/synthetic data, replication scripts, exact user commands, raw logs, full tracebacks, or file attachments. If the metadata is insufficient, ask for narrower additional **parameters/counters** first. Performance benchmarking is a separate workflow and still requires explicit user agreement through the benchmark-report policy.
