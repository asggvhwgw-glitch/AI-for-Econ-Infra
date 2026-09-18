# Real-data parity checks and benchmarking

Use this guide when validating econhdfe against an established implementation or when a user explicitly agrees to benchmark econhdfe on their data.

## Testing-release policy

econhdfe is currently a testing/beta package. For consequential empirical work, do not treat a successful fit as sufficient validation.

For each substantive run, batch, or regression table, randomly select at least one economically representative specification and cross-check it against an established reference implementation when that reference is available:

- OLS-HDFE: `reghdfe` or an equivalently mature implementation;
- linear IV-HDFE: `ivreghdfe`/`ivreg2` or an equivalently mature implementation;
- PPML-HDFE: `ppmlhdfe` or an equivalently mature implementation;
- features with no exact upstream analogue: compare the common subproblem and use independent numerical/theoretical checks for the extension.

At minimum compare the estimation sample/N, singleton/separation/omission decisions, coefficients, standard errors, inference DoF, absorbed DoF under the same DoF method, and the major fit statistics relevant to that estimator. For PPML also compare iteration count, separated-observation count, log pseudolikelihood, deviance, pseudo-R2 and model Wald chi-square when available. A mismatch must be surfaced; do not silently choose whichever output looks more plausible.

Do not claim exact parity when the two programs intentionally use different DoF methods, weighting semantics, separation policies, convergence tolerances, or IV fit-statistic conventions. Record those differences explicitly.

## User-data benchmark consent

A real-data benchmark may be run only after the user explicitly agrees to use their data for benchmarking. Treat the source data and source replication scripts as read-only unless the user separately asks for edits.

When benchmarking:

1. preserve the original econometric specification, sample filters, weights, FE, clusters, instruments and separation policy;
2. do not change a specification solely to make econhdfe faster;
3. run the reference and econhdfe on the same comparable estimation unit;
4. state whether timing is fit-only or wall-clock and whether JIT/startup/data loading is included;
5. separate cold and warm timings instead of mixing them;
6. pin or record thread counts and memory policy when possible;
7. record failures and mismatches, not only successful rows;
8. never embed raw observations, confidential paths, credentials or identifying data in the benchmark report;
9. write new benchmark artifacts rather than overwriting prior real-machine evidence.

Use `benchmark-report-template.md` as the canonical return format. An installed package can write an untouched copy with `econhdfe-report benchmark --output econhdfe-benchmark-report.md`; generating the template does not itself authorize a benchmark. The completed Markdown file should be self-contained enough for a developer to reproduce the interpretation of the benchmark without receiving the user's raw data.

## Recommended parity tolerance reporting

Do not use one universal pass threshold for all quantities. Report the observed absolute/relative difference and the tolerance used. Typical categories are:

- exact discrete parity: N, sample mask hash, omitted-variable identities, singleton/separation counts, cluster counts;
- tight numerical parity: coefficients, SEs, RSS/TSS, log likelihood/deviance;
- convention-sensitive parity: adjusted R2, RMSE, model F/Wald statistics, absorbed DoF;
- algorithm-sensitive diagnostics: iteration/subiteration counts and solver route.

A benchmark can pass performance while fail parity, or pass parity while fail performance. Keep those statuses separate.
## When benchmark evidence reveals a bug

If a benchmark exposes a crash, material numerical mismatch, convergence anomaly, or suspected correctness defect, keep the benchmark record intact and create a separate `error-report-template.md` report for the defect. Cross-link only a non-identifying benchmark/spec ID. The error report itself is parameter-only and must not request raw/synthetic data, exact commands/scripts, raw logs, full tracebacks, or attachments. Benchmark authorization remains separate and applies only to the benchmark workflow.

