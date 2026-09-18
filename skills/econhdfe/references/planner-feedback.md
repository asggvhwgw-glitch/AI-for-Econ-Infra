# Planner calibration and developer feedback

Use this guide when `threads="auto"`, memory/representation planning, or execution performance looks suspicious on a real machine.

## Automatic thread calibration

`threads="auto"` is calibrated against the actual runtime instead of assuming every available logical CPU should be used. On the first sufficiently large HDFE workload (currently 100k+ observations), econhdfe executes a short memory-bound Numba probe over deterministic package-generated arrays at candidate thread counts. It selects the **smallest** candidate whose median time is within 5% of the measured best. This favors the memory-bandwidth saturation point rather than maximum thread count.

The calibration:

- never uses user observations or model variables;
- skips calibration for small HDFE workloads and uses one thread there to avoid paying calibration/parallel overhead;
- does not alter samples, FE, estimators, inference or solver mathematics;
- is cached by an anonymous runtime fingerprint so later processes can reuse it;
- is invalidated automatically when execution-relevant Python/NumPy/Numba/CPU-cap properties change;
- falls back to the historical effective-CPU cap if calibration fails;
- can always be overridden with an explicit positive integer `ExecutionConfig(threads=...)`.

The persistent calibration cache contains only anonymous runtime hashes, candidate thread counts and aggregate timings. It contains no hostname, paths to user data, variables, commands, or observations.

## Developer feedback

Use `references/planner-report-template.md` for a performance/planner report. The package helper can generate the same privacy-minimized structure:

```python
from econhdfe.planner import write_planner_developer_report

write_planner_developer_report(
    "econhdfe-planner-report.md",
    result=result,  # optional; only anonymous counts/aggregate execution metadata are extracted
    suspected_issue="auto threads slower than explicit threads",
)
```

An environment/calibration-only report can be generated without model data:

```bash
python skills/econhdfe/scripts/planner_report.py --output econhdfe-planner-report.md
```

Use `--force-calibration` only when deliberately re-measuring the machine after a runtime/hardware change or when diagnosing a stale calibration. Do not attach calibration arrays; they are temporary package-generated data.

## What to return

Return only aggregate resource/planner/timing metadata. If explicit thread counts are compared, preserve the exact same econometric specification and report timing ratios plus whether statistical results are unchanged. Do not change FE, controls, instruments, clusters, weights or sample filters to make a planner benchmark look better.

Real-data benchmarking remains separate and requires explicit consent under `references/benchmarking.md`.
