from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import ThreadCalibration, get_thread_calibration
from .contracts import ExecutionPlan
from .resources import affinity_cpus, cpu_quota, runtime_resources


@dataclass(frozen=True, slots=True)
class PlannerDeveloperReport:
    """Privacy-minimized planner/performance feedback payload.

    This object intentionally contains no raw observations, variable names,
    file paths, commands, hostnames, full environment dumps, or tracebacks.
    """

    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.payload, indent=indent, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        p = self.payload
        runtime = p["runtime"]
        calib = p["thread_calibration"]
        run = p.get("run") or {}
        plan = p.get("execution_plan")
        lines = [
            "# econhdfe planner / performance developer report",
            "",
            "> Privacy-minimized, parameter-only report. It contains no raw observations, variable names, file paths, exact commands, hostnames, raw logs, or full tracebacks.",
            "",
            "## 1. Report identity",
            "",
            f"- Report schema: {p['schema_version']}",
            f"- Report ID: `{p['report_id']}`",
            f"- econhdfe version: {p.get('econhdfe_version', 'unknown')}",
            "",
            "## 2. Runtime resource summary",
            "",
            "| Item | Value |",
            "|---|---:|",
            f"| OS family / architecture | {runtime['system']} / {runtime['machine']} |",
            f"| Python | {runtime['python']} |",
            f"| NumPy | {runtime['numpy']} |",
            f"| Numba | {runtime['numba']} |",
            f"| Effective CPU threads | {runtime['effective_cpu_threads']} |",
            f"| CPU quota threads | {runtime['cpu_quota_threads']} |",
            f"| Affinity threads | {runtime['affinity_threads']} |",
            f"| Runtime memory limit bytes | {runtime['memory_limit_bytes']} |",
            f"| Runtime memory current bytes | {runtime['memory_current_bytes']} |",
            "",
            "## 3. Automatic thread calibration",
            "",
            f"- Calibration ID: `{calib['calibration_id']}`",
            f"- Calibration source: {calib['source']}",
            f"- Maximum eligible threads: {calib['max_threads']}",
            f"- Selected automatic threads: **{calib['selected_threads']}**",
            f"- Near-best rule: smallest candidate within {100*(calib['near_best_fraction']-1):.1f}% of best median time",
            f"- Synthetic working-set bytes: {calib['target_bytes']}",
            f"- Passes / repetitions: {calib['passes']} / {calib['repeats']}",
            "",
            "| Threads | Median seconds | Relative to best |",
            "|---:|---:|---:|",
        ]
        medians = calib["median_seconds"]
        best = min((x for x in medians if x > 0), default=0.0)
        for t, sec in zip(calib["candidates"], medians):
            rel = "n.a." if best <= 0 or sec <= 0 else f"{sec / best:.3f}x"
            lines.append(f"| {t} | {sec:.6f} | {rel} |")

        lines += ["", "## 4. Anonymous run summary", ""]
        if not run:
            lines.append("- No estimator result was supplied; this is an environment/calibration-only report.")
        else:
            for key in (
                "estimator", "nobs", "rank", "df_absorbed", "iterations", "converged",
                "vce", "fe_dimensions", "cluster_dimensions", "actual_absorb_threads",
                "projection_backend", "solver_route", "profile_total_seconds",
            ):
                if key in run and run[key] is not None:
                    lines.append(f"- {key}: {run[key]}")

        lines += ["", "## 5. Execution plan", ""]
        if plan is None:
            lines.append("- No explicit `ExecutionPlan` was supplied.")
        else:
            parallel = plan.get("parallel", {})
            memory = plan.get("memory", {})
            rep = plan.get("representation")
            lines.extend([
                f"- Parallel mode: {parallel.get('mode')}",
                f"- Outer workers / inner threads / BLAS threads: {parallel.get('outer_workers')} / {parallel.get('inner_threads')} / {parallel.get('blas_threads')}",
                f"- Parallel reason: {parallel.get('reason')}",
                f"- Memory pressure / feasible: {memory.get('pressure')} / {memory.get('feasible')}",
                f"- Effective memory budget bytes: {memory.get('effective_budget_bytes')}",
            ])
            if rep is not None:
                lines.append(f"- Representation: {rep.get('representation')} ({rep.get('reason')})")

        lines += [
            "",
            "## 6. Developer-facing assessment",
            "",
            f"- Suspected planner issue: {p.get('suspected_issue', 'not specified')}",
            f"- User-observed performance concern: {p.get('performance_concern', 'not specified')}",
            "- Recommended comparison: rerun the same econometric specification with explicit thread counts around the calibrated choice; do not change regressors, FE, instruments, weights, clustering, or sample solely for performance testing.",
            "",
            "## Privacy rule",
            "",
            "This report is parameter-only. Include no raw or synthetic observations. Do **not** add raw/sampled/synthetic observations, variable identifiers, confidential paths, exact commands/scripts, hostnames, credentials, raw logs, full tracebacks, or source data. If more evidence is needed, return aggregate timings/counters or use the separately authorized real-machine benchmark workflow.",
            "",
        ]
        return "\n".join(lines)


def _package_version() -> str:
    try:
        from importlib.metadata import version
        return version("econhdfe")
    except Exception:
        try:
            import econhdfe
            return str(getattr(econhdfe, "__version__", "unknown"))
        except Exception:
            return "unknown"


def _runtime_payload() -> dict[str, Any]:
    r = runtime_resources()
    try:
        import numba
        numba_version = numba.__version__
    except Exception:
        numba_version = "unknown"
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "numpy": np.__version__,
        "numba": numba_version,
        "effective_cpu_threads": int(r.cpu_threads),
        "cpu_quota_threads": cpu_quota(),
        "affinity_threads": affinity_cpus(),
        "memory_limit_bytes": r.memory_limit_bytes,
        "memory_current_bytes": r.memory_current_bytes,
    }


def _safe_result_summary(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    absorb = getattr(result, "absorb_info", None) or {}
    clusters = getattr(result, "cluster_counts", None)
    profile = getattr(result, "profile", None) or {}
    solver = absorb.get("solver_selection") if isinstance(absorb, dict) else None
    if isinstance(solver, dict):
        solver = solver.get("selected") or solver.get("resolved") or solver.get("requested")
    fe_names = getattr(result, "fe_names", ()) or ()
    return {
        "estimator": str(getattr(result, "estimator", type(result).__name__)),
        "nobs": _safe_int(getattr(result, "nobs", None)),
        "rank": _safe_int(getattr(result, "rank", None)),
        "df_absorbed": _safe_int(getattr(result, "df_absorbed", None)),
        "iterations": _safe_int(getattr(result, "iterations", None)),
        "converged": _safe_bool(getattr(result, "converged", None)),
        "vce": _safe_scalar(getattr(result, "vce", None)),
        "fe_dimensions": len(fe_names),
        "cluster_dimensions": 0 if clusters is None else len(clusters),
        "actual_absorb_threads": _safe_int(absorb.get("absorb_threads") if isinstance(absorb, dict) else None),
        "projection_backend": _safe_scalar(absorb.get("projection_backend") if isinstance(absorb, dict) else None),
        "solver_route": _safe_scalar(solver),
        "profile_total_seconds": _safe_float(profile.get("total_seconds") if isinstance(profile, dict) else None),
    }


def _safe_int(value):
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value):
    return None if value is None else bool(value)


def _safe_scalar(value):
    return value if isinstance(value, (str, int, float, bool, type(None))) else None


def build_planner_developer_report(
    *,
    result: Any = None,
    execution_plan: ExecutionPlan | None = None,
    calibration: ThreadCalibration | None = None,
    suspected_issue: str | None = None,
    performance_concern: str | None = None,
) -> PlannerDeveloperReport:
    calibration = get_thread_calibration() if calibration is None else calibration
    payload = {
        "schema_version": 1,
        "econhdfe_version": _package_version(),
        "runtime": _runtime_payload(),
        "thread_calibration": calibration.as_dict(),
        "run": _safe_result_summary(result),
        "execution_plan": None if execution_plan is None else execution_plan.as_dict(),
        "suspected_issue": suspected_issue or "not specified",
        "performance_concern": performance_concern or "not specified",
    }
    identity = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    payload["report_id"] = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return PlannerDeveloperReport(payload)


def write_planner_developer_report(
    path: str | os.PathLike[str],
    *,
    format: str | None = None,
    **kwargs,
) -> Path:
    out = Path(path)
    fmt = (format or out.suffix.lstrip(".") or "md").lower()
    report = build_planner_developer_report(**kwargs)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt in {"md", "markdown"}:
        out.write_text(report.to_markdown(), encoding="utf-8")
    elif fmt == "json":
        out.write_text(report.to_json(), encoding="utf-8")
    else:
        raise ValueError("planner developer report format must be md/markdown/json")
    return out
