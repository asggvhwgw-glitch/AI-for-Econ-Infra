from __future__ import annotations

import argparse
import math
import platform
import sys
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy

from .config import ExecutionConfig, HDFEConfig, InferenceConfig
from .errors import EconHDFEError


_TEMPLATE_PACKAGE = "econhdfe.templates"
_ISSUE_TYPES = {
    "crash", "wrong result", "reference-package mismatch", "convergence",
    "numerical instability", "performance regression", "API or documentation", "other",
}
_SEVERITIES = {"blocks analysis", "material result risk", "workaround available", "cosmetic"}
_SAFE_DETAIL_EXACT = {
    "iterations", "max_iter", "nobs", "nobs_raw", "rank", "rows", "cols",
    "dimensions", "n_dimensions", "degree", "max_degree", "min_degree",
    "df_resid", "df_model", "df_absorbed", "df_resid_fit",
    "dropped_singletons", "n_singletons", "n_separated",
}
_SAFE_DETAIL_SUFFIXES = (
    "_count", "_counts", "_iterations", "_rank", "_rows", "_cols",
    "_dimensions", "_dof", "_df",
)
_SAFE_RESULT_FIELDS = (
    "nobs", "nobs_raw", "rank", "df_resid", "df_absorbed", "converged",
    "iterations", "dropped_singletons", "weight_type", "vce", "cluster_counts",
    "df_model", "df_resid_fit", "vcov_rank", "n_separated", "n_singletons",
    "nobs_full",
)
_SAFE_PARITY_FIELDS = {
    "coefficient_max_abs_diff", "coefficient_max_rel_diff",
    "stderr_max_abs_diff", "stderr_max_rel_diff",
    "fit_stat_max_abs_diff", "fit_stat_max_rel_diff",
    "sample_n_difference", "absorbed_dof_difference", "inference_dof_difference",
}


def _template(name: str) -> str:
    return resources.files(_TEMPLATE_PACKAGE).joinpath(name).read_text(encoding="utf-8")


def error_report_template() -> str:
    """Return the canonical privacy-minimized error-report template."""
    return _template("error-report-template.md")


def benchmark_report_template() -> str:
    """Return the canonical real-machine benchmark template.

    Using user data for benchmarking still requires explicit user consent; this
    helper merely returns the blank report format.
    """
    return _template("benchmark-report-template.md")


def _safe_scalar(value: Any) -> int | float | bool | str | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _safe_numeric_details(details: Mapping[str, Any] | None) -> dict[str, int | float | bool]:
    """Keep only counter-like scalar details; never copy arbitrary strings."""
    out: dict[str, int | float | bool] = {}
    for key, value in (details or {}).items():
        key_l = str(key).lower()
        if key_l not in _SAFE_DETAIL_EXACT and not key_l.endswith(_SAFE_DETAIL_SUFFIXES):
            continue
        safe = _safe_scalar(value)
        if isinstance(safe, (bool, int, float)):
            out[str(key)] = safe
    return out


def _safe_result(result: Any | None) -> dict[str, Any]:
    if result is None:
        return {}
    out: dict[str, Any] = {}
    estimator = getattr(result, "estimator", None)
    if estimator is None:
        name = type(result).__name__.lower()
        estimator = "ivppml" if "ivppml" in name else "ppml" if "ppml" in name else None
    if estimator in {"ols", "iv", "2sls", "liml", "gmm", "gmm2s", "ppml", "ivppml"}:
        out["estimator"] = estimator
    for field in _SAFE_RESULT_FIELDS:
        if not hasattr(result, field):
            continue
        value = getattr(result, field)
        if field == "cluster_counts":
            try:
                out[field] = tuple(int(x) for x in value)
            except (TypeError, ValueError):
                pass
            continue
        if field in {"weight_type", "vce"}:
            if value is None:
                out[field] = None
            elif str(value) in {
                "none", "aweight", "fweight", "pweight", "robust", "cluster",
                "hc0", "hc1", "unadjusted",
            }:
                out[field] = str(value)
            continue
        safe = _safe_scalar(value)
        if safe is not None:
            out[field] = safe
    # Counts are useful and do not disclose variable identities.
    for source, target in (
        ("names", "explicit_parameter_count"),
        ("endog_names", "endogenous_count"),
        ("instrument_names", "excluded_instrument_count"),
        ("fe_names", "absorbed_fe_dimensions"),
    ):
        value = getattr(result, source, None)
        if value is not None:
            try:
                out[target] = len(value)
            except TypeError:
                pass
    return out


def _safe_configs(
    hdfe_config: HDFEConfig | None,
    execution_config: ExecutionConfig | None,
    inference_config: InferenceConfig | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if hdfe_config is not None:
        out["hdfe"] = {
            "solver": hdfe_config.solver,
            "tolerance": hdfe_config.tolerance,
            "max_iter": hdfe_config.max_iter,
            "dof_method": hdfe_config.dof_method,
            "canonicalize": hdfe_config.canonicalize,
        }
    if execution_config is not None:
        out["execution"] = {
            "threads": execution_config.threads,
            "memory_budget_mb": execution_config.memory_budget_mb,
            "profile": execution_config.profile,
            "cache": execution_config.cache,
            "cache_validation": execution_config.cache_validation,
        }
    if inference_config is not None:
        out["inference"] = {
            "vce": inference_config.vce,
            "confidence_level": inference_config.confidence_level,
            "diagnostics": inference_config.diagnostics,
        }
    return out


def _safe_error(error: BaseException | None) -> dict[str, Any]:
    if error is None:
        return {}
    if isinstance(error, EconHDFEError):
        return {
            "type": type(error).__name__,
            "code": error.code,
            "stage": error.stage,
            "numeric_details": _safe_numeric_details(error.details),
        }
    # Unknown errors expose only the exception class. Messages are deliberately
    # omitted because they commonly contain paths, variable names or values.
    return {"type": type(error).__name__, "code": None, "stage": None, "numeric_details": {}}


def _safe_stack_signature(error: BaseException | None, *, limit: int = 5) -> tuple[str, ...]:
    if error is None:
        return ()
    frames: list[str] = []
    tb = error.__traceback__
    while tb is not None:
        module = str(tb.tb_frame.f_globals.get("__name__", ""))
        if module == "econhdfe" or module.startswith("econhdfe."):
            frames.append(f"{module}:{tb.tb_frame.f_code.co_name}")
        tb = tb.tb_next
    return tuple(frames[-limit:])


def _safe_parity(parity: Mapping[str, Any] | None) -> dict[str, int | float]:
    out: dict[str, int | float] = {}
    for key, value in (parity or {}).items():
        if key not in _SAFE_PARITY_FIELDS:
            continue
        safe = _safe_scalar(value)
        if isinstance(safe, (int, float)) and not isinstance(safe, bool):
            out[key] = safe
    return out


def safe_error_payload(
    *,
    error: BaseException | None = None,
    result: Any | None = None,
    hdfe_config: HDFEConfig | None = None,
    execution_config: ExecutionConfig | None = None,
    inference_config: InferenceConfig | None = None,
    issue_type: str = "other",
    severity: str = "material result risk",
    parity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract a privacy-minimized, data-free report payload.

    The function never reads DataFrames, coefficient arrays, residuals, fitted
    values, variable names, file paths, error messages, raw logs, or traceback
    text.  Only a narrow allowlist of scalar/counter metadata is emitted.
    """
    if issue_type not in _ISSUE_TYPES:
        raise ValueError(f"unsupported issue_type: {issue_type!r}")
    if severity not in _SEVERITIES:
        raise ValueError(f"unsupported severity: {severity!r}")
    import econhdfe as _pkg

    return {
        "report_schema": "econhdfe.parameter_error.v1",
        "issue_type": issue_type,
        "severity": severity,
        "environment": {
            "python": platform.python_version(),
            "econhdfe": _pkg.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "os_family": platform.system() or None,
            "cpu_architecture": platform.machine() or None,
        },
        "result": _safe_result(result),
        "config": _safe_configs(hdfe_config, execution_config, inference_config),
        "error": _safe_error(error),
        "stack_signature": _safe_stack_signature(error),
        "parity_difference_magnitudes": _safe_parity(parity),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n.a."
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, tuple):
        return ", ".join(map(str, value)) if value else "none"
    return str(value)


def build_error_report(**kwargs: Any) -> str:
    """Render a parameter-only Markdown error report from safe object metadata."""
    p = safe_error_payload(**kwargs)
    env = p["environment"]
    res = p["result"]
    cfg = p["config"]
    err = p["error"]
    parity = p["parity_difference_magnitudes"]
    stack = p["stack_signature"]

    lines = [
        "# econhdfe privacy-minimized error / mismatch report",
        "",
        "> Auto-generated from an explicit allowlist of scalar/counter metadata. ",
        "> No observations, coefficient values, variable names, file paths, commands, raw logs, or full traceback text are included.",
        "",
        "## 1. Report identity",
        "",
        f"- Report schema: `{p['report_schema']}`",
        f"- Issue type: {_fmt(p['issue_type'])}",
        f"- Severity: {_fmt(p['severity'])}",
        "",
        "## 2. Runtime/environment parameters",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    for key in ("python", "econhdfe", "numpy", "scipy", "os_family", "cpu_architecture"):
        lines.append(f"| {key} | {_fmt(env.get(key))} |")

    lines += ["", "## 3. Anonymous result/specification fingerprint", ""]
    if res:
        lines += ["| Parameter | Value |", "|---|---|"]
        for key, value in res.items():
            lines.append(f"| {key} | {_fmt(value)} |")
    else:
        lines.append("- No result object available.")

    lines += ["", "## 4. Numerical/execution configuration", ""]
    if cfg:
        for block, values in cfg.items():
            lines.append(f"### {block}")
            lines.append("")
            lines += ["| Parameter | Value |", "|---|---|"]
            for key, value in values.items():
                lines.append(f"| {key} | {_fmt(value)} |")
            lines.append("")
    else:
        lines.append("- No configuration objects supplied.")

    lines += ["", "## 5. Error fingerprint", ""]
    if err:
        lines += [
            f"- Exception class: `{_fmt(err.get('type'))}`",
            f"- Error code: `{_fmt(err.get('code'))}`",
            f"- Stage: `{_fmt(err.get('stage'))}`",
        ]
        details = err.get("numeric_details") or {}
        if details:
            lines.append("- Numeric/counter details:")
            for key, value in details.items():
                lines.append(f"  - {key}: {_fmt(value)}")
    else:
        lines.append("- No exception object supplied.")

    lines += ["", "## 6. Safe stack signature", ""]
    if stack:
        for frame in stack:
            lines.append(f"- `{frame}`")
    else:
        lines.append("- n.a.")

    lines += ["", "## 7. Reference-package parity difference magnitudes", ""]
    if parity:
        lines += ["| Quantity | Difference magnitude |", "|---|---:|"]
        for key, value in parity.items():
            lines.append(f"| {key} | {_fmt(value)} |")
    else:
        lines.append("- n.a.")

    lines += [
        "",
        "## 8. Privacy contract",
        "",
        "This report intentionally excludes raw/sampled/synthetic observations, actual variable/entity identifiers, file paths or filenames, exact commands/scripts, coefficient or fitted/residual values, raw logs, full tracebacks, stack locals, and attachments. If additional information is required, request narrower parameters/counters first. Real-data benchmarking is a separate explicit-consent workflow.",
        "",
    ]
    return "\n".join(lines)


def write_error_report(path: str | Path, **kwargs: Any) -> Path:
    target = Path(path)
    target.write_text(build_error_report(**kwargs), encoding="utf-8")
    return target


def write_template(kind: str, path: str | Path) -> Path:
    if kind == "error":
        text = error_report_template()
    elif kind == "benchmark":
        text = benchmark_report_template()
    else:
        raise ValueError("kind must be 'error' or 'benchmark'")
    target = Path(path)
    target.write_text(text, encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="econhdfe-report",
        description="Write econhdfe privacy-safe error or benchmark report templates.",
    )
    parser.add_argument("kind", choices=("error", "benchmark"))
    parser.add_argument("--output", "-o", type=Path, help="Write to this Markdown file instead of stdout.")
    args = parser.parse_args(argv)
    text = error_report_template() if args.kind == "error" else benchmark_report_template()
    if args.output is None:
        sys.stdout.write(text)
        if text and not text.endswith("\n"):
            sys.stdout.write("\n")
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
