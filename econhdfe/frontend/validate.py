from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import pandas as pd

from ..errors import DataTypeError, MissingDataError, ShapeError, InputError, SpecificationError
from .roles import VariableRole
from .report import PreflightIssue, PreflightReport


def _role(role) -> VariableRole:
    return role if isinstance(role, VariableRole) else VariableRole(str(role))


def _examples(a, limit=3):
    vals = np.asarray(a, dtype=object).reshape(-1)
    out = []
    for x in vals:
        if pd.isna(x):
            continue
        s = repr(x)
        if s not in out:
            out.append(s)
        if len(out) >= limit:
            break
    return out


def require_numeric(values, *, name: str, role=VariableRole.REGRESSOR,
                    ndim: int | tuple[int, ...] | None = None, finite: bool = True,
                    nonnegative: bool = False, positive: bool = False) -> np.ndarray:
    r = _role(role)
    raw = np.asarray(values)
    if ndim is not None:
        allowed = (ndim,) if isinstance(ndim, int) else tuple(ndim)
        if raw.ndim not in allowed:
            raise ShapeError(
                f"{r.value} {name!r} must have dimension {allowed}; got {raw.ndim}",
                details={"variable": name, "role": r.value, "shape": raw.shape},
            )
    if raw.dtype.kind not in "biufc":
        raise DataTypeError(
            f"{r.value} {name!r} must be numeric; got dtype {raw.dtype}",
            details={"variable": name, "role": r.value, "dtype": str(raw.dtype),
                     "examples": _examples(raw)},
            suggestion=(
                "If the column contains numeric text, convert it explicitly with "
                "pandas.to_numeric(). If it is categorical, specify it as a factor/FE instead."
            ),
        )
    out = np.asarray(raw, dtype=np.float64)
    if finite and np.any(~np.isfinite(out)):
        bad = int(np.count_nonzero(~np.isfinite(out)))
        raise MissingDataError(
            f"{r.value} {name!r} contains {bad} missing or non-finite value(s)",
            code="input.non_finite", details={"variable": name, "role": r.value, "count": bad},
        )
    if positive and np.any(out <= 0):
        raise InputError(
            f"{r.value} {name!r} must be strictly positive",
            code="input.non_positive", details={"variable": name, "role": r.value},
        )
    if nonnegative and np.any(out < 0):
        raise InputError(
            f"{r.value} {name!r} must be nonnegative",
            code="input.negative", details={"variable": name, "role": r.value},
        )
    return out


def require_identifier(values, *, name: str, role=VariableRole.FE) -> np.ndarray:
    r = _role(role)
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ShapeError(
            f"{r.value} {name!r} must be one-dimensional",
            details={"variable": name, "role": r.value, "shape": raw.shape},
        )
    missing = pd.isna(raw)
    if np.any(missing):
        raise MissingDataError(
            f"{r.value} {name!r} contains missing identifiers",
            details={"variable": name, "role": r.value, "count": int(np.sum(missing))},
            suggestion="Drop/fill missing identifiers explicitly before estimation.",
        )
    return raw


def preflight_dataframe(data: pd.DataFrame, roles: Mapping[str, str | VariableRole], *,
                        model: str | None = None, level: str = "basic") -> PreflightReport:
    """Cheap deterministic schema checks without fitting a model.

    This deliberately does not attempt collinearity, separation, FE rank, or
    weak-identification diagnostics; those belong to estimator stages.
    """
    if level not in {"basic", "strict"}:
        raise SpecificationError(
            "preflight level must be 'basic' or 'strict'",
            code="specification.preflight_level",
            stage="frontend",
            details={"level": level, "allowed": ["basic", "strict"]},
        )
    issues = []
    for name, role in roles.items():
        r = _role(role)
        if name not in data.columns:
            issues.append(PreflightIssue(
                "input.missing_column", f"column {name!r} is not present", name, r.value,
                suggestion="Check the requested column names.",
            ))
            continue
        s = data[name]
        if r.requires_numeric and not pd.api.types.is_numeric_dtype(s.dtype):
            issues.append(PreflightIssue(
                "input.non_numeric", f"{r.value} {name!r} must be numeric; got dtype {s.dtype}",
                name, r.value,
                suggestion="Convert numeric text explicitly, or mark categorical variables as factors/FE.",
                details={"dtype": str(s.dtype), "examples": _examples(s.to_numpy())},
            ))
            continue
        if s.isna().any():
            issues.append(PreflightIssue(
                "input.missing", f"{r.value} {name!r} contains missing values",
                name, r.value, details={"count": int(s.isna().sum())},
            ))
            continue
        if r.requires_numeric:
            vals = s.to_numpy(dtype=np.float64, copy=False)
            if np.any(~np.isfinite(vals)):
                issues.append(PreflightIssue(
                    "input.non_finite", f"{r.value} {name!r} contains non-finite values",
                    name, r.value,
                ))
            if r is VariableRole.OUTCOME and model in {"ppml", "ivppml"} and np.any(vals < 0):
                issues.append(PreflightIssue(
                    "input.negative_outcome", f"Poisson outcome {name!r} must be nonnegative",
                    name, r.value,
                ))
            if r is VariableRole.EXPOSURE and np.any(vals <= 0):
                issues.append(PreflightIssue(
                    "input.non_positive_exposure", f"exposure {name!r} must be strictly positive",
                    name, r.value,
                ))
    if level == "strict":
        for name, role in roles.items():
            if name not in data.columns:
                continue
            r = _role(role)
            s = data[name]
            if s.isna().any():
                continue
            if r is VariableRole.CLUSTER:
                ng = int(s.nunique(dropna=True))
                if ng < 30:
                    issues.append(PreflightIssue(
                        "warning.few_clusters",
                        f"cluster {name!r} has only {ng} groups; small-cluster inference may be unreliable",
                        name, r.value, severity="warning", details={"n_clusters": ng},
                    ))
            if r.requires_numeric and pd.api.types.is_numeric_dtype(s.dtype):
                vals = s.to_numpy(dtype=np.float64, copy=False)
                finite = vals[np.isfinite(vals)]
                if finite.size and r in {VariableRole.REGRESSOR, VariableRole.EXOGENOUS, VariableRole.ENDOGENOUS, VariableRole.INSTRUMENT}:
                    scale = max(float(np.max(np.abs(finite))), 1.0)
                    if float(np.std(finite)) <= 1e-12 * scale:
                        issues.append(PreflightIssue(
                            "warning.near_constant", f"{r.value} {name!r} is nearly constant",
                            name, r.value, severity="warning",
                        ))
                if finite.size and r is VariableRole.WEIGHT:
                    pos = finite[finite > 0]
                    if pos.size and float(np.max(pos) / np.min(pos)) > 1e6:
                        issues.append(PreflightIssue(
                            "warning.extreme_weights", f"weight {name!r} spans more than six orders of magnitude",
                            name, r.value, severity="warning",
                            details={"max_min_ratio": float(np.max(pos) / np.min(pos))},
                        ))
                if finite.size and r is VariableRole.OUTCOME and model in {"ppml", "ivppml"}:
                    zero_share = float(np.mean(finite == 0))
                    if zero_share > 0.99:
                        issues.append(PreflightIssue(
                            "warning.extreme_zero_share",
                            f"Poisson outcome {name!r} is {zero_share:.2%} zeros; separation/convergence deserves inspection",
                            name, r.value, severity="warning", details={"zero_share": zero_share},
                        ))
    return PreflightReport(tuple(issues))
