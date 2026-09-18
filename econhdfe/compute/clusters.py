from __future__ import annotations

from collections.abc import Iterable
import numpy as np

from .encoding import factorize_1d
from ..errors import InputError, ShapeError


def normalize_cluster_arrays(clusters, *, nobs: int, required: bool = False) -> tuple[np.ndarray, ...]:
    """Return dense int32 cluster codes with one array per clustering dimension.

    This is the canonical post-estimation cluster normalizer. Estimator frontends
    may already supply dense codes; those arrays are reused when possible.
    """
    if clusters is None:
        specs = []
    elif isinstance(clusters, np.ndarray):
        a = np.asarray(clusters)
        if a.ndim == 1:
            specs = [a]
        elif a.ndim == 2:
            specs = [a[:, j] for j in range(a.shape[1])]
        else:
            raise ShapeError("cluster array must be one- or two-dimensional")
    else:
        specs = list(clusters)

    if required and not specs:
        raise InputError(
            "cluster inference requires at least one clustering dimension",
            code="inference.cluster_required", stage="inference",
        )
    if len(specs) > 10:
        raise InputError(
            "at most 10 cluster dimensions are supported",
            code="inference.cluster_dimensions", stage="inference",
            details={"dimensions": len(specs)},
        )

    out: list[np.ndarray] = []
    for j, values in enumerate(specs):
        a = np.asarray(values)
        if a.ndim != 1 or len(a) != int(nobs):
            raise ShapeError(
                "each cluster dimension must be one-dimensional with nobs entries",
                details={"dimension": j, "expected": int(nobs), "shape": a.shape},
            )
        codes, nlev = factorize_1d(a)
        if nlev < 2:
            raise InputError(
                "cluster inference requires at least two clusters per dimension",
                code="inference.insufficient_clusters", stage="inference",
                details={"dimension": j, "clusters": int(nlev)},
            )
        out.append(codes)
    return tuple(out)


def normalize_vce_clusters(clusters, *, nobs: int) -> tuple[np.ndarray, ...]:
    """Encode arbitrary cluster labels on the estimation sample for a VCE.

    Counts are observed levels, never max(label)+1. Degenerate dimensions
    invalidate clustered inference instead of falling back to another VCE.
    """
    from ..errors import InferenceError
    try:
        return normalize_cluster_arrays(clusters, nobs=nobs, required=True)
    except InputError as exc:
        if exc.code in {"inference.insufficient_clusters", "inference.cluster_required"}:
            raise InferenceError(str(exc), code=exc.code, details=exc.details) from exc
        raise
