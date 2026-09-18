from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ...hdfe.plan import FEPlan
from ...compute.linalg import weighted_lstsq


@dataclass(slots=True)
class ReLUInfo:
    separated: np.ndarray
    iterations: int
    converged: bool


def relu_separation(
    y, X, plan: FEPlan, *, tol=1e-4, zero_tol=1e-8, hdfe_tol=1e-9,
    max_iter=100, engine="replica", projector=None,
):
    """IR/ReLU separation certificate following Correia-Guimaraes-Zylkin."""
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n = len(y)
    boundary = y == 0
    if not np.any(boundary):
        return ReLUInfo(np.zeros(n, dtype=bool), 0, True)

    u = boundary.astype(np.float64)
    M = float(np.ceil(np.finfo(float).eps ** -0.5))
    # Upstream create_mask(N, M, boundary, 1): equality constraints (y>0)
    # receive the large weight, boundary observations weight one.
    weights = np.where(boundary, 1.0, M)
    norm_b = np.linalg.norm(X[boundary], 1) if X.size else 0.0
    norm_i = np.linalg.norm(X[~boundary], 1) if X.size else 0.0
    ratio = max(norm_b / norm_i, 1.0) if norm_i > 0 else 1.0

    if plan.groups:
        method = "map" if engine == "optimized" else "lsmr"
        if projector is None:
            from ...hdfe.weighted_projection import WeightedFEProjector
            projector = WeightedFEProjector(plan.groups, engine=engine, method=method)
        absorber = projector.prepare(weights, tol=max(tol * tol, 1e-13))
    else:
        absorber = None
    xtilde = None
    utilde = None
    u_last = None
    for it in range(1, max_iter + 1):
        if it == 1:
            utilde_input = u
        else:
            utilde_input = u + utilde - u_last
        u_last = u.copy()

        if absorber is not None:
            utilde = absorber.residualize(utilde_input)
            if it == 1:
                xtilde = absorber.residualize(X) if X.shape[1] else X
        else:
            utilde = utilde_input.copy()
            if it == 1:
                xtilde = X

        if X.shape[1]:
            _, resid = weighted_lstsq(xtilde, utilde, weights, fast=False)
        else:
            resid = utilde.copy()
        epsilon = max(float(np.dot(u, u) / M * ratio * ratio), 1e-8)
        delta = epsilon + tol
        xbd = u - resid
        xbd[~boundary] = 0.0
        near = boundary & (xbd >= -0.1 * delta) & (xbd <= delta)
        xbd[near] = 0.0
        if np.all(xbd[boundary] >= 0.0):
            return ReLUInfo(boundary & (xbd > 0.0), it, True)

        resid[np.abs(resid) <= zero_tol] = 0.0
        if np.min(resid[boundary]) >= 0.0:
            xbd[boundary & (resid > delta)] = 0.0
            return ReLUInfo(boundary & (xbd > 0.0), it, True)
        u[boundary] = np.maximum(xbd[boundary], 0.0)

    return ReLUInfo(np.zeros(n, dtype=bool), max_iter, False)
