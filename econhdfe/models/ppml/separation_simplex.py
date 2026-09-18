from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ...hdfe.plan import FEPlan
from ...compute.linalg import independent_columns


@dataclass(slots=True)
class SimplexInfo:
    separated: np.ndarray
    suspect_columns: np.ndarray
    iterations: int
    converged: bool


def _drop_zero_columns(X, tol):
    if X.shape[1] == 0:
        return X, np.empty(0, dtype=np.int64)
    keep = np.max(np.abs(X), axis=0) > tol
    return X[:, keep], np.flatnonzero(keep)


def _presolve(X, tol):
    """Mata ppmlhdfe simplex presolve, translated without estimator state."""
    X = np.asarray(X, dtype=np.float64).copy()
    n, k0 = X.shape
    A = np.zeros((n, k0), dtype=np.float64)
    nonbasic = np.full(k0, -1, dtype=np.int64)
    row_keep = np.ones(n, dtype=bool)
    col_keep = np.ones(k0, dtype=bool)
    for j in range(k0):
        nz = np.flatnonzero(np.abs(X[:, j]) > tol)
        if nz.size == 0:
            col_keep[j] = False
            continue
        i = int(nz[0])
        nonbasic[j] = i
        row_keep[i] = False
        pivot = -1.0 / X[i, j]
        pivot_row = pivot * X[i, :]
        pivot_col = X[:, j].copy()
        A[i, j] = 1.0
        A += np.outer(pivot_col * pivot, A[i, :])
        X += np.outer(pivot_col, pivot_row)
        X[np.abs(X) <= tol] = 0.0
    kept_cols = np.flatnonzero(col_keep)
    Xred = A[row_keep][:, kept_cols]
    nb = nonbasic[kept_cols]
    basic = np.flatnonzero(row_keep)
    return Xred, basic, nb, row_keep, kept_cols


def simplex_flag_separated_obs(X, *, tol=1e-12, max_iter=1000):
    """Modified zero-RHS simplex used by ppmlhdfe.

    Returns a Boolean mask over the rows of X: True means separated.  The
    tableau/pivot rules intentionally mirror the upstream Mata routine; costs
    are labels (0/1), not dual values to be recomputed after a pivot.
    """
    X = np.asarray(X, dtype=np.float64)
    n0 = X.shape[0]
    if n0 == 0 or X.shape[1] == 0:
        return np.zeros(n0, dtype=bool), 0, True
    X, _ = _drop_zero_columns(X, tol)
    if X.shape[1] == 0:
        return np.zeros(n0, dtype=bool), 0, True

    Xr, basic, nonbasic, row_keep, _ = _presolve(X, tol)
    k = len(nonbasic)
    if n0 == k:
        return np.ones(n0, dtype=bool), 0, True

    c_basic = np.ones(len(basic), dtype=np.float64)
    c_nonbasic = np.ones(k, dtype=np.float64)
    keep_mask = row_keep.copy()
    last_pivot_row = -1

    for it in range(1, max_iter + 1):
        r = c_nonbasic - c_basic @ Xr
        r[np.abs(r) <= max(tol, np.finfo(float).eps * 100)] = 0.0
        j = int(np.argmax(r))
        if r[j] <= 0:
            costs = np.empty(n0, dtype=np.float64)
            costs[basic] = c_basic
            costs[nonbasic] = c_nonbasic
            return costs == 0.0, it, True

        pivot_col = Xr[:, j].copy()
        if pivot_col.size == 0 or np.all(pivot_col <= 0):
            neg = pivot_col < 0
            c_basic[neg] = 0.0
            c_nonbasic[j] = 0.0
            continue

        maxv = np.max(pivot_col)
        candidates = np.flatnonzero(pivot_col == maxv)
        i = int(candidates[0])
        if i == last_pivot_row and candidates.size > 1:
            i = int(candidates[1])
        entering = int(nonbasic[j])
        leaving = int(basic[i])
        pivot = float(Xr[i, j])

        scaled_row = Xr[i, :].copy() / pivot
        Xr -= pivot_col[:, None] * scaled_row[None, :]
        Xr[i, :] = scaled_row
        Xr[:, j] = -pivot_col / pivot
        Xr[i, j] = 1.0 / pivot
        Xr[np.abs(Xr) <= np.finfo(float).eps * 10] = 0.0

        keep_mask[entering] = True
        keep_mask[leaving] = False
        nonbasic[j] = leaving
        basic[i] = entering
        c_nonbasic[j], c_basic[i] = c_basic[i], c_nonbasic[j]
        last_pivot_row = i

    return np.zeros(n0, dtype=bool), max_iter, False

def mixed_simplex_separation(
    y, X, plan: FEPlan, weights=None, *, tol=1e-12, hdfe_tol=1e-9,
    max_iter=1000, engine="replica", projector=None,
):
    """Faithful modular form of ppmlhdfe's HDFE+X simplex precheck.

    The positive-outcome sample identifies regressors that are redundant there;
    their residual directions are then tested on y=0 observations.
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    n = len(y)
    if X.shape[1] == 0 or np.all(y > 0):
        return SimplexInfo(np.zeros(n, dtype=bool), np.empty(0, dtype=np.int64), 0, True)
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    pos = y > 0
    zero = ~pos
    wp = np.where(pos, w, 0.0)

    if plan.groups:
        if projector is None:
            from ...hdfe.weighted_projection import WeightedFEProjector
            projector = WeightedFEProjector(plan.groups, engine=engine, method="map")
        R, _ = projector.residualize(X, wp, tol=hdfe_tol, return_info=True)
    else:
        R = X.copy()

    swp = np.sqrt(w[pos])
    keep, suspect = independent_columns(R[pos] * swp[:, None], tol=max(tol, 1e-12))
    if suspect.size == 0:
        return SimplexInfo(np.zeros(n, dtype=bool), suspect, 0, True)

    Z = R[:, suspect].copy()
    if keep.size:
        A = R[:, keep]
        b, *_ = np.linalg.lstsq(A[pos] * swp[:, None], Z[pos] * swp[:, None], rcond=None)
        Z -= A @ b
    Z0 = Z[zero]

    sep0 = np.zeros(Z0.shape[0], dtype=bool)
    active_cols = np.ones(Z0.shape[1], dtype=bool)
    # Exact cheap sign presolve used before the tableau.
    for j in range(Z0.shape[1]):
        c = Z0[:, j]
        if np.max(np.abs(c)) <= tol:
            active_cols[j] = False
        elif np.all(c >= -tol):
            sep0 |= c > tol
            active_cols[j] = False
        elif np.all(c <= tol):
            sep0 |= c < -tol
            active_cols[j] = False

    remaining_rows = ~sep0
    Zr = Z0[remaining_rows][:, active_cols]
    iters = 0; converged = True
    if Zr.size and Zr.shape[1]:
        local, iters, converged = simplex_flag_separated_obs(Zr, tol=tol, max_iter=max_iter)
        idx = np.flatnonzero(remaining_rows)
        sep0[idx[local]] = True

    sep = np.zeros(n, dtype=bool)
    sep[np.flatnonzero(zero)[sep0]] = True
    return SimplexInfo(sep, suspect, iters, converged)
