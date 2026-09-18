from __future__ import annotations

import numpy as np
from .stable_linalg import equilibrated_lstsq, equilibrated_gram_inverse


def as_2d(x, n):
    """Return a dense float64 matrix with ``n`` rows.

    ``None`` maps to an empty design. One-dimensional inputs become a single
    column. This is a generic numerical primitive used by OLS/IV/PPML layers.
    """
    if x is None:
        return np.empty((n, 0), dtype=np.float64)
    a = np.asarray(x, dtype=np.float64)
    return a[:, None] if a.ndim == 1 else a


def weighted_arrays(y, X, Z=None, weights=None):
    """Apply square-root observation weights to dense arrays."""
    if weights is None:
        return y, X, Z, None
    sw = np.sqrt(np.asarray(weights, dtype=np.float64))
    yw = y * sw
    Xw = X * sw[:, None]
    Zw = None if Z is None else Z * sw[:, None]
    return yw, Xw, Zw, sw


def weighted_lstsq(X, y, w, *, fast=False, resid_out=None, chunk_rows=250_000):
    """Weighted least squares with a bounded-memory fast path.

    ``fast=True`` uses normal equations, matching the intermediate PPML solve,
    while avoiding an N x K weighted-design temporary. The exact path uses
    ``numpy.linalg.lstsq``. ``resid_out`` allows iterative estimators to reuse
    an existing N-vector workspace.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if X.shape[1] == 0:
        if resid_out is None:
            return np.empty(0), y.copy()
        resid_out[:] = y
        return np.empty(0), resid_out

    if fast:
        k = X.shape[1]
        G = np.zeros((k, k), dtype=np.float64)
        rhs = np.zeros(k, dtype=np.float64)
        chunk = max(1, int(chunk_rows))
        for lo in range(0, len(y), chunk):
            hi = min(len(y), lo + chunk)
            xb = X[lo:hi]
            wb = w[lo:hi]
            wx = xb * wb[:, None]
            G += xb.T @ wx
            rhs += xb.T @ (wb * y[lo:hi])
        beta = equilibrated_gram_inverse(G) @ rhs
    else:
        sw = np.sqrt(w)
        beta, _, _ = equilibrated_lstsq(X * sw[:, None], y * sw)

    if resid_out is None:
        return beta, y - X @ beta
    np.dot(X, beta, out=resid_out)
    resid_out *= -1.0
    resid_out += y
    return beta, resid_out
