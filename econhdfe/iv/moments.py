from __future__ import annotations

import numpy as np


def additive_score_matrix(residual, instruments, *, weights=None) -> np.ndarray:
    """Observation-level additive IV scores ``q_i * residual_i``.

    For linear IV ``residual = y - X beta``.  For IV-PPML the same primitive
    applies with ``residual = y - mu`` and ``q`` equal to the full instrument
    matrix (included exogenous regressors plus excluded instruments).
    """
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    q = np.asarray(instruments, dtype=np.float64)
    if q.ndim == 1:
        q = q[:, None]
    if q.shape[0] != len(r):
        raise ValueError("residual and instrument matrices must have the same number of rows")
    scores = q * r[:, None]
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(w) != len(r):
            raise ValueError("weights must have the same number of rows")
        scores = scores * w[:, None]
    return scores


def additive_moments(residual, instruments, *, weights=None, normalize=True) -> np.ndarray:
    scores = additive_score_matrix(residual, instruments, weights=weights)
    total = scores.sum(axis=0)
    return total / len(scores) if normalize and len(scores) else total
