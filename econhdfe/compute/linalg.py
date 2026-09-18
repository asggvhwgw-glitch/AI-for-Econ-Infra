from __future__ import annotations
import numpy as np
from scipy.linalg import qr


from .wls import weighted_lstsq
from .design_ops import gram_matrix
from .design_plan import StructuralDesignPlan
from .block_design import BlockDesign


def _clear_full_column_rank(X, *, tol: float, structure: StructuralDesignPlan | None = None) -> bool:
    """Cheap K-space certificate for clearly full-rank tall designs.

    The certificate is intentionally conservative.  Columns are normalized in
    Gram space and only well-separated positive eigenvalues are accepted.  Any
    zero, non-finite, short/wide, or numerically ambiguous design falls back to
    the established pivoted-QR resolver in ``independent_columns``.
    """
    n, k = X.shape
    if k == 0:
        return True
    if n < k:
        return False
    if isinstance(X, BlockDesign):
        G = X.gram()
    else:
        X = np.asarray(X, dtype=np.float64)
        if not np.all(np.isfinite(X)):
            return False
        G = gram_matrix(X, structure=structure)
    if not np.all(np.isfinite(G)):
        return False
    diag = np.diag(G)
    if np.any(diag <= 0.0) or not np.all(np.isfinite(diag)):
        return False
    inv_norm = 1.0 / np.sqrt(diag)
    C = G * inv_norm[:, None] * inv_norm[None, :]
    C = (C + C.T) * 0.5
    try:
        eig = np.linalg.eigvalsh(C)
    except np.linalg.LinAlgError:
        return False
    if eig.size == 0 or not np.all(np.isfinite(eig)):
        return False
    scale = max(float(eig[-1]), 1.0)
    # Squared relative singular-value threshold plus a roundoff guard.  The
    # factor of 16 makes this a certificate for easy cases, not a replacement
    # for QR in numerically marginal designs.
    threshold = max((16.0 * float(tol)) ** 2 * scale,
                    64.0 * np.finfo(np.float64).eps * max(1, k) * scale)
    return bool(float(eig[0]) > threshold)


def independent_columns(X, *, tol=1e-10, structure: StructuralDesignPlan | None = None):
    is_block = isinstance(X, BlockDesign)
    if not is_block:
        X = np.asarray(X, dtype=np.float64)
    if X.shape[1] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if _clear_full_column_rank(X, tol=float(tol), structure=structure):
        keep = np.arange(X.shape[1], dtype=np.int64)
        return keep, np.empty(0, dtype=np.int64)
    # Ambiguous/block-rank-deficient cases deliberately materialize only for
    # the established pivoted-QR fallback, preserving its exact omit semantics.
    if is_block:
        X = X.materialize()
    _, R, piv = qr(X, mode="economic", pivoting=True)
    diag = np.abs(np.diag(R))
    scale = diag[0] if diag.size else 0.0
    rank = int(np.sum(diag > tol * max(scale, 1.0)))
    keep = np.sort(np.asarray(piv[:rank], dtype=np.int64))
    drop = np.setdiff1d(np.arange(X.shape[1]), keep, assume_unique=True)
    return keep, drop
