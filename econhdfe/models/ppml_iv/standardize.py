from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ...compute.block_design import BlockDesign


def _quadvariance_scale_block(A: BlockDesign, weights=None):
    """Exact weighted sample-variance scales without global dense materialization."""
    A._ensure_layout()
    n, k = A.shape
    if k == 0:
        return np.empty(0, dtype=np.float64)
    if n <= 1:
        return np.ones(k, dtype=np.float64)
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("standardization weights must be finite, positive, and match nobs")
    ld = np.longdouble
    wl = w.astype(ld, copy=False)
    sw = np.sum(wl, dtype=ld)
    if not np.isfinite(sw) or sw <= 0:
        raise ValueError("standardization weights must have positive finite sum")

    sums = np.zeros(k, dtype=ld)
    present_w = np.zeros(k, dtype=ld)
    for block in A.blocks:
        rows = block.rows
        wb = wl[rows]
        vals = block.values.astype(ld, copy=False)
        cols = block.columns
        sums[cols] += np.sum(wb[:, None] * vals, axis=0, dtype=ld)
        present_w[cols] += np.sum(wb, dtype=ld)
    means = sums / sw

    ss = np.zeros(k, dtype=ld)
    for block in A.blocks:
        rows = block.rows
        wb = wl[rows]
        vals = block.values.astype(ld, copy=False)
        cols = block.columns
        d = vals - means[cols][None, :]
        ss[cols] += np.sum(wb[:, None] * d * d, axis=0, dtype=ld)
    # A column omitted from a row component is an exact structural zero, so
    # those rows contribute w * (0 - mean)^2 to the centered variance.
    ss += (sw - present_w) * means * means
    factor = ld(n) / (sw * ld(n - 1))
    scale = np.sqrt(np.maximum(ss * factor, ld(0)))
    out = np.ones(k, dtype=np.float64)
    finite = np.isfinite(scale) & (scale > 0)
    out[finite] = np.asarray(scale[finite], dtype=np.float64)
    return out


def _quadvariance_scale(A, weights=None):
    """Stata/Mata ``sqrt(quadvariance())`` column scales.

    Mata's weighted variance normalizes the supplied positive weights to the
    physical row count and uses the sample denominator ``n - 1``.  The
    calculation here mirrors that formula while using long-double reductions
    when the platform provides them.  Only scaling is applied; data are not
    demeaned before estimation.
    """
    if isinstance(A, BlockDesign):
        return _quadvariance_scale_block(A, weights)
    A = np.asarray(A, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    n, k = A.shape
    if k == 0:
        return np.empty(0, dtype=np.float64)
    if n <= 1:
        return np.ones(k, dtype=np.float64)

    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (n,) or np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("standardization weights must be finite, positive, and match nobs")

    # ``longdouble`` is 80/128-bit on platforms that support it and aliases
    # float64 elsewhere.  Either way the formula remains deterministic.
    ld = np.longdouble
    wl = w.astype(ld, copy=False)
    sw = np.sum(wl, dtype=ld)
    if not np.isfinite(sw) or sw <= 0:
        raise ValueError("standardization weights must have positive finite sum")

    out = np.ones(k, dtype=np.float64)
    factor = ld(n) / (sw * ld(n - 1))
    for j in range(k):
        x = A[:, j].astype(ld, copy=False)
        mean = np.sum(wl * x, dtype=ld) / sw
        d = x - mean
        var = factor * np.sum(wl * d * d, dtype=ld)
        scale = np.sqrt(max(var, ld(0)))
        sf = float(scale)
        if np.isfinite(sf) and sf > 0:
            out[j] = sf
    return out


# Backward-private alias kept inside the module only; callers should use the
# Standardization object rather than depend on a scaling implementation detail.
_weighted_scale = _quadvariance_scale


@dataclass(frozen=True, slots=True)
class IVPPMLStandardization:
    x_scale: np.ndarray
    z_scale: np.ndarray

    @classmethod
    def fit(cls, X, excluded, weights=None):
        return cls(_quadvariance_scale(X, weights), _quadvariance_scale(excluded, weights))

    @classmethod
    def identity(cls, kx: int, kz: int):
        return cls(np.ones(kx, dtype=np.float64), np.ones(kz, dtype=np.float64))

    def transform(self, X, excluded):
        Xs = X.scale_columns(self.x_scale) if isinstance(X, BlockDesign) else np.asarray(X, dtype=np.float64) / self.x_scale
        Zs = excluded.scale_columns(self.z_scale) if isinstance(excluded, BlockDesign) else np.asarray(excluded, dtype=np.float64) / self.z_scale
        return Xs, Zs

    def unscale_coef(self, beta):
        return np.asarray(beta, dtype=np.float64) / self.x_scale

    def unscale_vcov(self, V):
        s = self.x_scale
        return np.asarray(V, dtype=np.float64) / (s[:, None] * s[None, :])


    def unscale_first_stage(self, first_stage, n_exog):
        """Map Q_std Gamma_std = X_std to original within-variable units."""
        q_scale = np.r_[self.x_scale[:n_exog], self.z_scale]
        gamma = np.asarray(first_stage, dtype=np.float64)
        if gamma.shape != (len(q_scale), len(self.x_scale)):
            raise ValueError("first_stage shape does not match role scales")
        return gamma * self.x_scale[None, :] / q_scale[:, None]
