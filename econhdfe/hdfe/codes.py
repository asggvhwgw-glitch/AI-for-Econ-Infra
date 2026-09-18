"""Validation of the observed-level dense categorical coding contract."""
from __future__ import annotations
import numpy as np


def validate_dense_codes(values, *, dtype=np.int64):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in "biuf":
        raise ValueError("dense FE codes must be a one-dimensional integer-valued numeric array")
    if not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a != np.floor(a)):
        raise ValueError("dense FE codes must be finite nonnegative integers")
    # A contiguous observed code cannot exceed n-1. Check before casting or
    # allocating max(code)+1 arrays (also prevents overflow/unbounded allocation).
    if a.size and np.max(a) >= a.size:
        raise ValueError("dense FE codes must be zero-based and contiguous on observed levels")
    if a.size and np.max(a) > np.iinfo(dtype).max:
        raise ValueError("dense FE code exceeds the supported integer range")
    codes = a.astype(dtype, copy=False)
    if codes.size:
        observed = np.unique(codes)
        if len(observed) != int(codes.max()) + 1:
            raise ValueError("dense FE codes must be zero-based and contiguous on observed levels")
    return codes
