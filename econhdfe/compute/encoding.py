from __future__ import annotations
import numpy as np
import pandas as pd


def factorize_1d(x) -> tuple[np.ndarray, int]:
    """Dense int32 codes; missing values are treated as a regular category.

    Already-dense nonnegative integer codes are certified and reused. This is
    common in large econometric pipelines and avoids a needless N-row hash
    factorization plus another code array.
    """
    a = np.asarray(x)
    if a.ndim == 1 and a.size and np.issubdtype(a.dtype, np.integer):
        lo = int(a.min()); hi = int(a.max())
        if lo == 0 and hi < a.size and hi <= np.iinfo(np.int32).max:
            counts = np.bincount(a.astype(np.int64, copy=False), minlength=hi + 1)
            if counts.size == hi + 1 and bool(np.all(counts > 0)):
                return a.astype(np.int32, copy=False), hi + 1
    codes, uniques = pd.factorize(x, sort=False, use_na_sentinel=False)
    return np.asarray(codes, dtype=np.int32), len(uniques)


def factorize_interaction(columns) -> tuple[np.ndarray, int]:
    """Factorize a categorical interaction without materializing string keys.

    Common economic FE interactions (city×year, industry×year, etc.) are
    encoded with a mixed-radix uint64 key built from dense component codes.
    This avoids constructing a large pandas MultiIndex/object interaction at
    10M+ rows.  Extremely wide interactions that would overflow uint64 fall
    back to pandas MultiIndex factorization for correctness.
    """
    columns = list(columns)
    if not columns:
        raise ValueError("interaction requires at least one column")
    if len(columns) == 1:
        return factorize_1d(columns[0])

    n = len(columns[0])
    key = np.zeros(n, dtype=np.uint64)
    stride = 1
    limit = np.iinfo(np.uint64).max
    for col in columns:
        if len(col) != n:
            raise ValueError("interaction columns must have equal length")
        codes, nlev = factorize_1d(col)
        nlev = int(nlev)
        if nlev and stride > limit // nlev:
            mi = pd.MultiIndex.from_arrays(columns)
            out, uniques = pd.factorize(mi, sort=False, use_na_sentinel=False)
            return np.asarray(out, dtype=np.int32), len(uniques)
        key += np.asarray(codes, dtype=np.uint64) * np.uint64(stride)
        stride *= max(nlev, 1)

    out, uniques = pd.factorize(key, sort=False, use_na_sentinel=False)
    return np.asarray(out, dtype=np.int32), len(uniques)


def combine_dense_codes(code_arrays, nlevels) -> tuple[np.ndarray, int]:
    """Factorize an interaction from already-dense component codes.

    This is the cached-component counterpart of :func:`factorize_interaction`;
    it avoids re-factorizing the same year/city/industry component across many
    high-dimensional interaction FEs.
    """
    code_arrays = [np.asarray(c, dtype=np.int32) for c in code_arrays]
    nlevels = [int(v) for v in nlevels]
    if not code_arrays:
        raise ValueError("interaction requires at least one component")
    if len(code_arrays) == 1:
        c = code_arrays[0]
        return c, (int(c.max()) + 1 if c.size else 0)
    n = len(code_arrays[0])
    key = np.zeros(n, dtype=np.uint64)
    stride = 1
    limit = np.iinfo(np.uint64).max
    for codes, nlev in zip(code_arrays, nlevels, strict=False):
        if len(codes) != n:
            raise ValueError("interaction code arrays must have equal length")
        if nlev and stride > limit // nlev:
            # Rare overflow fallback: MultiIndex on integer codes remains much
            # cheaper than refactorizing arbitrary original object columns.
            mi = pd.MultiIndex.from_arrays(code_arrays)
            out, uniques = pd.factorize(mi, sort=False, use_na_sentinel=False)
            return np.asarray(out, dtype=np.int32), len(uniques)
        key += codes.astype(np.uint64, copy=False) * np.uint64(stride)
        stride *= max(nlev, 1)
    out, uniques = pd.factorize(key, sort=False, use_na_sentinel=False)
    return np.asarray(out, dtype=np.int32), len(uniques)
