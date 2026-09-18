from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _peel_active_mask(codes2d, levels):
    """Return active rows after exact degree-one hypergraph peeling.

    Rows are hyperedges; every FE level is a vertex.  Whenever a vertex has
    degree one, orthogonality to that dummy forces the incident residual to
    zero, so the row can be removed exactly and the process repeated.
    """
    G, n = codes2d.shape
    offsets = np.empty(G + 1, dtype=np.int64)
    offsets[0] = 0
    for g in range(G):
        offsets[g + 1] = offsets[g] + levels[g]
    V = offsets[G]
    counts = np.zeros(V, dtype=np.int64)
    row_sum = np.zeros(V, dtype=np.int64)
    for gi in range(G):
        off = offsets[gi]
        for i in range(n):
            v = off + int(codes2d[gi, i])
            counts[v] += 1
            row_sum[v] += i

    queue = np.empty(V, dtype=np.int64)
    qh = 0
    qt = 0
    for v in range(V):
        if counts[v] == 1:
            queue[qt] = v
            qt += 1

    active = np.ones(n, dtype=np.uint8)
    peeled = 0
    while qh < qt:
        v = queue[qh]
        qh += 1
        if counts[v] != 1:
            continue
        i = row_sum[v]
        if i < 0 or i >= n or active[i] == 0:
            continue
        active[i] = 0
        peeled += 1
        for gi in range(G):
            vv = offsets[gi] + int(codes2d[gi, i])
            counts[vv] -= 1
            row_sum[vv] -= i
            if counts[vv] == 1:
                queue[qt] = vv
                qt += 1
    return active, peeled


def _dense_subset_codes(group: np.ndarray, idx: np.ndarray):
    """Dense-remap one already-dense FE code vector on a row subset in O(N+L)."""
    g = np.asarray(group, dtype=np.int32)
    sub = g[idx]
    if sub.size == 0:
        return sub
    L = int(g.max()) + 1
    present = np.zeros(L, dtype=np.uint8)
    present[sub] = 1
    mapping = np.empty(L, dtype=np.int32)
    nxt = 0
    for level in np.flatnonzero(present):
        mapping[level] = nxt
        nxt += 1
    return mapping[sub]


@dataclass(frozen=True, slots=True)
class NumericalCorePlan:
    nobs: int
    core_index: np.ndarray
    core_groups: tuple[np.ndarray, ...]
    peeled: int

    @property
    def core_nobs(self) -> int:
        return int(self.core_index.size)

    @property
    def peeled_fraction(self) -> float:
        return float(self.peeled / self.nobs) if self.nobs else 0.0

    @property
    def changed(self) -> bool:
        return self.peeled > 0


def build_numerical_core(groups) -> NumericalCorePlan:
    """Compile exact numerical core for pure intercept categorical FEs.

    Unlike rank preprocessing, duplicate observation tuples are *not*
    collapsed: each row carries its own RHS value and weight.
    """
    gs = tuple(np.asarray(g, dtype=np.int32) for g in groups)
    if not gs:
        return NumericalCorePlan(0, np.empty(0, dtype=np.int64), (), 0)
    n = len(gs[0])
    if any(len(g) != n for g in gs):
        raise ValueError("all FE groups must have equal length")
    levels = np.asarray([int(g.max()) + 1 if n else 0 for g in gs], dtype=np.int64)
    codes2d = np.ascontiguousarray(np.vstack(gs), dtype=np.int32)
    active, peeled = _peel_active_mask(codes2d, levels)
    idx = np.flatnonzero(active).astype(np.int64, copy=False)
    core_groups = tuple(_dense_subset_codes(g, idx) for g in gs)
    return NumericalCorePlan(n, idx, core_groups, int(peeled))
