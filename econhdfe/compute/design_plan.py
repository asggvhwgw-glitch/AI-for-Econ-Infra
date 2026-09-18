from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _find(parent: np.ndarray, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


@njit(cache=True, nogil=True)
def _union(parent: np.ndarray, size: np.ndarray, a: int, b: int) -> None:
    ra = _find(parent, a)
    rb = _find(parent, b)
    if ra == rb:
        return
    if size[ra] < size[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    size[ra] += size[rb]


@njit(cache=True, nogil=True)
def _union_fe_pairs(
    parent: np.ndarray,
    size: np.ndarray,
    base: np.ndarray,
    other: np.ndarray,
    base_offset: int,
    other_offset: int,
) -> None:
    for i in range(base.size):
        _union(parent, size, base_offset + int(base[i]), other_offset + int(other[i]))


@njit(cache=True, nogil=True)
def _union_columns_to_fe(
    parent: np.ndarray,
    size: np.ndarray,
    X: np.ndarray,
    group: np.ndarray,
    group_offset: int,
    column_offset: int,
    zero_tol: float,
) -> np.ndarray:
    n, k = X.shape
    counts = np.zeros(k, dtype=np.int64)
    for i in range(n):
        fe_node = group_offset + int(group[i])
        for j in range(k):
            if abs(X[i, j]) > zero_tol:
                counts[j] += 1
                _union(parent, size, column_offset + j, fe_node)
    return counts


@njit(cache=True, nogil=True)
def _union_columns_by_row(
    parent: np.ndarray,
    size: np.ndarray,
    X: np.ndarray,
    column_offset: int,
    zero_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    n, k = X.shape
    counts = np.zeros(k, dtype=np.int64)
    first_col = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        first = -1
        for j in range(k):
            if abs(X[i, j]) > zero_tol:
                counts[j] += 1
                if first < 0:
                    first = j
                    first_col[i] = j
                else:
                    _union(parent, size, column_offset + first, column_offset + j)
    return counts, first_col


@njit(cache=True, nogil=True)
def _compress_all(parent: np.ndarray) -> None:
    for i in range(parent.size):
        parent[i] = _find(parent, i)


@dataclass(frozen=True, slots=True)
class DesignBlock:
    index: int
    columns: tuple[int, ...]
    nrows: int
    dense_bytes: int

    def as_dict(self) -> dict:
        return {
            "index": int(self.index),
            "columns": self.columns,
            "ncols": len(self.columns),
            "nrows": int(self.nrows),
            "dense_bytes": int(self.dense_bytes),
        }


@dataclass(slots=True)
class StructuralDesignPlan:
    """Exact connectivity description of a numerical design and categorical FE graph.

    Nodes are regressor columns and observed FE levels.  Every observation joins
    the FE levels that co-occur on that row; a nonzero regressor additionally
    joins its column node to that row's FE component.  Distinct connected
    components therefore provide an exact certificate that estimation can be
    decomposed by component.  No probabilistic sampling or tolerance-based
    topology inference is used.

    ``row_blocks`` is an execution aid rather than public statistical state.
    ``-1`` denotes a nuisance-only component with no active regressor column;
    such a design is deliberately *not* certified for block execution yet.
    """

    nobs: int
    ncols: int
    column_nnz: np.ndarray
    blocks: tuple[DesignBlock, ...]
    row_blocks: np.ndarray
    certified_block_separable: bool
    reason: str
    dense_bytes: int
    block_dense_bytes: int
    structural_zero_fraction: float

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def effective_max_width(self) -> int:
        return max((len(b.columns) for b in self.blocks), default=0)

    @property
    def block_dense_savings_fraction(self) -> float:
        if self.dense_bytes <= 0:
            return 0.0
        return max(0.0, 1.0 - self.block_dense_bytes / self.dense_bytes)

    def as_dict(self) -> dict:
        return {
            "nobs": int(self.nobs),
            "ncols": int(self.ncols),
            "nnz": int(np.sum(self.column_nnz)),
            "structural_zero_fraction": float(self.structural_zero_fraction),
            "block_count": int(self.block_count),
            "effective_max_width": int(self.effective_max_width),
            "certified_block_separable": bool(self.certified_block_separable),
            "reason": self.reason,
            "dense_bytes": int(self.dense_bytes),
            "block_dense_bytes": int(self.block_dense_bytes),
            "block_dense_savings_fraction": float(self.block_dense_savings_fraction),
            "blocks": tuple(b.as_dict() for b in self.blocks),
        }


def _normalize_groups(groups, nobs: int) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    dense = []
    levels = []
    for g in groups or ():
        a = np.asarray(g)
        if a.ndim != 1 or len(a) != nobs:
            raise ValueError("every fixed-effect code array must have one value per observation")
        if not np.issubdtype(a.dtype, np.integer):
            raise TypeError("structural design analysis requires dense integer FE codes")
        if a.size:
            amin = int(a.min()); amax = int(a.max())
            if amin < 0:
                raise ValueError("fixed-effect codes must be nonnegative")
            # A genuinely dense code vector over n observations cannot contain a
            # level index >= n. Reject this before max(code)+1 can trigger a huge
            # union-find allocation on malformed sparse/raw identifiers.
            if amax >= max(int(nobs), 1):
                raise ValueError("fixed-effect codes must be densely factorized before structural analysis")
            L = amax + 1
        else:
            L = 0
        dtype = np.int32 if L <= np.iinfo(np.int32).max else np.int64
        dense.append(np.asarray(a, dtype=dtype))
        levels.append(L)
    return tuple(dense), tuple(levels)


def analyze_design_structure(X, *, groups=(), zero_tol: float = 0.0) -> StructuralDesignPlan:
    """Analyze exact block structure without changing estimator semantics.

    This is deliberately estimator-agnostic.  It accepts an already numerical
    design plus dense categorical FE codes and returns a connectivity
    certificate that later execution planners may consume.  ``zero_tol``
    defaults to exact structural zero detection; callers should not increase it
    when an exact decomposition certificate is required.
    """
    if not np.isfinite(float(zero_tol)) or float(zero_tol) < 0.0:
        raise ValueError("zero_tol must be a finite nonnegative number")
    A = np.asarray(X, dtype=np.float64)
    if A.ndim == 1:
        A = A[:, None]
    if A.ndim != 2:
        raise ValueError("X must be a 2D numerical design")
    n, k = map(int, A.shape)
    if not np.all(np.isfinite(A)):
        raise ValueError("X must contain only finite values")
    groups_t, levels = _normalize_groups(groups, n)
    offsets = []
    cursor = 0
    for L in levels:
        offsets.append(cursor)
        cursor += int(L)
    column_offset = cursor
    total_nodes = cursor + k
    parent = np.arange(total_nodes, dtype=np.int64)
    size = np.ones(total_nodes, dtype=np.int64)

    if groups_t:
        base = groups_t[0]
        base_offset = offsets[0]
        for j in range(1, len(groups_t)):
            _union_fe_pairs(parent, size, base, groups_t[j], base_offset, offsets[j])
        counts = _union_columns_to_fe(
            parent, size, A, base, base_offset, column_offset, float(zero_tol)
        )
        first_col = None
    else:
        counts, first_col = _union_columns_by_row(
            parent, size, A, column_offset, float(zero_tol)
        )

    _compress_all(parent)
    if k:
        col_roots = parent[column_offset:column_offset + k]
        unique_roots = np.unique(col_roots)
    else:
        col_roots = np.empty(0, dtype=np.int64)
        unique_roots = np.empty(0, dtype=np.int64)
    root_to_block = {int(root): i for i, root in enumerate(unique_roots.tolist())}

    if groups_t:
        base_nodes = offsets[0] + groups_t[0].astype(np.int64, copy=False)
        row_roots = parent[base_nodes]
        row_blocks = np.fromiter(
            (root_to_block.get(int(r), -1) for r in row_roots),
            dtype=np.int32,
            count=n,
        )
    else:
        row_blocks = np.full(n, -1, dtype=np.int32)
        if first_col is not None:
            active = first_col >= 0
            if np.any(active):
                roots = col_roots[first_col[active]]
                row_blocks[active] = np.fromiter(
                    (root_to_block[int(r)] for r in roots),
                    dtype=np.int32,
                    count=int(np.sum(active)),
                )

    blocks = []
    block_dense_bytes = 0
    itemsize = int(A.dtype.itemsize)
    for b, root in enumerate(unique_roots.tolist()):
        cols = tuple(np.flatnonzero(col_roots == root).astype(int).tolist())
        nr = int(np.sum(row_blocks == b))
        payload = nr * len(cols) * itemsize
        block_dense_bytes += payload
        blocks.append(DesignBlock(b, cols, nr, payload))

    dense_bytes = n * k * itemsize
    nnz = int(np.sum(counts))
    zero_fraction = 0.0 if n * k == 0 else 1.0 - nnz / float(n * k)
    nuisance_rows = bool(np.any(row_blocks < 0))
    if k == 0:
        certified = False
        reason = "no_regressor_columns"
    elif nuisance_rows:
        certified = False
        reason = "nuisance_only_component"
    elif len(blocks) <= 1:
        certified = False
        reason = "single_connected_component"
    else:
        certified = True
        reason = "exact_disconnected_components"

    return StructuralDesignPlan(
        nobs=n,
        ncols=k,
        column_nnz=np.asarray(counts, dtype=np.int64),
        blocks=tuple(blocks),
        row_blocks=row_blocks,
        certified_block_separable=certified,
        reason=reason,
        dense_bytes=int(dense_bytes),
        block_dense_bytes=int(block_dense_bytes),
        structural_zero_fraction=float(zero_fraction),
    )


def _active_term_columns(terms):
    """Map each structural term level to the final global column index.

    ``StructuralTerm.active`` already reflects user omissions and exact
    pre-materialization structural-collinearity reductions.  Keeping this map
    compact avoids an N x K support mask: each observation contributes at most
    one active column per term.
    """
    maps = []
    cursor = 0
    for term in terms:
        active = np.asarray(term.active, dtype=bool)
        level_to_col = np.full(len(active), -1, dtype=np.int64)
        levels = np.flatnonzero(active)
        if levels.size:
            level_to_col[levels] = np.arange(cursor, cursor + len(levels), dtype=np.int64)
            cursor += len(levels)
        maps.append(level_to_col)
    return tuple(maps), int(cursor)


def _validated_term_codes(term, mapping: np.ndarray, nobs: int) -> np.ndarray:
    codes = np.asarray(term.codes)
    if codes.ndim != 1 or len(codes) != int(nobs):
        raise ValueError("all structural terms must have one code per observation")
    if not np.issubdtype(codes.dtype, np.integer):
        raise TypeError("structural term codes must be integer arrays")
    if codes.size:
        lo = int(codes.min()); hi = int(codes.max())
        if lo < 0 or hi >= len(mapping):
            raise ValueError("structural term code is outside the term level range")
    dtype = np.int32 if len(mapping) <= np.iinfo(np.int32).max else np.int64
    return np.asarray(codes, dtype=dtype)


@njit(cache=True, nogil=True)
def _union_one_symbolic_term_to_fe(
    parent: np.ndarray,
    size: np.ndarray,
    codes: np.ndarray,
    mapping: np.ndarray,
    base: np.ndarray,
    base_offset: int,
    column_offset: int,
    counts: np.ndarray,
) -> None:
    # One-term kernel keeps the Numba signature independent of the number of
    # factor-variable terms.  A tuple-of-arrays kernel would otherwise compile
    # a new specialization for every term count seen in an empirical model.
    for i in range(base.size):
        col = int(mapping[int(codes[i])])
        if col >= 0:
            counts[col] += 1
            _union(parent, size, column_offset + col, base_offset + int(base[i]))


@njit(cache=True, nogil=True)
def _union_one_symbolic_term_by_row(
    parent: np.ndarray,
    size: np.ndarray,
    codes: np.ndarray,
    mapping: np.ndarray,
    column_offset: int,
    counts: np.ndarray,
    first_col: np.ndarray,
) -> None:
    for i in range(codes.size):
        col = int(mapping[int(codes[i])])
        if col < 0:
            continue
        counts[col] += 1
        first = int(first_col[i])
        if first < 0:
            first_col[i] = col
        else:
            _union(parent, size, column_offset + first, column_offset + col)


def analyze_structural_terms(terms, *, groups=(), itemsize: int = 8) -> StructuralDesignPlan:
    """Certify block structure from compact term support before X exists.

    Each expanded factor/interaction column has structural support
    ``term.codes == level``.  A purely continuous term is represented by the
    single level whose code is zero on every row, so it correctly acts as a
    global-support column.  Continuous values are deliberately *not* inspected:
    an accidental numerical zero can only make this certificate conservative,
    never create a false disconnected component.

    This is the pre-materialization analogue of :func:`analyze_design_structure`.
    Its cost scales with observations times structural terms rather than with
    observations times expanded columns.
    """
    terms = tuple(terms or ())
    if terms:
        n = int(len(np.asarray(terms[0].codes)))
        for term in terms:
            if len(np.asarray(term.codes)) != n:
                raise ValueError("all structural terms must have one code per observation")
    else:
        groups0 = tuple(groups or ())
        n = int(len(np.asarray(groups0[0]))) if groups0 else 0

    groups_t, levels = _normalize_groups(groups, n)
    mappings, k = _active_term_columns(terms)
    term_codes = tuple(
        _validated_term_codes(term, mapping, n)
        for term, mapping in zip(terms, mappings, strict=False)
    )

    offsets = []
    cursor = 0
    for L in levels:
        offsets.append(cursor)
        cursor += int(L)
    column_offset = cursor
    total_nodes = cursor + k
    parent = np.arange(total_nodes, dtype=np.int64)
    size = np.ones(total_nodes, dtype=np.int64)

    if groups_t:
        base = groups_t[0]
        base_offset = offsets[0]
        for j in range(1, len(groups_t)):
            _union_fe_pairs(parent, size, base, groups_t[j], base_offset, offsets[j])
        counts = np.zeros(k, dtype=np.int64)
        for codes, mapping in zip(term_codes, mappings, strict=False):
            _union_one_symbolic_term_to_fe(
                parent, size, codes, mapping, base, base_offset, column_offset, counts
            )
        first_col = None
    else:
        counts = np.zeros(k, dtype=np.int64)
        first_col = np.full(n, -1, dtype=np.int64)
        for codes, mapping in zip(term_codes, mappings, strict=False):
            _union_one_symbolic_term_by_row(
                parent, size, codes, mapping, column_offset, counts, first_col
            )

    _compress_all(parent)
    if k:
        col_roots = parent[column_offset:column_offset + k]
        unique_roots = np.unique(col_roots)
    else:
        col_roots = np.empty(0, dtype=np.int64)
        unique_roots = np.empty(0, dtype=np.int64)
    root_to_block = {int(root): i for i, root in enumerate(unique_roots.tolist())}

    if groups_t:
        base_nodes = offsets[0] + groups_t[0].astype(np.int64, copy=False)
        row_roots = parent[base_nodes]
        row_blocks = np.fromiter(
            (root_to_block.get(int(r), -1) for r in row_roots),
            dtype=np.int32, count=n,
        )
    else:
        row_blocks = np.full(n, -1, dtype=np.int32)
        if first_col is not None:
            active_rows = first_col >= 0
            if np.any(active_rows):
                roots = col_roots[first_col[active_rows]]
                row_blocks[active_rows] = np.fromiter(
                    (root_to_block[int(r)] for r in roots),
                    dtype=np.int32, count=int(np.sum(active_rows)),
                )

    blocks = []
    block_dense_bytes = 0
    itemsize = int(itemsize)
    for b, root in enumerate(unique_roots.tolist()):
        cols = tuple(np.flatnonzero(col_roots == root).astype(int).tolist())
        nr = int(np.sum(row_blocks == b))
        payload = nr * len(cols) * itemsize
        block_dense_bytes += payload
        blocks.append(DesignBlock(b, cols, nr, payload))

    dense_bytes = n * k * itemsize
    nnz = int(np.sum(counts))
    zero_fraction = 0.0 if n * k == 0 else 1.0 - nnz / float(n * k)
    nuisance_rows = bool(np.any(row_blocks < 0))
    if k == 0:
        certified = False; reason = "no_regressor_columns"
    elif nuisance_rows:
        certified = False; reason = "nuisance_only_component"
    elif len(blocks) <= 1:
        certified = False; reason = "single_connected_component"
    else:
        certified = True; reason = "exact_symbolic_disconnected_components"

    return StructuralDesignPlan(
        nobs=n, ncols=k, column_nnz=np.asarray(counts, dtype=np.int64),
        blocks=tuple(blocks), row_blocks=row_blocks,
        certified_block_separable=certified, reason=reason,
        dense_bytes=int(dense_bytes), block_dense_bytes=int(block_dense_bytes),
        structural_zero_fraction=float(zero_fraction),
    )


@dataclass(slots=True)
class PartitionedDesignPlan:
    """Exact FE-row partition with component-local structural column support.

    Unlike ``StructuralDesignPlan``, columns are allowed to appear in multiple
    row components.  This captures the common empirical pattern of many
    block-local interactions plus a small set of coefficients shared across
    blocks.  The row partition is certified solely by FE topology, while term
    metadata certifies which columns can be nonzero inside each row component.
    """

    nobs: int
    ncols: int
    column_nnz: np.ndarray
    blocks: tuple[DesignBlock, ...]
    row_blocks: np.ndarray
    column_block_counts: np.ndarray
    dense_bytes: int
    block_dense_bytes: int
    structural_zero_fraction: float
    reason: str

    @property
    def block_count(self) -> int:
        return len(self.blocks)

    @property
    def columns_disjoint(self) -> bool:
        return bool(np.all(self.column_block_counts <= 1))

    @property
    def shared_columns(self) -> tuple[int, ...]:
        return tuple(np.flatnonzero(self.column_block_counts > 1).astype(int).tolist())

    @property
    def certified_partitioned(self) -> bool:
        return bool(self.ncols > 0 and self.block_count > 0 and not np.any(self.row_blocks < 0))

    # Compatibility with the representation planner/projector contract.  Here
    # "block separable" means storage/projection separable by rows; coefficient
    # solves may still be coupled through shared columns.
    @property
    def certified_block_separable(self) -> bool:
        return self.certified_partitioned

    @property
    def effective_max_width(self) -> int:
        return max((len(b.columns) for b in self.blocks), default=0)

    @property
    def block_dense_savings_fraction(self) -> float:
        if self.dense_bytes <= 0:
            return 0.0
        return max(0.0, 1.0 - self.block_dense_bytes / self.dense_bytes)

    def as_dict(self) -> dict:
        return {
            "nobs": int(self.nobs), "ncols": int(self.ncols),
            "nnz": int(np.sum(self.column_nnz)),
            "block_count": int(self.block_count),
            "effective_max_width": int(self.effective_max_width),
            "columns_disjoint": bool(self.columns_disjoint),
            "shared_columns": self.shared_columns,
            "dense_bytes": int(self.dense_bytes),
            "block_dense_bytes": int(self.block_dense_bytes),
            "block_dense_savings_fraction": float(self.block_dense_savings_fraction),
            "structural_zero_fraction": float(self.structural_zero_fraction),
            "reason": self.reason,
            "blocks": tuple(b.as_dict() for b in self.blocks),
        }


def _fe_row_partition(groups, nobs: int) -> np.ndarray:
    groups_t, levels = _normalize_groups(groups, nobs)
    if not groups_t:
        return np.zeros(nobs, dtype=np.int32)
    offsets = []
    cursor = 0
    for L in levels:
        offsets.append(cursor); cursor += int(L)
    parent = np.arange(cursor, dtype=np.int64)
    size = np.ones(cursor, dtype=np.int64)
    base = groups_t[0]
    for j in range(1, len(groups_t)):
        _union_fe_pairs(parent, size, base, groups_t[j], offsets[0], offsets[j])
    _compress_all(parent)
    roots = parent[offsets[0] + base.astype(np.int64, copy=False)]
    unique = np.unique(roots)
    root_to_block = {int(root): i for i, root in enumerate(unique.tolist())}
    return np.fromiter(
        (root_to_block[int(root)] for root in roots), dtype=np.int32, count=nobs
    )


def analyze_structural_partition(terms, *, groups=(), itemsize: int = 8) -> PartitionedDesignPlan:
    """Build an exact row-partitioned design plan before numeric X exists.

    FE connectivity determines disjoint observation components.  Compact
    structural-term codes then determine which global coefficient columns can
    be active inside each component.  A coefficient may appear in several
    components; such shared columns couple only the small coefficient solve,
    not FE projection or physical X storage.
    """
    terms = tuple(terms or ())
    if terms:
        n = int(len(np.asarray(terms[0].codes)))
        for term in terms:
            if len(np.asarray(term.codes)) != n:
                raise ValueError("all structural terms must have one code per observation")
    else:
        groups0 = tuple(groups or ())
        n = int(len(np.asarray(groups0[0]))) if groups0 else 0

    mappings, k = _active_term_columns(terms)
    row_blocks = _fe_row_partition(groups, n)
    B = int(row_blocks.max()) + 1 if row_blocks.size else 0
    block_columns = [set() for _ in range(B)]
    column_nnz = np.zeros(k, dtype=np.int64)
    column_block_counts = np.zeros(k, dtype=np.int64)

    for term, mapping in zip(terms, mappings, strict=False):
        codes = _validated_term_codes(term, mapping, n)
        mapped = mapping[codes]
        active = mapped >= 0
        if not np.any(active):
            continue
        cols = mapped[active].astype(np.int64, copy=False)
        column_nnz += np.bincount(cols, minlength=k)
        keys = row_blocks[active].astype(np.int64, copy=False) * max(k, 1) + cols
        for key in np.unique(keys).tolist():
            b = int(key // max(k, 1)); col = int(key % max(k, 1))
            block_columns[b].add(col)

    blocks = []
    block_dense_bytes = 0
    itemsize = int(itemsize)
    for b in range(B):
        cols = tuple(sorted(block_columns[b]))
        nr = int(np.count_nonzero(row_blocks == b))
        payload = nr * len(cols) * itemsize
        block_dense_bytes += payload
        blocks.append(DesignBlock(b, cols, nr, payload))
        for col in cols:
            column_block_counts[col] += 1

    dense_bytes = int(n * k * itemsize)
    nnz = int(np.sum(column_nnz))
    zero_fraction = 0.0 if n * k == 0 else 1.0 - nnz / float(n * k)
    if k == 0:
        reason = "no_regressor_columns"
    elif B <= 1:
        reason = "single_fe_row_component"
    elif np.all(column_block_counts <= 1):
        reason = "exact_fe_partition_disjoint_columns"
    else:
        reason = "exact_fe_partition_shared_columns"
    return PartitionedDesignPlan(
        nobs=n, ncols=k, column_nnz=column_nnz, blocks=tuple(blocks),
        row_blocks=row_blocks, column_block_counts=column_block_counts,
        dense_bytes=dense_bytes, block_dense_bytes=int(block_dense_bytes),
        structural_zero_fraction=float(zero_fraction), reason=reason,
    )


def analyze_execution_structure(terms, *, groups=(), itemsize: int = 8):
    """Choose the strongest exact pre-materialization execution certificate.

    Joint coefficient/FE connectivity is preferred when it yields truly
    disconnected blocks.  If shared coefficients collapse that graph, an exact
    FE-row partition may still preserve local projection/storage while leaving
    the small coefficient solve globally coupled.  The function only chooses
    between exact certificates; it never uses numerical-zero heuristics.
    """
    exact = analyze_structural_terms(terms, groups=groups, itemsize=itemsize)
    if exact.certified_block_separable:
        return exact
    partitioned = analyze_structural_partition(terms, groups=groups, itemsize=itemsize)
    if partitioned.block_count > 1 and partitioned.certified_partitioned:
        return partitioned
    return exact
