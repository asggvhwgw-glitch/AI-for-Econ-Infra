from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _is_refinement_dense(fine: np.ndarray, coarse: np.ndarray, n_fine: int) -> bool:
    """True iff each observed ``fine`` level belongs to one ``coarse`` level."""
    mapping = np.full(n_fine, -1, dtype=np.int64)
    for i in range(fine.size):
        f = int(fine[i])
        c = int(coarse[i])
        old = mapping[f]
        if old == -1:
            mapping[f] = c
        elif old != c:
            return False
    return True


@njit(cache=True, nogil=True)
def _is_refinement_sampled(
    fine: np.ndarray, coarse: np.ndarray, sample_idx: np.ndarray, n_fine: int
) -> bool:
    """Cheap rejection-only refinement check on a deterministic row sample."""
    mapping = np.full(n_fine, -1, dtype=np.int64)
    for k in range(sample_idx.size):
        i = int(sample_idx[k])
        f = int(fine[i])
        c = int(coarse[i])
        old = mapping[f]
        if old == -1:
            mapping[f] = c
        elif old != c:
            return False
    return True


@dataclass(frozen=True, slots=True)
class FEPartitionMeta:
    """Structural metadata for one absorbed FE partition.

    ``components`` contains exact categorical source tokens when they are known
    from the specification.  For example ``city#year`` has components
    ``("city", "year")``.  Opaque pre-computed arrays/columns can leave it
    empty; exact data certification still detects their nesting relationships.
    """

    name: str
    components: tuple[str, ...] = ()
    pure_intercept: bool = True


@dataclass(frozen=True, slots=True)
class RefinementCertificate:
    """Exact certificate that ``coarse`` is spanned by ``fine``."""

    coarse_index: int
    coarse: str
    fine_index: int
    fine: str
    proof_type: str
    exact: bool = True

    def as_dict(self) -> dict:
        return {
            "coarse": self.coarse,
            "fine": self.fine,
            "proof_type": self.proof_type,
            "exact": bool(self.exact),
        }


@dataclass(frozen=True, slots=True)
class EquivalentPartition:
    left_index: int
    left: str
    right_index: int
    right: str
    proof_type: str

    def as_dict(self) -> dict:
        return {
            "left": self.left,
            "right": self.right,
            "proof_type": self.proof_type,
        }


@dataclass(frozen=True, slots=True)
class RedundantFixedEffect:
    index: int
    name: str
    spanned_by_index: int
    spanned_by: str
    reason: str = "nested_intercept"
    proof_type: str = "exact_mapping"


@dataclass(frozen=True, slots=True)
class FECanonicalization:
    requested_names: tuple[str, ...]
    effective_names: tuple[str, ...]
    effective_indices: tuple[int, ...]
    dropped: tuple[RedundantFixedEffect, ...]
    requested_levels: tuple[int, ...] = ()
    effective_levels: tuple[int, ...] = ()
    certificates: tuple[RefinementCertificate, ...] = ()
    equivalences: tuple[EquivalentPartition, ...] = ()
    passes: int = 1
    sample_size: int = 0
    candidate_pairs: int = 0
    specification_proofs: int = 0
    sample_rejections: int = 0
    full_data_checks: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.dropped)

    def as_dict(self) -> dict:
        return {
            "requested": self.requested_names,
            "effective": self.effective_names,
            "requested_levels": self.requested_levels,
            "effective_levels": self.effective_levels,
            "dropped": tuple({
                "name": d.name,
                "spanned_by": d.spanned_by,
                "reason": d.reason,
                "proof_type": d.proof_type,
            } for d in self.dropped),
            "refinement_edges": tuple(c.as_dict() for c in self.certificates),
            "equivalences": tuple(e.as_dict() for e in self.equivalences),
            "passes": int(self.passes),
            "diagnostics": {
                "sample_size": int(self.sample_size),
                "candidate_pairs": int(self.candidate_pairs),
                "specification_proofs": int(self.specification_proofs),
                "sample_rejections": int(self.sample_rejections),
                "full_data_checks": int(self.full_data_checks),
            },
        }


def _levels_dense(g) -> int:
    g = np.asarray(g)
    return int(g.max()) + 1 if g.size else 0


def _sample_indices(n: int, sample_size: int) -> np.ndarray:
    """Deterministic spread-out sample used only to reject false candidates."""
    m = min(max(int(sample_size), 0), int(n))
    if m == 0 or m == n:
        return np.arange(n, dtype=np.int64)
    # linspace can duplicate integer endpoints after casting for tiny n; unique
    # keeps the rejection sample exact and deterministic.
    return np.unique(np.linspace(0, n - 1, m, dtype=np.int64))


def _component_proves_refinement(coarse_meta, fine_meta) -> bool:
    """Specification-only proof for categorical interaction containment."""
    if coarse_meta is None or fine_meta is None:
        return False
    coarse = frozenset(coarse_meta.components)
    fine = frozenset(fine_meta.components)
    return bool(coarse) and coarse.issubset(fine)


def _ultimate_parent(i: int, parent: dict[int, int]) -> int:
    seen = set()
    j = i
    while j in parent and j not in seen:
        seen.add(j)
        j = parent[j]
    return j


def canonicalize_fixed_effects(
    groups,
    slopes,
    intercepts,
    names,
    *,
    metadata=None,
    sample_size: int = 32_768,
    passes: int = 1,
):
    """Canonicalize categorical FE partitions using exact refinement proofs.

    For pure categorical intercept FE ``A`` and an intercept-containing FE ``B``,
    ``A`` can be removed from numerical absorption when every level of ``B``
    maps to exactly one level of ``A``.  Equivalently, the partition induced by
    ``B`` refines ``A`` and ``col(D_A)`` is contained in ``col(D_B)``.

    The implementation deliberately accepts only exact certificates:

    1. specification containment (e.g. ``year`` in ``city#year``), or
    2. a full-data functional-dependency scan after a cheap rejection sample.

    This handles complex empirical FE structures such as
    ``province#year <= city#year`` whenever the estimation sample proves
    ``city -> province``.  Heterogeneous-slope terms are never removed as a
    whole, because retaining their intercept can materially improve numerical
    conditioning in reghdfe-style algorithms.
    """
    groups = [np.asarray(g, dtype=np.int32) for g in groups]
    slopes = list(slopes)
    intercepts = list(intercepts)
    names = list(map(str, names))
    G = len(groups)
    if not (len(slopes) == len(intercepts) == len(names) == G):
        raise ValueError("fixed-effect metadata length mismatch")
    if G and any(len(g) != len(groups[0]) for g in groups):
        raise ValueError("fixed-effect groups must have equal length")

    if metadata is None:
        metadata = [FEPartitionMeta(names[i], (), bool(intercepts[i] and slopes[i] is None)) for i in range(G)]
    else:
        metadata = list(metadata)
        if len(metadata) != G:
            raise ValueError("partition metadata length mismatch")

    levels = [_levels_dense(g) for g in groups]
    n = len(groups[0]) if G else 0
    sample_idx = _sample_indices(n, sample_size)

    # relation[(coarse, fine)] = proof type.  Only relations sufficient to
    # remove a *pure intercept* coarse FE are certified here.
    relation: dict[tuple[int, int], str] = {}
    candidate_pairs = specification_proofs = sample_rejections = full_data_checks = 0

    for coarse in range(G):
        if not intercepts[coarse] or slopes[coarse] is not None:
            continue
        for fine in range(G):
            if fine == coarse or not intercepts[fine]:
                continue
            # A refinement cannot have fewer observed cells than the coarse FE.
            if levels[fine] < levels[coarse]:
                continue
            candidate_pairs += 1

            if _component_proves_refinement(metadata[coarse], metadata[fine]):
                relation[(coarse, fine)] = "component_containment"
                specification_proofs += 1
                continue

            # Sample checks are rejection-only.  Surviving candidates must pass
            # the exact full-data mapping test before an FE can be dropped.
            if sample_idx.size < n and not _is_refinement_sampled(
                groups[fine], groups[coarse], sample_idx, levels[fine]
            ):
                sample_rejections += 1
                continue

            full_data_checks += 1
            if _is_refinement_dense(groups[fine], groups[coarse], levels[fine]):
                relation[(coarse, fine)] = "exact_mapping"

    certificates = tuple(
        RefinementCertificate(i, names[i], j, names[j], proof)
        for (i, j), proof in sorted(relation.items())
    )

    # Equal-cardinality one-way refinement is an observed-sample equivalence.
    equivalences = []
    for (i, j), proof in sorted(relation.items()):
        if i < j and levels[i] == levels[j] and (j, i) in relation:
            p2 = relation[(j, i)]
            p = proof if proof == p2 else f"{proof}+{p2}"
            equivalences.append(EquivalentPartition(i, names[i], j, names[j], p))

    # Choose one spanning parent for every removable FE.  Prefer a strictly
    # finer partition; among ties choose the earlier requested term.  This makes
    # normalization deterministic while keeping maximal partitions.
    parent: dict[int, int] = {}
    parent_proof: dict[int, str] = {}
    for coarse in range(G):
        if not intercepts[coarse] or slopes[coarse] is not None:
            continue
        candidates = [j for (i, j) in relation if i == coarse]
        eligible = [
            j for j in candidates
            if levels[j] > levels[coarse] or (levels[j] == levels[coarse] and j < coarse)
        ]
        if not eligible:
            continue
        best = max(eligible, key=lambda j: (levels[j], -j))
        parent[coarse] = best
        parent_proof[coarse] = relation[(coarse, best)]

    # Point dropped terms at an ultimately retained maximal partition when a
    # chain such as year <= province#year <= city#year is present.
    drop: dict[int, int] = {}
    for i in parent:
        drop[i] = _ultimate_parent(parent[i], parent)

    keep = tuple(i for i in range(G) if i not in drop)
    dropped = []
    for i in sorted(drop):
        j = drop[i]
        proof = relation.get((i, j), parent_proof.get(i, "transitive_refinement"))
        if (i, j) not in relation and j != parent.get(i):
            proof = "transitive_refinement"
        reason = "equivalent_intercept" if levels[i] == levels[j] else "nested_intercept"
        dropped.append(RedundantFixedEffect(i, names[i], j, names[j], reason, proof))

    plan = FECanonicalization(
        requested_names=tuple(names),
        effective_names=tuple(names[i] for i in keep),
        effective_indices=keep,
        dropped=tuple(dropped),
        requested_levels=tuple(map(int, levels)),
        effective_levels=tuple(int(levels[i]) for i in keep),
        certificates=certificates,
        equivalences=tuple(equivalences),
        passes=int(passes),
        sample_size=int(sample_idx.size),
        candidate_pairs=int(candidate_pairs),
        specification_proofs=int(specification_proofs),
        sample_rejections=int(sample_rejections),
        full_data_checks=int(full_data_checks),
    )
    return (
        [groups[i] for i in keep],
        [slopes[i] for i in keep],
        [intercepts[i] for i in keep],
        [names[i] for i in keep],
        plan,
    )


def identity_fixed_effect_plan(names, levels=()):
    names = tuple(map(str, names))
    levels = tuple(map(int, levels)) if levels else ()
    return FECanonicalization(names, names, tuple(range(len(names))), (), levels, levels)
