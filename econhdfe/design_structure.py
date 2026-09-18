from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def _refinement_mapping(fine: np.ndarray, coarse: np.ndarray, n_fine: int):
    """Map fine levels to coarse levels, returning ``(-1,)`` on violation."""
    mapping = np.full(n_fine, -1, dtype=np.int64)
    for i in range(fine.size):
        f = int(fine[i]); c = int(coarse[i])
        old = mapping[f]
        if old == -1:
            mapping[f] = c
        elif old != c:
            return np.empty(0, dtype=np.int64)
    return mapping


@dataclass(slots=True)
class StructuralTerm:
    """Compact column-space description for one expanded regressor term.

    A term represents columns ``1[group == g] * m`` for active partition
    levels ``g``, where ``m`` is a continuous monomial shared by the block.
    This representation is enough to prove many exact factor/interaction
    dependencies without materializing the N x K dummy block.
    """

    index: int
    name: str
    codes: np.ndarray
    n_levels: int
    active: np.ndarray
    level_names: tuple[str, ...]
    continuous_signature: tuple[str, ...]
    kind: str
    categorical_components: tuple[str, ...] = ()
    component_codes: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True, slots=True)
class StructuralOmission:
    term_index: int
    term: str
    level: int
    name: str
    reason: str
    dependent_on: tuple[str, ...]
    proof_type: str

    def as_dict(self) -> dict:
        return {
            "term_index": int(self.term_index),
            "term": self.term,
            "level": int(self.level),
            "name": self.name,
            "reason": self.reason,
            "dependent_on": self.dependent_on,
            "proof_type": self.proof_type,
        }


@dataclass(frozen=True, slots=True)
class StructuralDependency:
    coarse: str
    fine: str
    continuous_signature: tuple[str, ...]
    proof_type: str
    action: str
    columns_removed: int

    def as_dict(self) -> dict:
        return {
            "coarse": self.coarse,
            "fine": self.fine,
            "continuous_signature": self.continuous_signature,
            "proof_type": self.proof_type,
            "action": self.action,
            "columns_removed": int(self.columns_removed),
        }


@dataclass(frozen=True, slots=True)
class StructuralCollinearityPlan:
    omissions: tuple[StructuralOmission, ...]
    dependencies: tuple[StructuralDependency, ...]
    requested_columns: int
    materialized_columns: int
    absorbed_checks: int
    term_checks: int
    component_dependencies: tuple[StructuralDependency, ...] = ()
    closure_dependencies: tuple[StructuralDependency, ...] = ()
    component_checks: int = 0
    closure_inferences: int = 0

    def as_dict(self) -> dict:
        return {
            "requested_columns": int(self.requested_columns),
            "materialized_columns": int(self.materialized_columns),
            "omitted": tuple(o.as_dict() for o in self.omissions),
            "dependencies": tuple(d.as_dict() for d in self.dependencies),
            "component_dependencies": tuple(d.as_dict() for d in self.component_dependencies),
            "closure_dependencies": tuple(d.as_dict() for d in self.closure_dependencies),
            "diagnostics": {
                "absorbed_checks": int(self.absorbed_checks),
                "term_checks": int(self.term_checks),
                "component_checks": int(self.component_checks),
                "closure_inferences": int(self.closure_inferences),
            },
        }


def _levels(codes: np.ndarray) -> int:
    return int(codes.max()) + 1 if codes.size else 0


def _coverage_spans_coarse(
    fine_codes: np.ndarray,
    coarse_codes: np.ndarray,
    fine_active: np.ndarray,
    coarse_active: np.ndarray,
):
    """Return active coarse levels completely covered by active fine levels.

    The result is exact.  If ``fine`` does not refine ``coarse`` an empty mask
    is returned together with ``None``.
    """
    n_fine = len(fine_active)
    mapping = _refinement_mapping(
        np.asarray(fine_codes, dtype=np.int32),
        np.asarray(coarse_codes, dtype=np.int32),
        n_fine,
    )
    if mapping.size == 0:
        return np.zeros(len(coarse_active), dtype=bool), None

    observed = np.bincount(np.asarray(fine_codes, dtype=np.int64), minlength=n_fine) > 0
    bad = observed & ~fine_active
    n_coarse = len(coarse_active)
    bad_by_coarse = np.zeros(n_coarse, dtype=bool)
    if np.any(bad):
        bad_coarse = mapping[np.flatnonzero(bad)]
        bad_coarse = bad_coarse[bad_coarse >= 0]
        if bad_coarse.size:
            bad_by_coarse[np.unique(bad_coarse)] = True
    covered = np.asarray(coarse_active, dtype=bool) & ~bad_by_coarse
    return covered, mapping


def _drop_one_fine_per_covered_coarse(
    term: StructuralTerm,
    coarse_codes: np.ndarray,
    coarse_active: np.ndarray,
    *,
    dependency_name: str,
    reason: str,
    proof_type: str,
):
    covered, mapping = _coverage_spans_coarse(
        term.codes, coarse_codes, term.active, coarse_active
    )
    if mapping is None or not np.any(covered):
        return [], 0

    # Deterministically remove the last active observed fine cell in each
    # covered coarse cell. Keeping the earlier cells preserves user/level order.
    observed = np.bincount(term.codes.astype(np.int64), minlength=term.n_levels) > 0
    candidates = np.flatnonzero(term.active & observed)
    chosen: dict[int, int] = {}
    for f in candidates:
        c = int(mapping[f])
        if c >= 0 and c < len(covered) and covered[c]:
            chosen[c] = int(f)

    out = []
    for c, f in sorted(chosen.items()):
        term.active[f] = False
        out.append(StructuralOmission(
            term.index, term.name, f, term.level_names[f], reason,
            (dependency_name,), proof_type,
        ))
    return out, len(out)



def _deterministic_sample(n: int, size: int = 16_384) -> np.ndarray:
    m = min(int(n), int(size))
    if m <= 0 or m == n:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, m, dtype=np.int64))


def _component_dependency_closure(terms, protected_terms=()):
    """Build an exact refinement preorder with incremental transitive closure.

    Bijective partitions can create directed cycles; only the quotient by
    equivalence classes is a DAG. Edges are (coarse, fine), the reverse of
    the conventional functional-dependency arrow fine -> coarse.

    Components are ordered by cardinality. For each fine partition, the closest
    coarser candidates are certified first; once ``A <= B`` and ``B <= C`` are
    known, ``A <= C`` is inferred transitively and the expensive N-row mapping
    check for that pair is skipped. Sampling is rejection-only: every direct
    data-derived edge still receives a full exact mapping certificate.
    """
    registry = {}
    for term in tuple(protected_terms) + tuple(terms):
        for token, codes in zip(term.categorical_components, term.component_codes, strict=False):
            a = np.asarray(codes, dtype=np.int32)
            if token not in registry:
                registry[token] = a
    tokens = tuple(registry)
    levels = {t: _levels(registry[t]) for t in tokens}
    direct: dict[tuple[str, str], str] = {}
    reach: set[tuple[str, str]] = set()
    inferred: set[tuple[str, str]] = set()
    full_checks = 0

    if tokens:
        n = len(registry[tokens[0]])
        sample = _deterministic_sample(n)
        order = {t: i for i, t in enumerate(tokens)}
        fine_order = sorted(tokens, key=lambda t: (levels[t], order[t]))

        def add_edge(coarse: str, fine: str, proof: str):
            nonlocal reach, inferred
            if (coarse, fine) in direct:
                return
            direct[(coarse, fine)] = proof
            old = set(reach)
            reach.add((coarse, fine))
            # Incremental transitive closure through the newly added edge.
            ancestors = {coarse} | {a for (a, b) in reach if b == coarse}
            descendants = {fine} | {b for (a, b) in reach if a == fine}
            for a in ancestors:
                for b in descendants:
                    if a != b and (a, b) not in reach:
                        reach.add((a, b))
            inferred.update(reach - old - set(direct))

        for fine in fine_order:
            candidates = [
                coarse for coarse in tokens
                if coarse != fine and levels[coarse] <= levels[fine]
            ]
            # Nearest coarser partitions first, so chains can short-circuit
            # broader relations (city -> province -> region, etc.).
            candidates.sort(key=lambda t: (-levels[t], order[t]))
            for coarse in candidates:
                if (coarse, fine) in reach:
                    inferred.add((coarse, fine))
                    continue
                fc, cc = registry[fine], registry[coarse]
                if sample.size < n:
                    m = _refinement_mapping(fc[sample], cc[sample], levels[fine])
                    if m.size == 0:
                        continue
                full_checks += 1
                if _refinement_mapping(fc, cc, levels[fine]).size:
                    add_edge(coarse, fine, "exact_component_mapping")
                    # Equal-cardinality refinement is a bijection on observed
                    # levels, so the reverse relation is exact without a second
                    # full-data scan.
                    if levels[coarse] == levels[fine]:
                        add_edge(fine, coarse, "exact_component_bijection")

    direct_meta = tuple(
        StructuralDependency(a, b, (), proof, "component_refinement", 0)
        for (a, b), proof in sorted(direct.items())
    )
    closure_only = sorted(reach - set(direct))
    closure_meta = tuple(
        StructuralDependency(a, b, (), "transitive_component_refinement", "component_refinement", 0)
        for a, b in closure_only
    )
    return reach, direct_meta, closure_meta, full_checks, len(closure_only)


def _component_graph_proves_refinement(coarse: StructuralTerm, fine: StructuralTerm, reach) -> bool:
    """Sufficient exact proof that ``fine`` refines ``coarse`` by components."""
    if not coarse.categorical_components or not fine.categorical_components:
        return False
    for c in coarse.categorical_components:
        if not any(c == f or (c, f) in reach for f in fine.categorical_components):
            return False
    return True


def plan_structural_collinearity(
    terms: list[StructuralTerm],
    *,
    absorbed_groups=(),
    absorbed_names=(),
    protected_terms=(),
) -> StructuralCollinearityPlan:
    """Apply exact, order-preserving structural rank reductions.

    Rules are intentionally conservative:

    * a pure-dummy regressor block is removed when an absorbed FE partition
      refines it;
    * a finer pure-dummy block loses one cell per fully represented absorbed FE
      cell, because those within-FE cell dummies sum to the absorbed direction;
    * among regressors sharing the same continuous monomial, later blocks are
      reduced against earlier blocks using the same partition-refinement logic.

    No multi-block collective rank inference is attempted here. Remaining
    dependencies are resolved numerically after HDFE absorption.
    """
    omissions: list[StructuralOmission] = []
    dependencies: list[StructuralDependency] = []
    absorbed_groups = [np.asarray(g, dtype=np.int32) for g in absorbed_groups]
    absorbed_names = tuple(map(str, absorbed_names))
    protected_terms = tuple(protected_terms)
    absorbed_checks = term_checks = 0
    requested = int(sum(np.count_nonzero(t.active) for t in terms))
    component_reach, component_dependencies, closure_dependencies, component_checks, closure_inferences = (
        _component_dependency_closure(terms, protected_terms)
    )

    for term in terms:
        # Pure dummy terms can be simplified against absorbed categorical FEs.
        if not term.continuous_signature and np.any(term.active):
            for g, gname in zip(absorbed_groups, absorbed_names, strict=False):
                absorbed_checks += 1
                ng = _levels(g)
                # FE finer than term => every active term dummy is absorbed.
                covered, _ = _coverage_spans_coarse(
                    g, term.codes, np.ones(ng, dtype=bool), term.active
                )
                if np.all(covered[term.active]):
                    idx = np.flatnonzero(term.active)
                    for lev in idx:
                        omissions.append(StructuralOmission(
                            term.index, term.name, int(lev), term.level_names[int(lev)],
                            "absorbed_by_fe", (gname,), "partition_refinement",
                        ))
                    nrm = len(idx)
                    term.active[:] = False
                    dependencies.append(StructuralDependency(
                        term.name, gname, (), "partition_refinement",
                        "drop_block", nrm,
                    ))
                    break

                # Term finer than FE => one exact within-FE relation per fully
                # represented FE cell (e.g. qob#year after absorbing year).
                before = np.count_nonzero(term.active)
                om, nrm = _drop_one_fine_per_covered_coarse(
                    term, g, np.ones(ng, dtype=bool), dependency_name=gname,
                    reason="structural_within_fe_relation",
                    proof_type="partition_refinement",
                )
                if nrm:
                    omissions.extend(om)
                    dependencies.append(StructuralDependency(
                        gname, term.name, (), "partition_refinement",
                        "drop_basis_columns", nrm,
                    ))
                if before and not np.any(term.active):
                    break

        if not np.any(term.active):
            continue

        # Earlier regressor blocks have priority. Pairwise exact partition
        # relations remove only later columns, preserving user order.
        for prev in protected_terms + tuple(terms[:term.index]):
            if not np.any(prev.active):
                continue
            if prev.continuous_signature != term.continuous_signature:
                continue
            term_checks += 1

            # Earlier finer block spans the later coarse block. Component-DAG
            # closure can prove this without rescanning the joint interaction
            # partitions when the earlier block retains all of its cells.
            graph_proof = bool(np.all(prev.active)) and _component_graph_proves_refinement(
                term, prev, component_reach
            )
            if graph_proof:
                covered = np.asarray(term.active, dtype=bool).copy()
                proof_type = "component_dependency_closure"
            else:
                covered, _ = _coverage_spans_coarse(
                    prev.codes, term.codes, prev.active, term.active
                )
                proof_type = "partition_refinement"
            if np.all(covered[term.active]):
                idx = np.flatnonzero(term.active)
                for lev in idx:
                    omissions.append(StructuralOmission(
                        term.index, term.name, int(lev), term.level_names[int(lev)],
                        "structurally_spanned", (prev.name,), proof_type,
                    ))
                nrm = len(idx)
                term.active[:] = False
                dependencies.append(StructuralDependency(
                    term.name, prev.name, term.continuous_signature,
                    proof_type, "drop_block", nrm,
                ))
                break

            # Current finer block contains exact sum relations with an earlier
            # coarse block. Drop one current cell per fully covered coarse cell.
            om, nrm = _drop_one_fine_per_covered_coarse(
                term, prev.codes, prev.active, dependency_name=prev.name,
                reason="structural_linear_combination",
                proof_type="partition_refinement",
            )
            if nrm:
                omissions.extend(om)
                dependencies.append(StructuralDependency(
                    prev.name, term.name, term.continuous_signature,
                    "partition_refinement", "drop_basis_columns", nrm,
                ))
            if not np.any(term.active):
                break

    materialized = int(sum(np.count_nonzero(t.active) for t in terms))
    return StructuralCollinearityPlan(
        tuple(omissions), tuple(dependencies), requested, materialized,
        absorbed_checks, term_checks, component_dependencies, closure_dependencies,
        component_checks, closure_inferences,
    )
