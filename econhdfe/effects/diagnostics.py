from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from ..compute.encoding import factorize_1d
from ..errors import IdentificationError
from .topology import FEIdentification, identification_structure


@dataclass(frozen=True, slots=True)
class FEDiagnosticIssue:
    """One user-facing explanation for FE-recovery identification/support."""

    code: str
    severity: str
    message: str
    details: dict[str, Any]
    suggestion: str | None = None


@dataclass(frozen=True, slots=True)
class FERecoveryDiagnosis:
    """Compact identification diagnosis for categorical FE recovery."""

    estimable: bool
    final_identification: FEIdentification
    issues: tuple[FEDiagnosticIssue, ...]
    raw_identification: FEIdentification | None = None
    post_singleton_identification: FEIdentification | None = None

    @property
    def fatal_issues(self) -> tuple[FEDiagnosticIssue, ...]:
        return tuple(x for x in self.issues if x.severity == "error")

    def primary_error(self) -> FEDiagnosticIssue | None:
        fatal = self.fatal_issues
        return fatal[0] if fatal else None


class FixedEffectIdentificationError(IdentificationError):
    default_code = "identification.fe_recovery"
    default_stage = "fe_recovery"


def _dense_groups(groups, mask=None):
    out = []
    for raw in tuple(groups):
        a = np.asarray(raw)
        if a.ndim != 1:
            raise ValueError("each fixed-effect group must be one-dimensional")
        if mask is not None:
            a = a[mask]
        codes, _ = factorize_1d(a)
        out.append(codes)
    return tuple(out)


def _identification(groups, mask, rank_backend):
    if mask is not None and not np.any(mask):
        return None
    dense = _dense_groups(groups, mask)
    return identification_structure(dense, rank_backend=rank_backend)


def _nesting_relations(groups, names, mask=None):
    """Return exact pairwise refinement relations on the requested sample.

    ``finer -> coarser`` means every level of the finer FE maps to exactly one
    level of the coarser FE, so the coarser dummy span is contained in the
    finer span for intercept-only categorical effects.
    """
    arrays = [np.asarray(g) if mask is None else np.asarray(g)[mask] for g in groups]
    dense = [factorize_1d(a)[0] for a in arrays]
    rel = []
    for i, gi in enumerate(dense):
        Li = int(gi.max()) + 1 if gi.size else 0
        for j, gj in enumerate(dense):
            if i == j:
                continue
            # gi -> gj is deterministic iff each gi level has one gj value.
            first = np.full(Li, -1, dtype=np.int64)
            ok = True
            for a, b in zip(gi, gj, strict=False):
                a = int(a); b = int(b)
                if first[a] < 0:
                    first[a] = b
                elif first[a] != b:
                    ok = False
                    break
            if ok:
                rel.append((str(names[i]), str(names[j])))
    # Keep one direction for exact duplicate partitions to avoid noisy output.
    out = []
    seen = set()
    for a, b in rel:
        if (b, a) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
    return tuple(out)


def _issue(code, severity, message, *, suggestion=None, **details):
    return FEDiagnosticIssue(
        code=str(code), severity=str(severity), message=str(message),
        details=details, suggestion=suggestion,
    )


def diagnose_fe_recovery(
    groups,
    *,
    names=None,
    final_mask=None,
    singleton_mask=None,
    separation_mask=None,
    separation_by_method=None,
    reported_singletons: int = 0,
    reported_separated: int = 0,
    estimator: str | None = None,
    rank_backend: str = "auto",
) -> FERecoveryDiagnosis:
    """Explain whether categorical FE coefficients are recoverable.

    When raw-sample masks are available, diagnostics are staged as
    raw -> singleton-pruned -> final (e.g. PPML separation-pruned).  This lets
    the error attribute an *additional* rank failure to the stage that created
    it instead of merely observing the final deficiency.
    """
    groups = tuple(groups)
    if not groups:
        raise ValueError("at least one fixed-effect group is required")
    n = len(groups[0])
    if any(len(g) != n for g in groups):
        raise ValueError("all fixed-effect groups must have equal length")
    names = tuple(f"fe{i}" for i in range(len(groups))) if names is None else tuple(map(str, names))
    if len(names) != len(groups):
        raise ValueError("names must contain one name per fixed-effect group")

    def as_mask(value, label):
        if value is None:
            return None
        m = np.asarray(value, dtype=bool)
        if m.ndim != 1 or len(m) != n:
            raise ValueError(f"{label} must have one value per raw observation")
        return m

    final_mask = as_mask(final_mask, "final_mask")
    singleton_mask = as_mask(singleton_mask, "singleton_mask")
    separation_mask = as_mask(separation_mask, "separation_mask")

    raw_id = _identification(groups, None, rank_backend)
    if singleton_mask is not None:
        after_singleton = ~singleton_mask
        post_singleton_id = _identification(groups, after_singleton, rank_backend)
    else:
        after_singleton = None
        post_singleton_id = None

    if final_mask is None:
        if after_singleton is not None:
            final_mask = after_singleton.copy()
            if separation_mask is not None:
                final_mask &= ~separation_mask
        else:
            final_mask = np.ones(n, dtype=bool)
            if separation_mask is not None:
                final_mask &= ~separation_mask
    final_id = _identification(groups, final_mask, rank_backend)
    if final_id is None:
        # Construct an empty identification object only for a stable error payload.
        final_id = FEIdentification(0, 0, 0, 0, 0, 0, ())

    issues = []
    n_singletons = int(np.sum(singleton_mask)) if singleton_mask is not None else int(reported_singletons)
    n_separated = int(np.sum(separation_mask)) if separation_mask is not None else int(reported_separated)

    if n_singletons:
        issues.append(_issue(
            "fe_recovery.singleton_pruned", "warning",
            f"{n_singletons} observation(s) were removed by recursive singleton pruning before FE recovery.",
            suggestion="Interpret recovered effects only for levels that survive the final estimation sample.",
            dropped=n_singletons,
        ))
    if n_separated:
        issues.append(_issue(
            "fe_recovery.ppml_separation_pruned", "warning",
            f"{n_separated} observation(s) were removed because PPML did not have a finite estimate on that support.",
            suggestion="Recovered PPML effects are defined only on the finite-MLE estimation sample.",
            dropped=n_separated,
            by_method=dict(separation_by_method or {}),
        ))

    if final_id.n_components > 1:
        issues.append(_issue(
            "fe_recovery.disconnected_support", "warning",
            f"The final FE incidence structure has {final_id.n_components} disconnected component(s).",
            suggestion="Compare FE coefficients only within a connected component unless the structural model supplies additional restrictions.",
            n_components=int(final_id.n_components),
        ))

    nesting = _nesting_relations(groups, names, final_mask) if np.any(final_mask) else ()
    if nesting:
        issues.append(_issue(
            "fe_recovery.nested_or_redundant", "warning",
            "Some categorical FE partitions are nested or exact refinements on the final sample.",
            suggestion="Treat the coarser FE as separately interpretable only when the remaining support identifies within-component contrasts.",
            relations=nesting,
        ))

    if not np.any(final_mask):
        issues.append(_issue(
            "identification.fe_empty_final_sample", "error",
            "No observations remain in the final estimation sample, so fixed effects cannot be recovered.",
            suggestion="Inspect singleton pruning and PPML separation diagnostics.",
        ))
    elif final_id.extra_nullity > 0:
        raw_extra = raw_id.extra_nullity if raw_id is not None else None
        sing_extra = post_singleton_id.extra_nullity if post_singleton_id is not None else raw_extra
        bad_components = tuple(final_id.unidentified_components)
        good_components = tuple(final_id.identified_components)
        partial = bool(good_components and bad_components)

        if raw_extra == 0 and post_singleton_id is not None and sing_extra > 0:
            code = "identification.fe_singleton_induced"
            message = (
                "Recursive singleton pruning removed identifying support: the surviving FE design has "
                f"{final_id.extra_nullity} additional unidentified direction(s)."
            )
            suggestion = "Review singleton-heavy FE levels or the estimation sample; normalization cannot restore the lost identifying links."
        elif (sing_extra == 0 and n_separated > 0 and final_id.extra_nullity > 0):
            code = "identification.fe_ppml_separation_induced"
            message = (
                "PPML separation removed identifying support: after finite-MLE trimming the FE design has "
                f"{final_id.extra_nullity} additional unidentified direction(s)."
            )
            suggestion = "Inspect the reported separation methods/support; FE normalization cannot repair separation-induced rank loss."
        elif raw_extra is not None and raw_extra > 0:
            code = "identification.fe_data_rank_deficiency"
            message = (
                "The requested FE specification is not fully identified in the realized data: "
                f"{final_id.extra_nullity} unidentified direction(s) remain beyond ordinary additive-FE normalization."
            )
            suggestion = "Change the sample/specification or provide explicit structural restrictions through an advanced constraint interface."
        else:
            code = "identification.fe_extra_nullity"
            message = (
                f"The final FE design has {final_id.extra_nullity} unidentified direction(s) beyond ordinary normalization."
            )
            suggestion = "Inspect support/connectivity or provide explicit structural restrictions; do not resolve this with arbitrary dummy dropping."

        severity = "warning" if partial else "error"
        if partial:
            base_code = code
            code = "identification.fe_partial_component_rank_deficiency"
            message = (
                f"{len(bad_components)} disconnected FE component(s) are not fully identified, while "
                f"{len(good_components)} component(s) remain recoverable. Unidentified components will be returned as unavailable."
            )
            suggestion = (
                "Use the identified components normally. Compare effects only within their component; "
                "supply structural restrictions only if the unidentified blocks are substantively required."
            )
        else:
            base_code = code

        issues.append(_issue(
            code, severity, message, suggestion=suggestion,
            estimator=estimator,
            rank=int(final_id.rank),
            n_levels=int(final_id.n_levels),
            nullity=int(final_id.nullity),
            ordinary_shift_nullity=int(final_id.structural_shift_nullity),
            extra_nullity=int(final_id.extra_nullity),
            identified_components=good_components,
            unidentified_components=bad_components,
            component_extra_nullity={int(c.component): int(c.extra_nullity) for c in final_id.components},
            component_failure_code=base_code,
            raw_extra_nullity=None if raw_extra is None else int(raw_extra),
            post_singleton_extra_nullity=None if sing_extra is None else int(sing_extra),
            n_singletons=n_singletons,
            n_separated=n_separated,
            nesting_relations=nesting,
            separation_by_method=dict(separation_by_method or {}),
        ))

    return FERecoveryDiagnosis(
        estimable=not any(x.severity == "error" for x in issues),
        final_identification=final_id,
        issues=tuple(issues),
        raw_identification=raw_id,
        post_singleton_identification=post_singleton_id,
    )


def raise_for_fe_identification(diagnosis: FERecoveryDiagnosis) -> None:
    issue = diagnosis.primary_error()
    if issue is None:
        return
    raise FixedEffectIdentificationError(
        issue.message,
        code=issue.code,
        stage="fe_recovery",
        details=issue.details,
        suggestion=issue.suggestion,
    )
