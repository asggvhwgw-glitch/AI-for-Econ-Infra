from __future__ import annotations

from typing import Any
import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr

from ..hdfe.two_way import TwoWayFEAbsorber
from ..hdfe.codes import validate_dense_codes
from .topology import identification_structure
from .diagnostics import diagnose_fe_recovery, raise_for_fe_identification
from .results import (
    EffectTermResult,
    FixedEffectRecoveryResult,
    NormalizationSpec,
    RecoveryDiagnostics,
)


def _encode_group(values, *, assume_dense=False, levels=None):
    a = np.asarray(values)
    if a.ndim != 1:
        raise ValueError("each fixed-effect group must be one-dimensional")
    if assume_dense:
        codes = validate_dense_codes(a, dtype=np.int32)
        if codes.size and (codes.min() < 0 or not np.array_equal(np.unique(codes), np.arange(int(codes.max()) + 1))):
            raise ValueError("assume_dense=True requires zero-based contiguous codes")
        raw = np.arange(int(codes.max()) + 1 if codes.size else 0) if levels is None else np.asarray(levels)
        if len(raw) != (int(codes.max()) + 1 if codes.size else 0):
            raise ValueError("levels length does not match dense code range")
        return codes, raw
    raw, inv = np.unique(a, return_inverse=True)
    return inv.astype(np.int32, copy=False), raw


def _mass_by_level(codes, nlevels, obs_mass):
    return np.bincount(codes, weights=obs_mass, minlength=nlevels).astype(np.float64, copy=False)


def _generic_lsmr(groups, target, weights, tol, max_iter, *, diagnostics=None):
    levels = tuple(int(g.max()) + 1 if g.size else 0 for g in groups)
    offsets = np.cumsum((0,) + levels[:-1], dtype=np.int64)
    p = int(sum(levels))
    n = len(target)
    sw = np.sqrt(weights)
    mass = tuple(_mass_by_level(g, L, weights) for g, L in zip(groups, levels, strict=False))
    inv_scale = tuple(
        np.divide(1.0, np.sqrt(m), out=np.zeros_like(m), where=m > 0)
        for m in mass
    )

    def matvec(theta):
        theta = np.asarray(theta, dtype=np.float64)
        out = np.zeros(n, dtype=np.float64)
        for j, g in enumerate(groups):
            lo = int(offsets[j]); hi = lo + levels[j]
            out += theta[lo:hi][g] * inv_scale[j][g]
        return sw * out

    def rmatvec(v):
        z = sw * np.asarray(v, dtype=np.float64)
        out = np.empty(p, dtype=np.float64)
        for j, g in enumerate(groups):
            lo = int(offsets[j]); hi = lo + levels[j]
            out[lo:hi] = np.bincount(g, weights=z, minlength=levels[j]) * inv_scale[j]
        return out

    A = LinearOperator((n, p), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)
    sol = lsmr(A, sw * target, atol=tol, btol=tol, maxiter=max_iter)
    theta, istop, itn = sol[0], int(sol[1]), int(sol[2])
    if diagnostics is not None:
        reasons = {
            0: "zero_solution", 1: "residual_tolerance", 2: "normal_equation_tolerance",
            3: "condition_limit", 4: "residual_machine_precision",
            5: "normal_equation_machine_precision", 6: "condition_machine_limit",
            7: "iteration_limit",
        }
        diagnostics.update(stop_code=istop, stop_reason=reasons.get(istop, "unknown_stop"),
                           condition_estimate=float(sol[6]), normal_equation_residual_norm=float(sol[4]))
    coefs = []
    for j, L in enumerate(levels):
        lo = int(offsets[j]); hi = lo + L
        coefs.append(theta[lo:hi] * inv_scale[j])
    return tuple(coefs), (istop in {0, 1, 2, 4, 5}), itn, "lsmr"


def _two_way_schur(groups, target, weights, tol, max_iter):
    absorber = TwoWayFEAbsorber(groups, weights=weights, tol=tol, max_iter=max_iter)
    coefs, info = absorber.solve_coefficients(target)
    return coefs, bool(info.converged), int(info.iterations), "two_way_schur"


def _resolve_baseline(terms, baseline):
    if isinstance(baseline, int):
        idx = int(baseline)
        if idx < 0:
            idx += len(terms)
        if not 0 <= idx < len(terms):
            raise IndexError("normalization baseline term is out of range")
        return idx
    for i, term in enumerate(terms):
        if term.name == baseline:
            return i
    raise KeyError(f"normalization baseline term {baseline!r} not found")


def _reference_for(term, comp, refspec):
    levels = term.levels
    cidx = np.flatnonzero(term.component == comp)
    if not cidx.size:
        raise ValueError("empty component in normalization")
    if refspec is None:
        return int(cidx[0])
    if isinstance(refspec, dict):
        if comp not in refspec:
            raise ValueError(f"missing reference for component {comp} of {term.name!r}")
        wanted = refspec[comp]
    else:
        if np.unique(term.component).size != 1:
            raise ValueError(f"{term.name!r} has multiple components; references must be component-indexed")
        wanted = refspec
    hits = cidx[levels[cidx] == wanted]
    if hits.size != 1:
        raise ValueError(f"reference level {wanted!r} is not in component {comp} of {term.name!r}")
    return int(hits[0])


def renormalize(result: FixedEffectRecoveryResult, spec: NormalizationSpec) -> FixedEffectRecoveryResult:
    kind = str(spec.kind).lower()
    if kind == "canonical":
        kind = "mean_zero"
        spec = NormalizationSpec(kind="canonical", baseline=spec.baseline, references=spec.references)
    if kind == "solver":
        return result.with_terms(result.terms, spec, normalization_complete=(result.identification.nullity == 0))
    if kind not in {"mean_zero", "weighted_mean_zero", "reference"}:
        raise ValueError("normalization kind must be canonical/solver/reference/mean_zero/weighted_mean_zero")

    terms = [
        EffectTermResult(
            t.name, t.levels, t.coefficients.copy(), t.component, t.level_mass,
            None if t.identified_mask is None else t.identified_mask.copy(),
        )
        for t in result.terms
    ]
    if len(terms) <= 1 or result.identification.n_components == 0:
        return result.with_terms(terms, spec, normalization_complete=True)

    baseline = _resolve_baseline(terms, spec.baseline)
    base_coef = terms[baseline].coefficients

    for j, term in enumerate(terms):
        if j == baseline:
            continue
        coef = term.coefficients
        refspec = None if spec.references is None else spec.references.get(term.name)
        for comp in range(result.identification.n_components):
            if comp in result.identification.unidentified_components:
                continue
            idx = np.flatnonzero(term.component == comp)
            if not idx.size:
                continue
            if kind == "reference":
                ridx = _reference_for(term, comp, refspec)
                center = float(coef[ridx])
            elif kind == "weighted_mean_zero":
                w = term.level_mass[idx]
                denom = float(w.sum())
                center = float(np.dot(w, coef[idx]) / denom) if denom > 0 else float(coef[idx].mean())
            else:
                center = float(coef[idx].mean())
            coef[idx] -= center
            bidx = np.flatnonzero(terms[baseline].component == comp)
            base_coef[bidx] += center

    complete = bool(result.identification.extra_nullity == 0)
    return result.with_terms(terms, spec, normalization_complete=complete)


def recover_fixed_effects(
    target,
    groups,
    *,
    names=None,
    levels=None,
    weights=None,
    normalization: NormalizationSpec | str = "canonical",
    assume_dense: bool = False,
    solver: str = "auto",
    tol: float = 1e-10,
    max_iter: int = 16_000,
    rank_backend: str = "auto",
    strict_identification: bool = True,
    identification_diagnosis=None,
) -> FixedEffectRecoveryResult:
    """Recover additive categorical FE coefficients from an identified contribution.

    This is deliberately estimator-agnostic: ``target`` is any vector known to
    satisfy approximately ``target = D gamma`` on the estimation sample.  OLS,
    IV, PPML, and IV-PPML adapters only need to provide that contribution.

    The result stores level-sized coefficients and metadata; no observation-sized
    FE contributions are retained.
    """
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 1 or not np.all(np.isfinite(target)):
        raise ValueError("target must be a finite one-dimensional vector")
    if not np.isfinite(tol) or tol <= 0 or int(max_iter) != max_iter or max_iter <= 0:
        raise ValueError("tol must be finite and positive; max_iter must be a positive integer")
    raw_groups = tuple(groups)
    if not raw_groups:
        raise ValueError("at least one fixed-effect group is required")
    if any(len(g) != len(target) for g in raw_groups):
        raise ValueError("target and fixed-effect groups must have equal length")

    if names is None:
        names = tuple(f"fe{j}" for j in range(len(raw_groups)))
    else:
        names = tuple(str(x) for x in names)
    if len(names) != len(raw_groups) or len(set(names)) != len(names):
        raise ValueError("names must contain one unique name per fixed-effect group")

    level_inputs = (None,) * len(raw_groups) if levels is None else tuple(levels)
    if len(level_inputs) != len(raw_groups):
        raise ValueError("levels must contain one entry per fixed-effect group")
    encoded = [
        _encode_group(g, assume_dense=assume_dense, levels=level_inputs[j])
        for j, g in enumerate(raw_groups)
    ]
    codes = tuple(x[0] for x in encoded)
    raw_levels = tuple(x[1] for x in encoded)

    obs_w = np.ones(len(target), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if obs_w.ndim != 1 or len(obs_w) != len(target) or not np.all(np.isfinite(obs_w)) or np.any(obs_w <= 0):
        raise ValueError("first-stage FE recovery currently requires strictly positive observation weights")

    diagnosis = identification_diagnosis
    if diagnosis is None:
        diagnosis = diagnose_fe_recovery(codes, names=names, rank_backend=rank_backend)
    if strict_identification:
        # Partial component rank failures are warnings: independent identified
        # blocks are salvaged. A fatal error remains fatal (e.g. every block is
        # unidentified or the final sample is empty).
        raise_for_fe_identification(diagnosis)
    identification = diagnosis.final_identification

    bad_components = tuple(identification.unidentified_components)
    good_components = tuple(identification.identified_components)
    obs_component = (
        identification.component_by_term[0][codes[0]]
        if codes and len(target) else np.empty(0, dtype=np.int32)
    )

    salvage = bool(strict_identification and bad_components and good_components)
    if salvage:
        recover_mask = np.isin(obs_component, np.asarray(good_components, dtype=np.int32))
    else:
        recover_mask = np.ones(len(target), dtype=bool)

    # Re-factorize the recovery sample so the numerical solvers see compact
    # zero-based codes. Keep the map back to original level-sized arrays.
    solver_codes = []
    solver_to_global = []
    for g in codes:
        present, inv = np.unique(g[recover_mask], return_inverse=True)
        solver_codes.append(inv.astype(np.int32, copy=False))
        solver_to_global.append(present.astype(np.int64, copy=False))
    solver_codes = tuple(solver_codes)

    solver_key = str(solver).lower()
    if solver_key == "auto":
        solver_key = "schur" if len(codes) == 2 else "lsmr"
    target_s = target[recover_mask]
    weights_s = obs_w[recover_mask]
    solver_diagnostics = {}
    if solver_key == "schur":
        if len(codes) != 2:
            raise ValueError("schur recovery is available only for exactly two categorical FE dimensions")
        local_coefs, converged, iterations, solver_name = _two_way_schur(
            solver_codes, target_s, weights_s, tol, max_iter
        )
    elif solver_key == "lsmr":
        local_coefs, converged, iterations, solver_name = _generic_lsmr(
            solver_codes, target_s, weights_s, tol, max_iter, diagnostics=solver_diagnostics
        )
    else:
        raise ValueError("solver must be 'auto', 'schur', or 'lsmr'")

    # Map the solved coefficients back to the full level space.  In the default
    # strict mode, levels belonging to unidentified independent components are
    # deliberately NaN instead of receiving an arbitrary numerical solution.
    coefs = []
    identified_masks = []
    for j, (lv, comp, local_coef, global_idx) in enumerate(zip(
        raw_levels, identification.component_by_term, local_coefs, solver_to_global, strict=False
    )):
        full = np.full(len(lv), np.nan, dtype=np.float64) if salvage else np.zeros(len(lv), dtype=np.float64)
        full[global_idx] = np.asarray(local_coef, dtype=np.float64)
        identified = np.isin(comp, np.asarray(good_components, dtype=np.int32))
        coefs.append(full)
        identified_masks.append(identified)

    fitted = np.zeros(int(recover_mask.sum()), dtype=np.float64)
    terms = []
    for j, (g, lv, coef, comp, identified) in enumerate(zip(
        codes, raw_levels, coefs, identification.component_by_term, identified_masks, strict=False
    )):
        # Only the recovered sample contributes to numerical reconstruction.
        fitted += coef[g[recover_mask]]
        mass = _mass_by_level(g, len(coef), obs_w)
        terms.append(EffectTermResult(
            names[j], np.asarray(lv), np.asarray(coef), comp, mass, identified
        ))
    resid = target_s - fitted
    denom = max(float(np.linalg.norm(target_s)), np.finfo(float).eps)
    diagnostics = RecoveryDiagnostics(
        solver=solver_name,
        converged=bool(converged and np.all(np.isfinite(fitted)) and all(np.all(np.isfinite(c)) for c in local_coefs)),
        iterations=int(iterations),
        residual_norm=float(np.linalg.norm(resid)),
        reconstruction_error=float(np.linalg.norm(resid) / denom),
        normalization_complete=(identification.extra_nullity == 0),
        n_recovered_obs=int(recover_mask.sum()),
        n_total_obs=int(len(target)),
        n_recovered_components=len(good_components) if identification.components else identification.n_components,
        n_unidentified_components=len(bad_components),
        **solver_diagnostics,
    )
    result = FixedEffectRecoveryResult(tuple(terms), identification, NormalizationSpec("solver"), diagnostics, diagnosis)
    spec = NormalizationSpec(normalization) if isinstance(normalization, str) else normalization
    return renormalize(result, spec)
