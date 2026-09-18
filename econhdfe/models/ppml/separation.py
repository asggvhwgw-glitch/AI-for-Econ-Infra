from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from ...errors import SpecificationError
from ...config import ExecutionConfig
from .config import PPMLConfig
from ...hdfe.plan import FEPlan, fe_separated
from .separation_simplex import mixed_simplex_separation
from .separation_relu import relu_separation
from .execution import get_projector


@dataclass(slots=True)
class SeparationResult:
    separated: np.ndarray
    by_method: dict[str, int]
    timings: dict[str, float]
    iterations: dict[str, int]
    solvers: dict[str, str]


def detect_separation(
    y, X, plan: FEPlan, weights, config: PPMLConfig,
    execution_config: ExecutionConfig | None = None,
) -> SeparationResult:
    """Run separation methods in order with explicit stage diagnostics.

    Separation projectors are deliberately local to this call.  In
    particular, simplex uses zero observation weights; sharing that absorber
    with the positive-weight IRLS projector can disable topology-dependent
    optimized paths and increase retained memory.
    """
    execution = ExecutionConfig() if execution_config is None else execution_config
    execution.validate()
    y0 = np.asarray(y, dtype=np.float64)
    X0 = np.asarray(X, dtype=np.float64)
    w0 = np.ones(len(y0)) if weights is None else np.asarray(weights, dtype=np.float64)
    full_sep = np.zeros(len(y0), dtype=bool)
    counts: dict[str, int] = {}
    timings: dict[str, float] = {}
    iterations: dict[str, int] = {}
    solvers: dict[str, str] = {}

    for method in config.separation:
        t0 = perf_counter()
        if method == "mu":
            counts[method] = 0
            timings[method] = 0.0
            iterations[method] = 0
            solvers[method] = "post-irls"
            continue
        already_trimmed = bool(np.any(full_sep))
        if already_trimmed:
            active = ~full_sep
            if not np.any(active) or np.all(y0[active] > 0):
                counts[method] = 0
                timings[method] = perf_counter() - t0
                iterations[method] = 0
                solvers[method] = "skipped"
                continue
            ya, Xa, wa = y0[active], X0[active], w0[active]
            pa = plan.subset(active) if plan.groups else plan
        else:
            if np.all(y0 > 0):
                counts[method] = 0
                timings[method] = perf_counter() - t0
                iterations[method] = 0
                solvers[method] = "skipped"
                continue
            active = None
            ya, Xa, wa, pa = y0, X0, w0, plan

        if method == "fe":
            local = fe_separated(ya, pa)
            nit = 0
            solver = "structural"
        elif method == "simplex":
            solver_plan, _ = pa.for_engine(config.engine)
            projector = get_projector(
                solver_plan, engine=config.engine, method="map",
                execution=execution, context=None, cache=False,
            )
            info = mixed_simplex_separation(
                ya, Xa, solver_plan, wa, tol=config.simplex_tol,
                hdfe_tol=config.target_inner_tol, max_iter=config.simplex_max_iter,
                engine=config.engine, projector=projector,
            )
            local = info.separated
            nit = int(info.iterations)
            solver = f"{config.engine}:map"
        elif method == "relu":
            solver_plan, _ = pa.for_engine(config.engine)
            relu_method = "map" if config.engine == "optimized" else "lsmr"
            projector = get_projector(
                solver_plan, engine=config.engine, method=relu_method,
                execution=execution, context=None, cache=False,
            )
            info = relu_separation(
                ya, Xa, solver_plan, tol=config.relu_tol, zero_tol=config.relu_zero_tol,
                hdfe_tol=config.target_inner_tol, max_iter=config.relu_max_iter,
                engine=config.engine, projector=projector,
            )
            local = info.separated
            nit = int(info.iterations)
            solver = f"{config.engine}:{relu_method}"
        else:
            raise SpecificationError(f"unknown separation method: {method}", code="specification.separation")

        nlocal = int(np.sum(local))
        if nlocal:
            if active is None:
                full_sep |= local
            else:
                idx = np.flatnonzero(active)
                full_sep[idx[local]] = True
        counts[method] = nlocal
        timings[method] = perf_counter() - t0
        iterations[method] = nit
        solvers[method] = solver
    return SeparationResult(full_sep, counts, timings, iterations, solvers)
