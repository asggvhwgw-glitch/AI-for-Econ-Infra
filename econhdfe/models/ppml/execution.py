from __future__ import annotations

from ...config import ExecutionConfig
from ...compute.context import ExecutionContext
from ...hdfe.plan import FEPlan
from ...hdfe.weighted_projection import WeightedFEProjector


def resolve_execution(
    execution_config: ExecutionConfig | None = None,
    execution_context: ExecutionContext | None = None,
) -> tuple[ExecutionConfig, ExecutionContext]:
    """Return one validated PPML execution policy and lifecycle context."""
    if execution_config is None:
        execution_config = execution_context.config if execution_context is not None else ExecutionConfig()
    execution_config.validate()
    if execution_context is None:
        execution_context = ExecutionContext(execution_config)
    return execution_config, execution_context


def _projector_key(plan: FEPlan, *, engine: str, method: str, execution: ExecutionConfig) -> str:
    return (
        f"ppml.projector:{plan.fingerprint}:{engine}:{method}:"
        f"threads={execution.threads}:memory={execution.memory_budget_mb}"
    )


def get_projector(
    plan: FEPlan,
    *,
    engine: str,
    method: str = "map",
    execution: ExecutionConfig,
    context: ExecutionContext | None = None,
    cache: bool = True,
) -> WeightedFEProjector | None:
    """Build or reuse a resource-aware weighted FE projector.

    Only estimator projectors should normally be placed in a reusable model's
    persistent ``ExecutionContext``.  Separation projectors are intentionally
    short-lived because simplex can introduce zero weights, which changes the
    numerical topology available to optimized absorbers.
    """
    if not plan.groups:
        return None
    key = _projector_key(plan, engine=engine, method=method, execution=execution)
    if cache and context is not None:
        hit = context.get(key)
        if isinstance(hit, WeightedFEProjector):
            return hit
    out = WeightedFEProjector(
        plan.groups,
        engine=engine,
        method=method,
        absorb_threads=execution.threads,
        projection_memory_budget_mb=execution.memory_budget_mb,
    )
    if cache and context is not None:
        context.put(key, out)
    return out
