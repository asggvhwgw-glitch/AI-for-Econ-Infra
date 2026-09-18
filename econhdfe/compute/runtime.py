"""Compatibility surface for runtime resource discovery.

The canonical resource contract lives in :mod:`econhdfe.planner.resources` so
Data, Design and Numerical Execution can share one view of CPU/memory limits.
"""
from ..planner.resources import (
    RuntimeResources,
    affinity_cpus,
    configure_numba_runtime,
    cpu_quota,
    effective_cpu_count,
    runtime_resources,
)

__all__ = [
    "RuntimeResources", "affinity_cpus", "configure_numba_runtime",
    "cpu_quota", "effective_cpu_count", "runtime_resources",
]
