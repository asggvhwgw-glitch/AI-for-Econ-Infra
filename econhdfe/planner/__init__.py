"""Estimator-agnostic execution planning contracts.

This layer may choose how an already-defined econometric problem is executed;
it must never infer or alter the statistical model itself.
"""
from .contracts import (
    PlanCertificate, MemoryPlan, RepresentationCandidate, CandidateCost,
    RepresentationPlan, ParallelPlan, ExecutionPlan,
)
from .core import plan_memory, plan_parallelism, choose_representation, exact_certificate, plan_execution
from .resources import (
    RuntimeResources, runtime_resources, effective_cpu_count,
    cpu_quota, affinity_cpus, configure_numba_runtime,
)
from .calibration import (
    ThreadCalibration, auto_thread_count, calibrate_threads, get_thread_calibration,
    default_calibration_cache_path, runtime_thread_fingerprint,
)
from .report import (
    PlannerDeveloperReport, build_planner_developer_report, write_planner_developer_report,
)

__all__ = [
    "PlanCertificate", "MemoryPlan", "RepresentationCandidate", "CandidateCost",
    "RepresentationPlan", "ParallelPlan", "ExecutionPlan",
    "plan_memory", "plan_parallelism", "choose_representation", "exact_certificate", "plan_execution",
    "RuntimeResources", "runtime_resources", "effective_cpu_count",
    "cpu_quota", "affinity_cpus", "configure_numba_runtime",
    "ThreadCalibration", "auto_thread_count", "calibrate_threads", "get_thread_calibration",
    "default_calibration_cache_path", "runtime_thread_fingerprint",
    "PlannerDeveloperReport", "build_planner_developer_report", "write_planner_developer_report",
]
