from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PlanCertificate:
    """Exactness/eligibility certificate independent of performance heuristics."""

    name: str
    eligible: bool
    exact: bool
    reason: str
    conditions: tuple[str, ...] = ()

    @classmethod
    def exact_yes(cls, name: str, reason: str = "exact_certificate") -> "PlanCertificate":
        return cls(name=name, eligible=True, exact=True, reason=reason)

    @classmethod
    def no(cls, name: str, reason: str) -> "PlanCertificate":
        return cls(name=name, eligible=False, exact=False, reason=reason)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "eligible": bool(self.eligible),
            "exact": bool(self.exact),
            "reason": self.reason,
            "conditions": tuple(self.conditions),
        }


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    """Resource envelope used by execution planners; never changes model semantics."""

    requested_budget_bytes: int
    effective_budget_bytes: int
    runtime_limit_bytes: int | None
    runtime_current_bytes: int | None
    headroom_bytes: int | None
    reserve_bytes: int
    estimated_peak_bytes: int | None
    pressure: str
    feasible: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "requested_budget_bytes": int(self.requested_budget_bytes),
            "effective_budget_bytes": int(self.effective_budget_bytes),
            "runtime_limit_bytes": self.runtime_limit_bytes,
            "runtime_current_bytes": self.runtime_current_bytes,
            "headroom_bytes": self.headroom_bytes,
            "reserve_bytes": int(self.reserve_bytes),
            "estimated_peak_bytes": self.estimated_peak_bytes,
            "pressure": self.pressure,
            "feasible": bool(self.feasible),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RepresentationCandidate:
    name: str
    certificate: PlanCertificate
    payload_bytes: int
    setup_bytes: int = 0
    pass_bytes: int | None = None
    metadata_bytes: int = 0
    dispatch_penalty_bytes: int = 0

    @property
    def per_pass_bytes(self) -> int:
        return int(self.payload_bytes if self.pass_bytes is None else self.pass_bytes)


@dataclass(frozen=True, slots=True)
class CandidateCost:
    name: str
    eligible: bool
    feasible: bool
    estimated_peak_bytes: int
    estimated_traffic_bytes: int
    score: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "eligible": bool(self.eligible),
            "feasible": bool(self.feasible),
            "estimated_peak_bytes": int(self.estimated_peak_bytes),
            "estimated_traffic_bytes": int(self.estimated_traffic_bytes),
            "score": float(self.score),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RepresentationPlan:
    representation: str
    expected_passes: int
    selected_cost: CandidateCost
    candidates: tuple[CandidateCost, ...]
    reason: str

    def as_dict(self) -> dict:
        return {
            "representation": self.representation,
            "expected_passes": int(self.expected_passes),
            "selected_cost": self.selected_cost.as_dict(),
            "candidates": tuple(c.as_dict() for c in self.candidates),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ParallelPlan:
    mode: str
    outer_workers: int
    inner_threads: int
    blas_threads: int
    effective_cpu_threads: int
    requested_threads: int | str | None
    reason: str
    auto_source: str = "explicit"
    calibration_id: str | None = None

    @property
    def total_requested_workers(self) -> int:
        return int(self.outer_workers * self.inner_threads)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "outer_workers": int(self.outer_workers),
            "inner_threads": int(self.inner_threads),
            "blas_threads": int(self.blas_threads),
            "effective_cpu_threads": int(self.effective_cpu_threads),
            "requested_threads": self.requested_threads,
            "reason": self.reason,
            "auto_source": self.auto_source,
            "calibration_id": self.calibration_id,
        }


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Explainable top-level execution decision assembled from independent subplans."""

    memory: MemoryPlan
    parallel: ParallelPlan
    representation: RepresentationPlan | None = None
    reuse_count: int = 1
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "memory": self.memory.as_dict(),
            "parallel": self.parallel.as_dict(),
            "representation": None if self.representation is None else self.representation.as_dict(),
            "reuse_count": int(self.reuse_count),
            "notes": tuple(self.notes),
        }
