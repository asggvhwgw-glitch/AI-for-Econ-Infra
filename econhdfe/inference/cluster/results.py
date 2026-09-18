from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class ClusterDimensionDiagnostics:
    dimension: int
    n_clusters: int
    min_size: int
    median_size: float
    mean_size: float
    max_size: int
    size_cv: float
    largest_share: float
    effective_clusters_size: float
    singleton_clusters: int
    score_share_max: float | None = None
    score_effective_clusters: float | None = None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ClusterDiagnostics:
    nobs: int
    dimensions: tuple[ClusterDimensionDiagnostics, ...]
    pairwise_intersections: dict[tuple[int, int], int]

    @property
    def cluster_counts(self) -> tuple[int, ...]:
        return tuple(d.n_clusters for d in self.dimensions)

    @property
    def flags(self) -> tuple[str, ...]:
        out: list[str] = []
        for d in self.dimensions:
            out.extend(d.flags)
        return tuple(dict.fromkeys(out))


@dataclass(slots=True)
class WildClusterTestResult:
    statistic: float
    pvalue: float
    bootstrap_statistics: np.ndarray
    reps: int
    completed_reps: int
    bootstrap_type: str
    weight_distribution: str
    alternative: str
    cluster_count: int
    restriction_matrix: np.ndarray
    restriction_value: np.ndarray
    observed_difference: np.ndarray
    full_enumeration: bool = False
    requested_reps: int | None = None
