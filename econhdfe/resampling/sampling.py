from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

from ..errors import MissingDataError, InputError


def rademacher(rng: np.random.Generator, shape):
    return rng.integers(0, 2, size=shape, dtype=np.int8).astype(np.float64) * 2.0 - 1.0


def wild_weights(rng: np.random.Generator, shape, distribution: str = "rademacher") -> np.ndarray:
    """Draw standard wild-bootstrap multipliers."""
    kind = str(distribution).lower().replace("-", "_")
    if kind in {"rademacher", "radem", "rad"}:
        return rademacher(rng, shape)
    if kind == "mammen":
        root5 = math.sqrt(5.0)
        lo = (1.0 - root5) / 2.0
        hi = (1.0 + root5) / 2.0
        p_lo = (root5 + 1.0) / (2.0 * root5)
        return np.where(rng.random(shape) < p_lo, lo, hi)
    if kind == "webb":
        support = np.array([-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5)])
        return support[rng.integers(0, len(support), size=shape)]
    if kind in {"normal", "gaussian"}:
        return rng.standard_normal(size=shape)
    raise InputError(
        "wild bootstrap weight_distribution must be rademacher/mammen/webb/normal",
        code="resampling.wild_weights", stage="resampling",
    )


@dataclass(frozen=True, slots=True)
class ClusterSampler:
    """Precompiled row groups for repeated cluster resampling."""

    cluster: str
    groups: tuple[np.ndarray, ...]

    @classmethod
    def compile(cls, data: pd.DataFrame, cluster: str) -> "ClusterSampler":
        codes, uniques = pd.factorize(data[cluster], sort=False)
        if np.any(codes < 0):
            raise MissingDataError(
                f"bootstrap cluster {cluster!r} contains missing values",
                code="resampling.cluster_missing", stage="resampling",
                details={"cluster": cluster},
            )
        G = len(uniques)
        if G < 2:
            raise InputError(
                "cluster bootstrap requires at least two clusters",
                code="resampling.insufficient_clusters", stage="resampling",
                details={"cluster": cluster, "clusters": int(G)},
            )
        order = np.argsort(codes, kind="stable")
        counts = np.bincount(codes, minlength=G)
        cuts = np.cumsum(counts)
        starts = np.r_[0, cuts[:-1]]
        groups = tuple(order[a:b] for a, b in zip(starts, cuts, strict=True))
        return cls(cluster=cluster, groups=groups)

    def draw_rows(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        G = len(self.groups)
        pick = rng.integers(0, G, size=G)
        selected = tuple(self.groups[g] for g in pick)
        rows = np.concatenate(selected)
        lengths = np.fromiter((len(x) for x in selected), dtype=np.int64, count=G)
        fresh_ids = np.repeat(np.arange(1, G + 1), lengths)
        return rows, fresh_ids

    def draw(self, data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        rows, fresh_ids = self.draw_rows(rng)
        out = data.iloc[rows].copy().reset_index(drop=True)
        out.loc[:, self.cluster] = fresh_ids
        return out
