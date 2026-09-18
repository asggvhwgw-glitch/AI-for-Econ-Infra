from __future__ import annotations

import numpy as np

from .absorber import HDFEAbsorber
from .two_way import TwoWayFEAbsorber


class WeightedFEProjector:
    """Reusable weighted FE projector for iterative WLS estimators.

    Categorical topology is compiled once. Subsequent calls update only the
    numerical weights through the public absorber ``update_weights`` contract.
    Resource controls are explicit so iterative estimators can honor the same
    ``ExecutionConfig`` thread and memory policy as the linear estimators.
    """

    def __init__(
        self,
        groups,
        *,
        engine: str,
        method: str = "map",
        absorb_threads="auto",
        projection_memory_budget_mb: float = 512,
    ):
        self.groups = tuple(np.asarray(g, dtype=np.int32) for g in groups)
        self.engine = str(engine)
        self.method = str(method)
        self.absorb_threads_requested = absorb_threads
        self.projection_memory_budget_mb = float(projection_memory_budget_mb)
        self._absorber = None
        # Optimized two-way projection cannot be updated to a weight vector
        # that removes an entire FE level. Keep separate compiled variants so
        # zero-weight simplex projections never poison the positive-weight IRLS
        # fast path.
        self._absorbers: dict[str, object] = {}
        self._active_kind: str | None = None

    def _kind(self, weights) -> str:
        if (
            self.engine == "optimized"
            and len(self.groups) == 2
            and self.method != "lsmr"
            and np.all(np.asarray(weights, dtype=np.float64) > 0)
        ):
            return "twoway"
        return "generic"

    def _build(self, weights, tol, kind: str):
        if kind == "twoway":
            out = TwoWayFEAbsorber(self.groups, weights=weights, tol=tol)
            out._configure_execution(absorb_threads=self.absorb_threads_requested)
            return out
        return HDFEAbsorber(
            self.groups,
            weights=weights,
            tol=tol,
            method=self.method,
            transform="symmetric",
            acceleration="cg" if self.method == "map" else "none",
            projection_backend="auto",
            absorb_threads=self.absorb_threads_requested,
            projection_memory_budget_mb=self.projection_memory_budget_mb,
        )

    def prepare(self, weights, *, tol):
        """Bind one numerical weight state and return the compiled absorber."""
        w = np.asarray(weights, dtype=np.float64)
        kind = self._kind(w)
        absorber = self._absorbers.get(kind)
        if absorber is None:
            absorber = self._build(w, tol, kind)
            self._absorbers[kind] = absorber
        else:
            absorber.update_weights(w, tol=tol)
        self._absorber = absorber
        self._active_kind = kind
        return absorber

    def residualize(self, data, weights, *, tol, return_info=False, copy=True):
        absorber = self.prepare(weights, tol=tol)
        return absorber.residualize(data, copy=copy, return_info=return_info)

    @property
    def compiled(self) -> bool:
        return bool(self._absorbers)

    @property
    def resource_info(self) -> dict:
        absorber = self._absorber
        return {
            "engine": self.engine,
            "method": self.method,
            "requested_threads": self.absorb_threads_requested,
            "actual_threads": None if absorber is None else int(getattr(absorber, "absorb_threads", 1)),
            "memory_budget_mb": self.projection_memory_budget_mb,
            "backend": self._active_kind,
            "projection_backend": None if absorber is None else str(getattr(absorber, "projection_backend", getattr(absorber, "method", "unknown"))),
            "index_bytes": None if absorber is None else int(getattr(absorber, "index_bytes", 0)),
        }
