from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..errors import ShapeError


@dataclass(frozen=True, slots=True)
class SampleExclusion:
    stage: str
    reason: str
    dropped: int
    remaining: int


class EstimationSampleState:
    """Monotone record of which raw observations remain eligible.

    Stages may only remove rows.  This prevents frontend, HDFE, estimator and
    inference code from silently constructing incompatible estimation samples.
    """

    __slots__ = ("_mask", "_history")

    def __init__(self, nobs: int):
        n = int(nobs)
        if n < 0:
            raise ValueError("nobs must be nonnegative")
        self._mask = np.ones(n, dtype=bool)
        self._history: list[SampleExclusion] = []

    @property
    def nobs_raw(self) -> int:
        return int(self._mask.size)

    @property
    def mask(self) -> np.ndarray:
        out = self._mask.view()
        out.flags.writeable = False
        return out

    @property
    def nobs(self) -> int:
        return int(np.count_nonzero(self._mask))

    @property
    def history(self) -> tuple[SampleExclusion, ...]:
        return tuple(self._history)

    def exclude(self, rows, *, stage: str, reason: str) -> SampleExclusion:
        arr = np.asarray(rows)
        if arr.dtype == bool:
            if arr.shape != self._mask.shape:
                raise ShapeError(
                    "sample exclusion mask has wrong shape",
                    code="data.sample_shape", stage="data",
                    details={"expected": self._mask.shape, "actual": arr.shape},
                )
            remove = arr
        else:
            idx = np.asarray(arr, dtype=np.int64).reshape(-1)
            if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= self._mask.size):
                raise ShapeError(
                    "sample exclusion index is out of range",
                    code="data.sample_index", stage="data",
                    details={"nobs": self._mask.size},
                )
            remove = np.zeros_like(self._mask)
            remove[idx] = True
        before = self.nobs
        self._mask[remove] = False
        after = self.nobs
        event = SampleExclusion(str(stage), str(reason), before - after, after)
        self._history.append(event)
        return event

    def keep(self, valid, *, stage: str, reason: str) -> SampleExclusion:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != self._mask.shape:
            raise ShapeError(
                "sample keep mask has wrong shape",
                code="data.sample_shape", stage="data",
                details={"expected": self._mask.shape, "actual": valid.shape},
            )
        return self.exclude(~valid, stage=stage, reason=reason)

    def summary(self) -> dict:
        return {
            "nobs_raw": self.nobs_raw,
            "nobs_final": self.nobs,
            "dropped_total": self.nobs_raw - self.nobs,
            "stages": tuple({
                "stage": e.stage, "reason": e.reason,
                "dropped": e.dropped, "remaining": e.remaining,
            } for e in self._history),
        }
