from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..errors import MissingDataError


@dataclass(frozen=True, slots=True)
class EncodedBatch:
    codes: np.ndarray
    new_levels: tuple[object, ...]
    n_levels: int


class StableCategoricalEncoder:
    """First-seen dictionary encoding stable across streamed batches.

    Work is proportional to the number of distinct batch levels plus one
    vectorized remap over rows; it does not perform Python dictionary lookup per
    observation.
    """

    def __init__(self):
        self._level_to_code: dict[object, int] = {}
        self._levels: list[object] = []

    @property
    def n_levels(self) -> int:
        return len(self._levels)

    @property
    def levels(self) -> tuple[object, ...]:
        return tuple(self._levels)

    def transform(self, values) -> EncodedBatch:
        arr = np.asarray(values)
        if arr.ndim != 1:
            raise ValueError("categorical values must be one-dimensional")
        local_codes, uniques = pd.factorize(arr, sort=False, use_na_sentinel=True)
        if np.any(local_codes < 0):
            raise MissingDataError(
                "categorical identifier contains missing values",
                code="data.categorical_missing", stage="data",
                details={"count": int(np.count_nonzero(local_codes < 0))},
            )
        remap = np.empty(len(uniques), dtype=np.int32)
        new: list[object] = []
        for j, value in enumerate(uniques):
            key = value.item() if isinstance(value, np.generic) else value
            code = self._level_to_code.get(key)
            if code is None:
                code = len(self._levels)
                if code > np.iinfo(np.int32).max:
                    raise OverflowError("categorical level count exceeds int32 capacity")
                self._level_to_code[key] = code
                self._levels.append(key)
                new.append(key)
            remap[j] = int(code)
        codes = remap[np.asarray(local_codes, dtype=np.int64)]
        return EncodedBatch(codes=np.asarray(codes, dtype=np.int32), new_levels=tuple(new), n_levels=self.n_levels)
