from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ..errors import UnderidentifiedError

from ..compute.wls import as_2d


@dataclass(frozen=True)
class IVDesign:
    """Outcome-agnostic IV column roles.

    ``exog`` are included exogenous regressors, ``endog`` are included
    endogenous regressors, and ``excluded`` are excluded instruments.
    The full regressor and instrument matrices are exposed through ``X`` and
    ``Z`` respectively.  This object deliberately contains no outcome-model
    logic, so the same design can be used by linear IV and future IV-PPML.
    """

    exog: np.ndarray
    endog: np.ndarray
    excluded: np.ndarray

    @classmethod
    def from_arrays(cls, exog, endog, instruments, nobs: int) -> "IVDesign":
        c = as_2d(exog, nobs)
        e = as_2d(endog, nobs)
        z = as_2d(instruments, nobs)
        if e.shape[1] == 0:
            raise UnderidentifiedError("at least one endogenous regressor is required", details={"n_endog": 0})
        if z.shape[1] < e.shape[1]:
            raise UnderidentifiedError("number of excluded instruments must be >= number of endogenous regressors", details={"n_endog": e.shape[1], "n_excluded_instruments": z.shape[1]})
        return cls(c, e, z)

    @property
    def X(self) -> np.ndarray:
        return np.column_stack([self.exog, self.endog])

    @property
    def Z(self) -> np.ndarray:
        return np.column_stack([self.exog, self.excluded])

    @property
    def n_exog(self) -> int:
        return int(self.exog.shape[1])

    @property
    def n_endog(self) -> int:
        return int(self.endog.shape[1])

    @property
    def n_excluded(self) -> int:
        return int(self.excluded.shape[1])
