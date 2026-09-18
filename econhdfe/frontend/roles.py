from __future__ import annotations
from enum import Enum


class VariableRole(str, Enum):
    OUTCOME = "outcome"
    REGRESSOR = "regressor"
    EXOGENOUS = "exogenous"
    ENDOGENOUS = "endogenous"
    INSTRUMENT = "instrument"
    WEIGHT = "weight"
    OFFSET = "offset"
    EXPOSURE = "exposure"
    FE = "fixed_effect"
    CLUSTER = "cluster"
    FACTOR = "factor"
    SLOPE = "slope"

    @property
    def requires_numeric(self) -> bool:
        return self not in {self.FE, self.CLUSTER, self.FACTOR}
