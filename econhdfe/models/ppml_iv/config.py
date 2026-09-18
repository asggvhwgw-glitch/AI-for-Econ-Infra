from __future__ import annotations
from dataclasses import dataclass
from numbers import Integral
from ...errors import SpecificationError
from ..ppml.config import PPMLConfig


@dataclass(slots=True)
class IVPPMLConfig(PPMLConfig):
    """Numerical controls for IV-PPML-HDFE.

    Standardization is opt-in to match the upstream ``ivppmlhdfe`` command.
    The remaining PPML controls deliberately share the same semantics as the
    ordinary PPML estimator.
    """

    standardize: bool = False
    max_step_halving: int = 2
    step_halving_memory: float = 0.9

    def validate(self) -> None:
        PPMLConfig.validate(self)
        if isinstance(self.max_step_halving, bool) or not isinstance(self.max_step_halving, Integral) or self.max_step_halving < 0:
            raise SpecificationError("max_step_halving must be a nonnegative integer")
        if not 0.0 <= self.step_halving_memory < 1.0:
            raise SpecificationError("step_halving_memory must lie in [0, 1)")
