from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING
import numpy as np

from .topology import FEIdentification

if TYPE_CHECKING:
    from .diagnostics import FERecoveryDiagnosis


@dataclass(frozen=True, slots=True)
class NormalizationSpec:
    """Requested reporting normalization for additive categorical effects.

    ``baseline`` is the FE dimension that absorbs the compensating shifts.
    All other dimensions are normalized component-by-component.
    """

    kind: str = "canonical"
    baseline: int | str = 0
    references: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class EffectTermResult:
    name: str
    levels: np.ndarray
    coefficients: np.ndarray
    component: np.ndarray
    level_mass: np.ndarray
    identified_mask: np.ndarray | None = None

    @property
    def identified(self) -> np.ndarray:
        """Level-sized identification mask, independent of solver output values."""
        if self.identified_mask is None:
            return np.isfinite(self.coefficients)
        return np.asarray(self.identified_mask, dtype=bool)

    def index_of(self, level: Any) -> int:
        hits = np.flatnonzero(self.levels == level)
        if hits.size != 1:
            raise KeyError(f"level {level!r} not found uniquely in fixed effect {self.name!r}")
        return int(hits[0])


@dataclass(frozen=True, slots=True)
class RecoveryDiagnostics:
    solver: str
    converged: bool
    iterations: int
    residual_norm: float
    reconstruction_error: float
    normalization_complete: bool
    n_recovered_obs: int = 0
    n_total_obs: int = 0
    n_recovered_components: int = 0
    n_unidentified_components: int = 0
    stop_code: int | None = None
    stop_reason: str = ""
    condition_estimate: float | None = None
    normal_equation_residual_norm: float | None = None


@dataclass(frozen=True, slots=True)
class FixedEffectRecoveryResult:
    terms: tuple[EffectTermResult, ...]
    identification: FEIdentification
    normalization: NormalizationSpec
    diagnostics: RecoveryDiagnostics
    diagnosis: "FERecoveryDiagnosis | None" = None

    @property
    def identified_components(self) -> tuple[int, ...]:
        return self.identification.identified_components

    @property
    def unidentified_components(self) -> tuple[int, ...]:
        return self.identification.unidentified_components

    def term(self, key: int | str) -> EffectTermResult:
        if isinstance(key, int):
            return self.terms[key]
        for term in self.terms:
            if term.name == key:
                return term
        raise KeyError(key)

    def renormalize(self, spec: NormalizationSpec | str, **kwargs) -> "FixedEffectRecoveryResult":
        from .recover import renormalize
        if isinstance(spec, str):
            spec = NormalizationSpec(spec, **kwargs)
        return renormalize(self, spec)

    def with_diagnosis(self, diagnosis: "FERecoveryDiagnosis | None"):
        return replace(self, diagnosis=diagnosis)

    def with_terms(self, terms, normalization, *, normalization_complete: bool):
        return replace(
            self,
            terms=tuple(terms),
            normalization=normalization,
            diagnostics=replace(self.diagnostics, normalization_complete=bool(normalization_complete)),
        )
