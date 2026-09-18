from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np


class EconHDFEError(Exception):
    """Base class for user-facing econhdfe failures.

    Every public error exposes a stable machine-readable ``code`` and ``stage``
    plus structured ``details``.  The original exception is preserved as the
    Python exception cause when a boundary translates a lower-level failure.
    """

    default_code = "econhdfe.error"
    default_stage = "estimation"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        stage: str | None = None,
        details: Mapping[str, Any] | None = None,
        suggestion: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        self.code = str(code or self.default_code)
        self.stage = str(stage or self.default_stage)
        self.details = dict(details or {})
        self.suggestion = None if suggestion is None else str(suggestion)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "type": type(self).__name__,
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
            "details": dict(self.details),
        }
        if self.suggestion:
            out["suggestion"] = self.suggestion
        return out


# Mix in the built-in exception families so existing user code/tests that catch
# ValueError or RuntimeError remain backward compatible.
class InputError(EconHDFEError, ValueError):
    default_code = "input.invalid"
    default_stage = "frontend"


class DataTypeError(InputError):
    default_code = "input.non_numeric"


class ShapeError(InputError):
    default_code = "input.shape"


class MissingDataError(InputError):
    default_code = "input.missing"


class InvalidWeightError(InputError):
    default_code = "input.invalid_weight"


class SpecificationError(EconHDFEError, ValueError):
    default_code = "specification.invalid"
    default_stage = "specification"


class IdentificationError(EconHDFEError, ValueError):
    default_code = "identification.failure"
    default_stage = "identification"


class UnderidentifiedError(IdentificationError):
    default_code = "identification.underidentified"


class CollinearityError(IdentificationError):
    default_code = "identification.collinearity"


class ConvergenceError(EconHDFEError, RuntimeError):
    default_code = "convergence.failure"
    default_stage = "solver"


class NumericalError(EconHDFEError, RuntimeError):
    default_code = "numerical.failure"
    default_stage = "compute"




class LinearAlgebraError(NumericalError, np.linalg.LinAlgError):
    default_code = "numerical.linalg"

class DivergenceError(NumericalError):
    default_code = "numerical.divergence"


class InferenceError(EconHDFEError, RuntimeError):
    default_code = "inference.failure"
    default_stage = "inference"


class BootstrapError(EconHDFEError, RuntimeError):
    default_code = "resampling.failure"
    default_stage = "resampling"


class InternalEstimationError(EconHDFEError, RuntimeError):
    default_code = "estimation.internal"
    default_stage = "estimation"


def _wrap_unknown(exc: BaseException, *, stage: str) -> EconHDFEError:
    details = {"cause_type": type(exc).__name__}
    if isinstance(exc, KeyError):
        return InputError(
            f"missing input column or key: {exc}", code="input.missing_key",
            stage=stage, details=details,
            suggestion="Check the requested column names and the input DataFrame schema.",
        )
    if isinstance(exc, np.linalg.LinAlgError):
        return LinearAlgebraError(str(exc), stage=stage, details=details)
    if isinstance(exc, (FloatingPointError, OverflowError)):
        return NumericalError(str(exc), stage=stage, details=details)
    if isinstance(exc, ValueError):
        return SpecificationError(str(exc), stage=stage, details=details)
    if isinstance(exc, RuntimeError):
        return InternalEstimationError(str(exc), stage=stage, details=details)
    return InternalEstimationError(
        str(exc) or type(exc).__name__, stage=stage, details=details,
    )


def error_boundary(stage: str):
    """Translate lower-level failures at public API boundaries.

    The decorator intentionally does *not* catch arbitrary ``Exception``:
    programming errors such as AttributeError/NameError should remain visible
    during development instead of being mislabeled as user mistakes.
    """
    def decorate(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except EconHDFEError:
                raise
            except (KeyError, ValueError, RuntimeError, np.linalg.LinAlgError,
                    FloatingPointError, OverflowError) as exc:
                raise _wrap_unknown(exc, stage=stage) from exc
        return wrapped
    return decorate
