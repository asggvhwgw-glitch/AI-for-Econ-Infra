"""Shared instrumental-variable infrastructure.

This package is outcome-model agnostic.  Linear IV estimators live under
``econhdfe.models.linear_iv``; future IV-PPML can reuse these design, moment,
and weighted-2SLS primitives without depending on the linear-IV model layer.
"""
from .design import IVDesign
from .moments import additive_moments, additive_score_matrix
from .solve import Weighted2SLSResult, weighted_2sls

__all__ = [
    "IVDesign",
    "Weighted2SLSResult",
    "weighted_2sls",
    "additive_score_matrix",
    "additive_moments",
]
