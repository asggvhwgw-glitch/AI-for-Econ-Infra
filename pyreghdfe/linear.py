"""Compatibility exports for the pre-``econhdfe`` internal linear module.

This module intentionally contains no estimator implementation.  New code
should import the public estimators from :mod:`econhdfe` and numerical
primitives from :mod:`econhdfe.compute`.
"""
from econhdfe.models.ols import _fit_ols as fit_ols
from econhdfe.models.linear_iv.estimators import (
    fit_iv_2sls,
    fit_iv_gmm2s,
    fit_iv_kclass,
)
from econhdfe.compute.wls import as_2d as _as2d, weighted_arrays as _weighted_arrays

__all__ = [
    "fit_ols",
    "fit_iv_2sls",
    "fit_iv_gmm2s",
    "fit_iv_kclass",
]
