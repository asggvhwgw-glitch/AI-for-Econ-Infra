"""Backward-compatible bootstrap imports.

New code should use ``econhdfe.resampling`` for generic execution primitives;
model-specific OLS bootstrap remains in ``econhdfe.models.ols_bootstrap``.
"""
from .models.ols_bootstrap import wild_bootstrap, wild_cluster_bootstrap_ols
from .resampling import parallel_pairs_bootstrap
from .inference.cluster import wild_cluster_test_ols

__all__ = ["wild_bootstrap", "wild_cluster_bootstrap_ols", "wild_cluster_test_ols", "parallel_pairs_bootstrap"]
