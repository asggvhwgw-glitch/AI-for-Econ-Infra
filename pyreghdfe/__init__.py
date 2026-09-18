"""Compatibility namespace for pre-1.0 pyreghdfe users.

New code should import :mod:`econhdfe`.
"""
import importlib
import sys
from econhdfe import *  # noqa: F401,F403
from econhdfe import __version__

_ALIASES = {
    "api": "econhdfe.api",
    "absorber": "econhdfe.hdfe.absorber",
    "two_way": "econhdfe.hdfe.two_way",
    "group_individual": "econhdfe.hdfe.group_individual",
    "dof": "econhdfe.hdfe.dof",
    "rank": "econhdfe.hdfe.rank",
    "encoding": "econhdfe.hdfe.encoding",
    "fe_projection": "econhdfe.hdfe.projection",
    "fe_structure": "econhdfe.hdfe.structure",
    "fe_plan": "econhdfe.hdfe.plan",
    "specs": "econhdfe.hdfe.specs",
    "weighted_projection": "econhdfe.hdfe.weighted_projection",
    "backend": "econhdfe.compute.backend",
    "kernels": "econhdfe.compute.kernels",
    "linalg": "econhdfe.compute.linalg",
    "runtime": "econhdfe.compute.runtime",
    "execution": "econhdfe.compute.execution",
    "weights": "econhdfe.compute.weights",
    "vcov": "econhdfe.compute.vcov",
    "bootstrap": "econhdfe.bootstrap",
    "diagnostics": "econhdfe.models.linear_iv.diagnostics",
    "stock_yogo": "econhdfe.models.linear_iv.stock_yogo",
    "results": "econhdfe.results",
    "design": "econhdfe.design",
    "factorvars": "econhdfe.factorvars",
    "design_structure": "econhdfe.design_structure",
    "collinearity": "econhdfe.collinearity",
    "ppml": "econhdfe.models.ppml",
    "ppml_iv": "econhdfe.models.ppml_iv",
}
for _old, _new in _ALIASES.items():
    sys.modules[f"{__name__}.{_old}"] = importlib.import_module(_new)

for _name in ("api", "config", "estimator", "irls", "results", "separation", "separation_relu", "separation_simplex", "standardize", "vce"):
    sys.modules[f"{__name__}.ppml.{_name}"] = importlib.import_module(f"econhdfe.models.ppml.{_name}")
