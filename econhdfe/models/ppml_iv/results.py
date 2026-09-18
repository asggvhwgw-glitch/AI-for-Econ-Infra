from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from ...reporting import inference_table, reproducibility_dict


@dataclass(slots=True)
class IVPPMLResult:
    coef: np.ndarray
    vcov: np.ndarray
    stderr: np.ndarray
    names: tuple[str, ...]
    converged: bool
    iterations: int
    deviance: float
    loglike: float
    nobs: int
    n_separated: int
    sample_mask: np.ndarray
    separation_mask: np.ndarray
    mu: np.ndarray
    eta: np.ndarray
    df_absorbed: int
    exog_names: tuple[str, ...]
    endog_names: tuple[str, ...]
    instrument_names: tuple[str, ...]
    moments: np.ndarray
    diagnostics: dict = field(default_factory=dict)
    df_resid: float | None = None
    vce: str | None = None
    cluster_counts: tuple[int, ...] = ()
    fe_names: tuple[str, ...] = ()
    weight_type: str = "none"
    sum_weights: float | None = None
    n_singletons: int = 0
    confidence_level: float = 0.95
    profile: dict | None = None
    reproducibility: dict | None = None
    diagnostics_mode: str = "off"

    @property
    def params(self) -> dict[str, float]:
        return dict(zip(self.names, map(float, self.coef), strict=False))


def _coef_table(self, level=None):
    table = inference_table(self.names, self.coef, self.stderr,
                            level=self.confidence_level if level is None else level,
                            distribution="normal")
    normalized = np.array([name == "_cons" for name in self.names]) & (self.stderr == 0)
    table.loc[normalized, ["statistic", "p_value", "ci_low", "ci_high"]] = np.nan
    return table

def _pvalues(self):
    return self.coef_table()["p_value"].to_numpy()

def _conf_int(self, level=None):
    t = self.coef_table(level); return t[["ci_low", "ci_high"]].to_numpy()

def _model_stats(self):
    return {
        "estimator": "ivppml", "nobs": int(self.nobs),
        "df_resid": None if self.df_resid is None else float(self.df_resid),
        "df_absorbed": int(self.df_absorbed), "vce": self.vce,
        "cluster_counts": tuple(self.cluster_counts), "fixed_effects": tuple(self.fe_names),
        "n_separated": int(self.n_separated), "n_singletons": int(self.n_singletons),
        "iterations": int(self.iterations), "converged": bool(self.converged),
        "loglike": float(self.loglike), "deviance": float(self.deviance),
        "endogenous": tuple(self.endog_names), "excluded_instruments": tuple(self.instrument_names),
        "weight_type": self.weight_type, "sum_weights": self.sum_weights,
    }

def _publication_output(self, *, level=None, include_diagnostics=None, include_profile=False):
    out = {"coefficients": self.coef_table(level), "model": self.model_stats()}
    show_diag = self.diagnostics_mode != "off" if include_diagnostics is None else bool(include_diagnostics)
    if show_diag: out["diagnostics"] = self.diagnostics
    if include_profile and self.profile is not None: out["profile"] = self.profile
    out["reproducibility"] = self.reproducibility or reproducibility_dict()
    return out

IVPPMLResult.coef_table = _coef_table
IVPPMLResult.pvalues = property(_pvalues)
IVPPMLResult.conf_int = _conf_int
IVPPMLResult.model_stats = _model_stats
IVPPMLResult.publication_output = _publication_output
