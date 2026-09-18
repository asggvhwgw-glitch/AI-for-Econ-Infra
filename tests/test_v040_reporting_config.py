import numpy as np
import pandas as pd

from econhdfe import (
    olshdfe, ivhdfe, ppmlhdfe, ivppmlhdfe,
    HDFEConfig, InferenceConfig, ExecutionConfig,
    PPMLConfig, IVPPMLConfig, preflight_dataframe,
)
from econhdfe.hdfe.plan import FEPlan
from econhdfe.compute.context import ExecutionContext


def _linear_df(seed=101, n=1200):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, 80, n)
    year = rng.integers(0, 12, n)
    z = rng.normal(size=n)
    c = rng.normal(size=n)
    x = 0.8 * z + 0.2 * c + rng.normal(size=n)
    y = 1.3 * x + 0.4 * c + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "c": c, "z": z, "firm": firm, "year": year})


def test_ols_publication_output_has_top5_core_fields_and_hides_diagnostics():
    df = _linear_df()
    r = olshdfe(df, y="y", x=["x", "c"], absorb=["firm", "year"], vce="robust")
    tab = r.coef_table()
    assert list(tab.columns) == ["estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high", "stars"]
    assert np.all(np.isfinite(tab[["estimate", "std_error", "p_value"]].to_numpy()))
    out = r.publication_output()
    assert "diagnostics" not in out
    stats = out["model"]
    for key in ("nobs", "df_resid", "df_absorbed", "vce", "fixed_effects", "r2", "r2_adjusted", "r2_within", "r2_adjusted_within"):
        assert key in stats
    assert stats["vce"] == "robust"
    assert tuple(stats["fixed_effects"]) == ("firm", "year")
    assert "reproducibility" in out


def test_iv_publication_output_keeps_reportable_first_stage_not_fitted_matrix():
    df = _linear_df()
    r = ivhdfe(df, y="y", exog=["c"], endog=["x"], instruments=["z"], absorb=["firm", "year"])
    out = r.publication_output()
    assert "diagnostics" not in out
    assert "first_stage" in out
    assert "fitted_endog" not in out["first_stage"]
    assert "diagnostics" in out["first_stage"]
    out2 = r.publication_output(include_diagnostics=True)
    assert "diagnostics" in out2


def test_common_configs_control_strategy_reporting_and_profile():
    df = _linear_df(n=800)
    r = olshdfe(
        df, y="y", x=["x"], absorb=["firm", "year"],
        hdfe_config=HDFEConfig(solver="map", tolerance=1e-9, max_iter=4000, dof_method="firstpair"),
        inference_config=InferenceConfig(vce="robust", confidence_level=0.90, diagnostics="off"),
        execution_config=ExecutionConfig(threads=1, memory_budget_mb=128, profile="summary"),
    )
    assert r.vce == "robust"
    assert r.confidence_level == 0.90
    assert r.profile is not None and r.profile["total_seconds"] >= 0
    assert r.absorb_info["solver_selection"]["requested"] == "map"
    ci90 = r.conf_int()
    ci95 = r.conf_int(0.95)
    assert np.all((ci90[:, 1] - ci90[:, 0]) < (ci95[:, 1] - ci95[:, 0]))


def test_ppml_and_ivppml_publication_outputs_are_model_specific():
    rng = np.random.default_rng(202)
    n = 1800
    f = rng.integers(0, 90, n)
    t = rng.integers(0, 10, n)
    z = rng.normal(size=n)
    x = 0.7 * z + rng.normal(size=n)
    c = rng.normal(size=n)
    mu = np.exp(0.15 * x + 0.1 * c + 0.03 * (f % 5))
    y = rng.poisson(mu)
    pp = ppmlhdfe(y, np.column_stack([x, c]), absorb=[f, t], names=["x", "c"],
                  config=PPMLConfig(engine="optimized"))
    pout = pp.publication_output()
    assert "diagnostics" not in pout
    for key in ("nobs", "df_absorbed", "loglike", "deviance", "fixed_effects"):
        assert key in pout["model"]

    iv = ivppmlhdfe(
        y, exog=c[:, None], endog=x[:, None], instruments=z[:, None], absorb=[f, t],
        exog_names=["c"], endog_names=["x"], instrument_names=["z"],
        config=IVPPMLConfig(engine="optimized"),
    )
    iout = iv.publication_output()
    assert "diagnostics" not in iout
    assert iout["model"]["endogenous"] == ("x",)
    assert iout["model"]["excluded_instruments"] == ("z",)


def test_strict_preflight_warns_but_does_not_turn_warnings_into_errors():
    n = 200
    df = pd.DataFrame({
        "y": np.r_[np.zeros(199), 1.0],
        "z": np.ones(n) + np.linspace(0, 1e-14, n),
        "cluster": np.arange(n) % 8,
        "w": np.geomspace(1.0, 1e8, n),
    })
    report = preflight_dataframe(
        df, {"y": "outcome", "z": "instrument", "cluster": "cluster", "w": "weight"},
        model="ivppml", level="strict",
    )
    assert report.ok
    codes = {x.code for x in report.warnings}
    assert "warning.extreme_zero_share" in codes
    assert "warning.near_constant" in codes
    assert "warning.few_clusters" in codes
    assert "warning.extreme_weights" in codes


def test_feplan_fingerprint_and_execution_context_cache_are_content_based():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([0, 1, 0, 1, 0, 1])
    p1 = FEPlan.from_arrays([a, b])
    p2 = FEPlan.from_arrays([a.copy(), b.copy()])
    p3 = FEPlan.from_arrays([a, np.array([0, 1, 0, 1, 1, 0])])
    assert p1.fingerprint == p2.fingerprint
    assert p1.fingerprint != p3.fingerprint
    ctx = ExecutionContext(ExecutionConfig(cache="on"))
    ctx.put(p1.fingerprint, p1)
    assert ctx.get(p2.fingerprint) is p1
    assert ctx.get(p3.fingerprint) is None


def test_reusable_ppml_model_invalidates_fe_plan_when_source_fe_changes():
    from econhdfe import PPMLHDFE
    rng = np.random.default_rng(404)
    n = 500
    df = pd.DataFrame({
        "y": rng.poisson(1.2, n),
        "x": rng.normal(size=n),
        "firm": rng.integers(0, 30, n),
        "year": rng.integers(0, 6, n),
    })
    m = PPMLHDFE(df, absorb=["firm", "year"], execution_config=ExecutionConfig(cache_validation="signature"))
    sig0 = m.plan.fingerprint
    _ = m.fit("y", ["x"])
    df.loc[:20, "firm"] = 999
    _ = m.fit("y", ["x"])
    assert m.plan.fingerprint != sig0
