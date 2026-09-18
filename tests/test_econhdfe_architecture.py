from __future__ import annotations

import ast
from pathlib import Path
import numpy as np
import pandas as pd


def test_primary_and_legacy_estimator_names_are_consistent():
    import econhdfe
    import pyreghdfe

    assert econhdfe.reghdfe is econhdfe.olshdfe
    assert econhdfe.ivreghdfe is econhdfe.ivhdfe
    assert pyreghdfe.reghdfe is econhdfe.olshdfe
    assert pyreghdfe.ppmlhdfe is econhdfe.ppmlhdfe


def test_new_namespace_smoke_ols_and_iv():
    from econhdfe import olshdfe, ivhdfe

    rng = np.random.default_rng(918)
    n = 400
    firm = np.arange(n) % 40
    year = np.arange(n) % 8
    z = rng.normal(size=n)
    x = 0.7 * z + rng.normal(size=n)
    c = rng.normal(size=n)
    y = 1.25 * x - 0.3 * c + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x": x, "z": z, "c": c, "firm": firm, "year": year})

    r1 = olshdfe(df, y="y", x=["x", "c"], absorb=["firm", "year"])
    r2 = ivhdfe(df, y="y", exog=["c"], endog=["x"], instruments=["z"], absorb=["firm", "year"])
    assert np.all(np.isfinite(r1.params))
    assert np.all(np.isfinite(r2.params))


def test_dependency_boundaries_are_one_way():
    root = Path(__file__).parents[1] / "econhdfe"

    data_text = "\n".join(p.read_text() for p in (root / "data").rglob("*.py"))
    assert "..models" not in data_text
    assert "..iv" not in data_text
    assert "..hdfe" not in data_text

    compute_text = "\n".join(p.read_text() for p in (root / "compute").rglob("*.py"))
    assert "..hdfe" not in compute_text
    assert "..iv" not in compute_text
    assert "..models" not in compute_text
    assert "..data" not in compute_text

    planner_text = "\n".join(p.read_text() for p in (root / "planner").rglob("*.py"))
    assert "..data" not in planner_text
    assert "..compute" not in planner_text
    assert "..hdfe" not in planner_text
    assert "..iv" not in planner_text
    assert "..models" not in planner_text

    hdfe_text = "\n".join(p.read_text() for p in (root / "hdfe").rglob("*.py"))
    assert "..iv" not in hdfe_text
    assert "..models" not in hdfe_text

    iv_text = "\n".join(p.read_text() for p in (root / "iv").rglob("*.py"))
    assert "..models" not in iv_text


def test_legacy_linear_module_is_a_thin_compatibility_layer():
    from pyreghdfe.linear import fit_ols, fit_iv_2sls
    from econhdfe.models.ols import _fit_ols
    from econhdfe.models.linear_iv.estimators import fit_iv_2sls as new_fit_iv_2sls

    assert fit_ols is _fit_ols
    assert fit_iv_2sls is new_fit_iv_2sls


def test_generic_iv_uses_public_compute_primitives_only():
    root = Path(__file__).parents[1] / "econhdfe" / "iv"
    text = "\n".join(p.read_text() for p in root.rglob("*.py"))
    assert "._as2d" not in text
    assert "._weighted_arrays" not in text


def _relative_import_targets(package_dir: Path) -> set[str]:
    targets: set[str] = set()
    for path in package_dir.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                module = node.module or ""
                targets.add("." * node.level + module)
    return targets


def test_ivppml_dependency_direction_is_model_level_only():
    root = Path(__file__).parents[1] / "econhdfe" / "models"
    ppml_text = "\n".join(p.read_text() for p in (root / "ppml").rglob("*.py"))
    linear_iv_text = "\n".join(p.read_text() for p in (root / "linear_iv").rglob("*.py"))
    assert "ppml_iv" not in ppml_text
    assert "ppml_iv" not in linear_iv_text

    targets = _relative_import_targets(root / "ppml_iv")
    # The combined estimator may compose lower layers and ordinary PPML
    # primitives, but generic lower layers must never depend on it.
    assert any("ppml" in target for target in targets)
    assert any("iv" in target for target in targets)
    assert any("hdfe" in target for target in targets)
