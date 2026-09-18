"""Small installed-wheel smoke for v0.6 categorical FE recovery."""
from __future__ import annotations
import numpy as np
from econhdfe.effects import (
    NormalizationSpec,
    FixedEffectIdentificationError,
    recover_fixed_effects,
)

# Connected two-way indicator FE.
worker = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
firm = np.array(["x", "y", "y", "z", "z", "x"], dtype=object)
a = {"a": 0.4, "b": -0.2, "c": 0.1}
p = {"x": 0.3, "y": -0.1, "z": 0.2}
target = np.array([a[i] + p[j] for i, j in zip(worker, firm)], dtype=float)
res = recover_fixed_effects(target, [worker, firm], names=["worker", "firm"])
rec = res.term("worker").coefficients[res.term("worker").component * 0] if False else None
# Reconstruct through level lookup without materializing persistent observation arrays.
wa = {k: v for k, v in zip(res.term("worker").levels.tolist(), res.term("worker").coefficients.tolist())}
fa = {k: v for k, v in zip(res.term("firm").levels.tolist(), res.term("firm").coefficients.tolist())}
reconstructed = np.array([wa[i] + fa[j] for i, j in zip(worker, firm)])
assert np.max(np.abs(reconstructed - target)) < 1e-8
ref = res.renormalize(NormalizationSpec("reference", baseline="worker", references={"firm": "x"}))
wa2 = {k: v for k, v in zip(ref.term("worker").levels.tolist(), ref.term("worker").coefficients.tolist())}
fa2 = {k: v for k, v in zip(ref.term("firm").levels.tolist(), ref.term("firm").coefficients.tolist())}
reconstructed2 = np.array([wa2[i] + fa2[j] for i, j in zip(worker, firm)])
assert np.max(np.abs(reconstructed2 - target)) < 1e-8
assert abs(fa2["x"]) < 1e-10

# Connected 3-way incidence but extra nullity from no worker-firm mobility.
w = np.array(["a", "a", "b", "b"], dtype=object)
f = np.array(["x", "x", "y", "y"], dtype=object)
t = np.array([1, 2, 1, 2])
g = np.array([1.0, 1.2, 0.5, 0.7])
try:
    recover_fixed_effects(g, [w, f, t], names=["worker", "firm", "year"])
except FixedEffectIdentificationError as exc:
    assert exc.code in {"identification.fe_data_rank_deficiency", "identification.fe_extra_nullity"}
else:
    raise AssertionError("extra-nullity design must fail strict FE recovery")

print("econhdfe effects smoke PASS")

# Two independent 3-way components: preserve the identified block even when the
# second connected block has one extra null direction.
good = np.array(np.meshgrid(np.arange(2), np.arange(2), np.arange(2), indexing="ij")).reshape(3, -1).T
bad = np.array([[2, 2, 2], [2, 2, 3], [3, 3, 2]], dtype=int)
edges = np.vstack([good, bad])
a3, b3, c3 = edges.T
target3 = 0.4 * a3 - 0.2 * b3 + 0.1 * c3
partial = recover_fixed_effects(target3, [a3, b3, c3], names=["a", "b", "c"], solver="lsmr")
assert len(partial.identified_components) == 1
assert len(partial.unidentified_components) == 1
obs_comp = partial.identification.component_by_term[0][a3]
good_obs = np.isin(obs_comp, partial.identified_components)
reconstructed3 = np.zeros(len(edges))
reconstructed3[:] = np.nan
reconstructed3[good_obs] = sum(
    term.coefficients[g[good_obs]] for term, g in zip(partial.terms, [a3, b3, c3], strict=False)
)
assert np.max(np.abs(reconstructed3[good_obs] - target3[good_obs])) < 1e-8
assert all(np.all(np.isnan(t.coefficients[~t.identified])) for t in partial.terms)
