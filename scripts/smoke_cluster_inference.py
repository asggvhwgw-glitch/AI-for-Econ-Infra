from __future__ import annotations
import numpy as np
import pandas as pd
from econhdfe import olshdfe, cluster_diagnostics, wild_cluster_test_ols

rng = np.random.default_rng(460)
n = 320
cluster = np.arange(n) % 8
fe = np.arange(n) % 20
x = rng.normal(size=n)
y = 0.25 * x + rng.normal(size=20)[fe] + rng.normal(scale=0.5, size=8)[cluster] + rng.normal(size=n)
df = pd.DataFrame({"y": y, "x": x, "fe": fe, "cluster": cluster})
fit = olshdfe(df, y="y", x=["x"], absorb=["fe"], cluster="cluster", vce="cluster", keep_state=True, drop_singletons=False)
diag = cluster_diagnostics(fit)
out = wild_cluster_test_ols(fit, param="x", reps=999, batch_size=31, seed=7)
assert diag.cluster_counts == (8,)
assert out.full_enumeration and out.reps == 256 and 0 <= out.pvalue <= 1
print(f"cluster inference smoke PASS; G={diag.cluster_counts[0]}; WCR reps={out.reps}; p={out.pvalue:.6g}")
