from __future__ import annotations
import argparse, time
import numpy as np
from pyreghdfe import ivreghdfe

p = argparse.ArgumentParser()
p.add_argument('--n', type=int, default=300_000)
a = p.parse_args()
rng = np.random.default_rng(123)
n=a.n
g1=rng.integers(0,max(n//10,2),n,dtype=np.int32)
g2=rng.integers(0,1000,n,dtype=np.int32)
z1=rng.normal(size=n); z2=rng.normal(size=n); c=rng.normal(size=n)
v=rng.normal(size=n); e=.5*v+rng.normal(size=n)
x=.9*z1+.5*z2+.25*c+v
fe1=rng.normal(scale=.5,size=int(g1.max())+1); fe2=rng.normal(scale=.5,size=1000)
y=1.5*x+.4*c+fe1[g1]+fe2[g2]+e
for est in ('2sls','liml','gmm2s'):
    t=time.perf_counter()
    r=ivreghdfe(
        y=y, exog=c, endog=x, instruments=np.column_stack([z1,z2]),
        absorb=[g1,g2], vce='robust', drop_singletons=False, estimator=est,
    )
    print(est, {
        'seconds': time.perf_counter()-t,
        'params': r.params.tolist(),
        'iters': r.iterations,
        'kappa': r.kappa,
        'cragg_donald_f': r.diagnostics['cragg_donald_f'],
        'overid_pvalue': r.diagnostics['overidentification']['pvalue'],
    })
