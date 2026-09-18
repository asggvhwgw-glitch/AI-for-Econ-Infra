from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(240426)
n = 6000
g1 = rng.integers(0, 120, n)
g2 = rng.integers(0, 50, n)
c = rng.normal(size=n)
z = rng.normal(size=n)
z2 = rng.normal(size=n)
z3 = rng.normal(size=n)
e = .78*z + .18*c + rng.normal(scale=.55, size=n)
e2 = .65*z2 + .2*z3 - .1*c + rng.normal(scale=.6, size=n)
a = rng.normal(scale=.16, size=120)
b = rng.normal(scale=.12, size=50)
offset = rng.normal(scale=.15, size=n)
expo = np.exp(offset)
mu = np.exp(a[g1] + b[g2] + .10*c + .26*e - .12*e2)
y = rng.poisson(mu)
yexp = rng.poisson(expo * mu)
fw = rng.integers(1, 4, size=n)
pw = np.exp(rng.normal(scale=.35, size=n))
# Add one deep-tail zero useful for separation(all)/mu diagnostics.
z[-1] = 35.0; e[-1] = 35.0; y[-1] = 0; yexp[-1] = 0

df = pd.DataFrame(dict(y=y, yexp=yexp, c=c, e=e, e2=e2, z=z, z2=z2, z3=z3,
                       g1=g1+1, g2=g2+1, offset=offset, expo=expo, fw=fw, pw=pw,
                       cl1=g1+1, cl2=g2+1))
(ROOT/'fixtures').mkdir(exist_ok=True)
df.to_csv(ROOT/'fixtures'/'ivppml_feature_fixture.csv', index=False)
df.to_stata(ROOT/'fixtures'/'ivppml_feature_fixture.dta', write_index=False, version=118)
print(ROOT/'fixtures'/'ivppml_feature_fixture.dta')
