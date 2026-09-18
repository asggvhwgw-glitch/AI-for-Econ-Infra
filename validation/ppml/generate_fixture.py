from pathlib import Path
import numpy as np
import pandas as pd


def main(outdir=None):
    out = Path(outdir or Path(__file__).resolve().parent / "fixtures")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260911)

    G1, G2, R = 40, 20, 4
    g1 = np.repeat(np.arange(G1), G2 * R)
    g2 = np.tile(np.repeat(np.arange(G2), R), G1)
    rep = np.tile(np.arange(R), G1 * G2)
    n = len(g1)
    g3 = (7 * g1 + 3 * g2 + rep) % 17
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    a = rng.normal(scale=.18, size=G1)
    b = rng.normal(scale=.15, size=G2)
    c = rng.normal(scale=.10, size=17)
    eta = .15 * x1 - .09 * x2 + a[g1] + b[g2] + c[g3]
    y = rng.poisson(np.exp(eta))
    expo = np.exp(rng.normal(scale=.35, size=n))
    yexp = rng.poisson(expo * np.exp(eta))
    df = pd.DataFrame({
        "y": y.astype(float), "yexp": yexp.astype(float),
        "x1": x1, "x2": x2, "g1": g1, "g2": g2, "g3": g3,
        "cl1": g1, "cl2": g2, "expo": expo,
    })
    df.to_csv(out / "ppml_fixture.csv", index=False)
    df.to_stata(out / "ppml_fixture.dta", write_index=False, version=118)

    sep = pd.DataFrame({
        "y": [0., 1., 0., 0., 1.],
        "id1": [1, 1, 2, 2, 2],
        "id2": [1, 1, 1, 2, 2],
    })
    sep.to_stata(out / "separation_primer.dta", write_index=False, version=118)
    return out


if __name__ == "__main__":
    print(main())
