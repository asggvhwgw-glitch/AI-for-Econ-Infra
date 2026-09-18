from __future__ import annotations
import json, time, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from econhdfe import OLSHDFESession
from econhdfe.config import ExecutionConfig
from econhdfe.data import CSVSource


def main(n=120_000, raw_columns=64, specs_count=40, seed=90211):
    rng=np.random.default_rng(seed)
    firm=np.arange(n)%12_000
    year=(np.arange(n)//12_000)%10
    data={"firm":firm.astype(str), "year":year}
    for j in range(8): data[f"x{j}"]=rng.normal(size=n)
    common=rng.normal(size=n)
    for j in range(5): data[f"y{j}"]=0.25*data[f"x{j%8}"]+0.15*common+rng.normal(scale=.8,size=n)
    for j in range(raw_columns-len(data)): data[f"unused{j}"]=rng.normal(size=n)
    frame=pd.DataFrame(data)
    specs=[]; role_specs=[]
    for r in range(specs_count):
        k=2+(r%6); y=f"y{r%5}"; x=[f"x{j}" for j in range(k)]
        specs.append((y,x)); role_specs.append({"y":y,"x":x,"absorb":["firm","year"]})
    cfg=ExecutionConfig(cache_validation="signature",threads=1,memory_budget_mb=512)

    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/"wide.csv"; frame.to_csv(path,index=False)
        fit_specs=[{"y": y, "x": x, "absorb": ["firm", "year"]} for y,x in specs]
        t=time.perf_counter(); full=pd.read_csv(path); sess=OLSHDFESession(full,execution_config=cfg,drop_singletons=False)
        sess.fit_many(fit_specs)
        baseline=time.perf_counter()-t
        full_payload=int(full.memory_usage(index=False,deep=True).sum())

        t=time.perf_counter(); sess2=OLSHDFESession(CSVSource(path),execution_config=cfg,drop_singletons=False)
        sess2.fit_many(fit_specs)
        optimized=time.perf_counter()-t
        ds=sess2.dataset

        out={
            "nobs":n,"raw_columns":raw_columns,"specifications":specs_count,
            "required_columns":len(ds.columns),"baseline_full_read_plus_session_seconds":baseline,
            "encoded_projected_plus_session_seconds":optimized,"workflow_speedup":baseline/optimized,
            "full_dataframe_bytes":full_payload,"encoded_dataset_bytes":ds.encoded_bytes,
            "payload_reduction_fraction":1-ds.encoded_bytes/full_payload,
            "session_cache":sess2.cache_info(),
        }
        print(json.dumps(out,indent=2,default=str))
        Path("benchmarks/data/repeated_workflow_encoded_dataset.json").write_text(json.dumps(out,indent=2,default=str))

if __name__ == "__main__": main()
