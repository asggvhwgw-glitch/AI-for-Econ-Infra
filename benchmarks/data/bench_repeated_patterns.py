from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np,pandas as pd
from econhdfe import OLSHDFESession
from econhdfe.config import ExecutionConfig
from econhdfe.data import EncodedEconometricDataset


def build(n=150_000,seed=50031):
 rng=np.random.default_rng(seed); firm=np.arange(n)%15_000; year=(np.arange(n)//15_000)%10; industry=firm%30
 d={'firm':firm,'year':year,'industry':industry}
 for j in range(10): d[f'x{j}']=rng.normal(size=n)
 common=rng.normal(size=n)
 for j in range(6): d[f'y{j}']=.25*d[f'x{j%10}']+.1*common+rng.normal(size=n)
 return pd.DataFrame(d)

def run(data,specs,cfg):
 s=OLSHDFESession(data,execution_config=cfg,drop_singletons=False)
 t=time.perf_counter()
 for y,x,fe in specs:s.fit(y=y,x=x,absorb=fe)
 return time.perf_counter()-t,s.cache_info()

def main():
 df=build(); cfg=ExecutionConfig(cache_validation='signature',threads=1,memory_budget_mb=512)
 t=time.perf_counter();ds=EncodedEconometricDataset.from_frame(df,identifier_columns=['firm','year','industry']);build_s=time.perf_counter()-t
 patterns={
  'change_y':[(f'y{r%6}',['x0','x1','x2'],['firm','year']) for r in range(30)],
  'add_drop_x':[(f'y{r%3}',[f'x{j}' for j in range(2+(r%8))],['firm','year']) for r in range(30)],
  'change_fe':[(f'y{r%3}',['x0','x1','x2'],(['firm'] if r%3==0 else ['firm','year'] if r%3==1 else ['firm','year','industry'])) for r in range(30)],
 }
 out={'nobs':len(df),'specifications_each':30,'encoded_build_seconds':build_s,'patterns':{}}
 for name,specs in patterns.items():
  td,idf=run(df,specs,cfg);te,ie=run(ds,specs,cfg)
  out['patterns'][name]={'dataframe_seconds':td,'encoded_seconds':te,'speedup':td/te,'dataframe_cache':idf,'encoded_cache':ie}
 print(json.dumps(out,indent=2,default=str));Path('benchmarks/data/repeated_patterns_encoded_dataset.json').write_text(json.dumps(out,indent=2,default=str))
if __name__=='__main__':main()
