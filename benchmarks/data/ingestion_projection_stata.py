from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
CHILD = r'''
import json, resource, sys
from time import perf_counter
import pandas as pd
path=sys.argv[1]; mode=sys.argv[2]; wanted=tuple(sys.argv[3].split(','))
if mode!='full':
    from econhdfe.data import StataSource, materialize_required_data
t0=perf_counter()
if mode=='full':
    frame=pd.read_stata(path, convert_categoricals=False)
    extra={}
else:
    out=materialize_required_data(StataSource(path, convert_categoricals=False), wanted, memory_budget_mb=128)
    frame=out.frame
    extra={'planned_batch_rows':out.plan.batch_rows,'required_column_count':out.plan.projected_column_count}
print(json.dumps({'seconds':perf_counter()-t0,'dataframe_mb':int(frame.memory_usage(index=False,deep=True).sum())/1024**2,'peak_rss_mb':float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1024.0,**extra}))
'''
def child(path,mode,wanted):
    env=dict(os.environ); env['PYTHONPATH']=str(ROOT)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    p=subprocess.run([sys.executable,'-c',CHILD,str(path),mode,','.join(wanted)],capture_output=True,text=True,check=True,env=env)
    return json.loads(p.stdout)
def main():
    rng=np.random.default_rng(260914); n=100_000; k=48; wanted=tuple(f'x{j}' for j in range(6))
    with TemporaryDirectory() as td:
        path=Path(td)/'wide.dta'
        arr=rng.normal(size=(n,k)); pd.DataFrame(arr,columns=[f'x{j}' for j in range(k)]).to_stata(path,write_index=False,version=118); del arr
        full=child(path,'full',wanted); projected=child(path,'projected',wanted)
        out={'nobs':n,'source_columns':k,'required_columns':len(wanted),'dta_size_mb':path.stat().st_size/1024**2,'full':full,'projected':projected,'speed_ratio_full_over_projected':full['seconds']/projected['seconds'],'payload_reduction':1-projected['dataframe_mb']/full['dataframe_mb'],'peak_rss_reduction':1-projected['peak_rss_mb']/full['peak_rss_mb']}
        print(json.dumps(out,indent=2)); Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__': main()
