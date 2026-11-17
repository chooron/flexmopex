#!/usr/bin/env python3
"""Pair three-seed RF_S0 fixed-off with frozen LOPO and V2 evidence classes."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

LA = Path(__file__).resolve().parents[1] / 'inputs' / 'evaluation'
OUT = None

def save(df,name):
 with (OUT/name).open('x') as f:df.to_csv(f,index=False)

def ci(x,rng,nboot=2000):
 a=np.asarray(x,float);assert len(a)>0 and np.isfinite(a).all()
 v=np.median(a[rng.integers(0,len(a),size=(nboot,len(a)))],axis=1)
 return np.quantile(v,[.025,.975])

def main():
 global OUT
 parser = argparse.ArgumentParser()
 parser.add_argument('--output', type=Path, required=True)
 args = parser.parse_args()
 OUT = args.output
 protected=[LA.parent.resolve()]+[(LA.parent/p).resolve() for p in ('evaluation','evidence','trained','synthetic')]
 target=OUT.resolve()
 if any(target==base or base in target.parents for base in protected):raise ValueError('output overlaps inputs')
 if OUT.exists() and any(OUT.iterdir()):raise FileExistsError(f'output directory is not empty: {OUT}')
 OUT.mkdir(parents=True, exist_ok=True)
 raw=pd.concat([pd.read_csv(LA/f'fixed_off_per_basin_s{s}.csv') for s in (42,43,44)],ignore_index=True)
 assert len(raw)==671*3*4 and not raw.duplicated(['gauge_id','process','seed']).any()
 assert np.allclose(raw.delta_nse,raw.nse_full-raw.nse_off) and np.allclose(raw.delta_kge,raw.kge_full-raw.kge_off)
 cl=pd.read_csv(LA/'rf_evidence_classes_comparison.csv');cl['process']=cl.process.str.removeprefix('w_')
 raw=raw.merge(cl[['gauge_id','process','evidence_class_v2']],on=['gauge_id','process'],validate='many_to_one')
 lo=pd.read_csv(LA/'rf_lopo_basinwise.csv');lo=lo.melt(id_vars=['gauge_id','process'],value_vars=['loss42','loss43','loss44'],var_name='seed',value_name='lopo_delta_nse');lo['seed']=lo.seed.str[-2:].astype(int)
 pairs=raw.merge(lo,on=['gauge_id','process','seed'],validate='one_to_one');pairs['fixed_minus_lopo']=pairs.delta_nse-pairs.lopo_delta_nse
 assert len(pairs)==len(raw)
 save(raw,'fixed_off_per_basin_seed.csv');save(pairs,'fixed_off_vs_lopo.csv')
 mean=pairs.groupby(['gauge_id','process','evidence_class_v2'],as_index=False)[['delta_nse','delta_kge','lopo_delta_nse','fixed_minus_lopo']].mean()
 rng=np.random.default_rng(20261106);rows=[]
 for p,g in mean.groupby('process'):
  for group in ('all','decisive_open','decisive_closed','uncertain'):
   q=g if group=='all' else g[g.evidence_class_v2==group]
   for metric in ('delta_nse','delta_kge','lopo_delta_nse','fixed_minus_lopo'):
    x=q[metric].to_numpy();lo_,hi_=ci(x,rng)
    rows.append(dict(process=p,group=group,metric=metric,n=len(x),median=float(np.median(x)),mean=float(np.mean(x)),ci_lo=lo_,ci_hi=hi_,positive=float(np.mean(x>0)),negative=float(np.mean(x<0)),near_zero_abs_lt_0p001=float(np.mean(abs(x)<.001)),definition='per-basin three-seed mean; 2000 basin bootstrap median; near-zero threshold exploratory'))
  for metric in ('delta_nse','delta_kge','lopo_delta_nse','fixed_minus_lopo'):
   on=g[g.evidence_class_v2=='decisive_open'][metric].to_numpy();off=g[g.evidence_class_v2=='decisive_closed'][metric].to_numpy()
   boot=np.median(on[rng.integers(len(on),size=(2000,len(on)))],axis=1)-np.median(off[rng.integers(len(off),size=(2000,len(off)))],axis=1)
   rows.append(dict(process=p,group='ON_minus_OFF',metric=metric,n=len(on)+len(off),n_on=len(on),n_off=len(off),median=float(np.median(on)-np.median(off)),ci_lo=np.quantile(boot,.025),ci_hi=np.quantile(boot,.975),definition='difference of medians, stratified group basin bootstrap 2000'))
 out=pd.DataFrame(rows);save(out,'fixed_off_summary.csv')
 print(out[(out.metric=='delta_nse')&out.group.isin(['all','ON_minus_OFF'])].to_string(index=False))
if __name__=='__main__':main()
