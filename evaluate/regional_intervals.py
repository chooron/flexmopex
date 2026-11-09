#!/usr/bin/env python3
"""Region-stratified basin bootstrap with full-data evidence as BA reference."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
INPUT = Path(__file__).resolve().parents[1] / 'inputs/evidence/rf_loro_classes_fullref_verified.csv'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 inputs=Path(__file__).resolve().parents[1]/'inputs';target=args.output.resolve()
 protected=[inputs.resolve()]+[(inputs/p).resolve() for p in ('evidence','trained','synthetic')]
 if any(target==base or base in target.parents for base in protected):raise ValueError('output overlaps inputs')
 d=pd.read_csv(INPUT); rng=np.random.default_rng(20260516); rows=[]
 code={'decisive_closed':0,'uncertain':1,'decisive_open':2}
 for p in ('w_snow','w_sub','w_phen','w_int'):
  q=d[d.process==p]
  for region in range(8):
   x=q if region==0 else q[q.region==region]
   a=x.evidence_class_v2.map(code).to_numpy(); b=x.loro_class.map(code).to_numpy(); r=x.region.to_numpy()
   ids=[np.flatnonzero(r==k) for k in np.unique(r)]
   out=[]
   for _ in range(1000):
    take=np.concatenate([rng.choice(i,len(i),replace=True) for i in ids]); aa=a[take]; bb=b[take]
    mask=(aa!=1)&(bb!=1); on=(aa==2)&mask; off=(aa==0)&mask
    ba=(np.mean(bb[on]==2)+np.mean(bb[off]==0))/2 if on.any() and off.any() else np.nan
    out.append((np.mean(aa==bb),ba))
   vals=np.asarray(out)
   for j,metric in enumerate(('exact_agreement','direction_balanced_accuracy')):
    valid=vals[:,j][np.isfinite(vals[:,j])]
    rows.append(dict(process=p,scope='pooled' if not region else f'R{region}',metric=metric,ci_lo=np.quantile(valid,.025) if len(valid) else np.nan,ci_hi=np.quantile(valid,.975) if len(valid) else np.nan,n_valid=len(valid),bootstrap='1000 basin resamples, within-region stratification',seed=20260516))
 with args.output.open('x') as f:pd.DataFrame(rows).to_csv(f,index=False)
 print('bootstrap rows',len(rows))
if __name__=='__main__':main()
