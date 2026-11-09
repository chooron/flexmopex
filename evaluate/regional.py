#!/usr/bin/env python3
"""Summarize held-out LORO classes as predictions of full-data V2 evidence."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr
from evidence import classify

LA = Path(__file__).resolve().parents[1] / 'inputs' / 'evidence'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    inputs = Path(__file__).resolve().parents[1] / 'inputs'
    protected = [inputs.resolve()] + [(inputs / p).resolve() for p in ('evidence', 'trained', 'synthetic')]
    if any(target == base or base in target.parents for base in protected):
        raise ValueError('output overlaps inputs')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'output directory is not empty: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    blocks=[]
    for region in range(7):
        for seed in (42,43,44):
            d=pd.read_csv(LA/f'rf_counterfactual_loro_r{region}_s{seed}.csv')
            d['region']=region+1; blocks.append(d)
    d=pd.concat(blocks,ignore_index=True)
    assert len(d)==8052 and not d.duplicated(['gauge_id','process','seed']).any()
    assert np.allclose(d.delta_J,d.L_fit_off-d.L_fit_on-d.penalty,rtol=1e-5,atol=1e-7)
    c=classify(d).rename(columns={'evidence_class':'loro_class','delta_median':'loro_delta_median'})
    regions=d[['gauge_id','region']].drop_duplicates()
    assert len(regions)==671 and not regions.gauge_id.duplicated().any()
    c=c.merge(regions,on='gauge_id',validate='many_to_one')
    full=pd.read_csv(LA/'rf_evidence_classes_comparison.csv')
    c=c.merge(full[['gauge_id','process','delta_median','evidence_class_v2']],on=['gauge_id','process'],validate='one_to_one')
    codes={'decisive_closed':0,'uncertain':1,'decisive_open':2}
    rows=[]
    for p in ('w_snow','w_sub','w_phen','w_int'):
        for r in [0,1,2,3,4,5,6,7]:
            q=c[(c.process==p) & ((c.region==r) if r else True)]
            a=q.evidence_class_v2.map(codes).to_numpy(); b=q.loro_class.map(codes).to_numpy()
            valid=(a!=1)&(b!=1); n_on=int((a[valid]==2).sum()); n_off=int((a[valid]==0).sum())
            tp=int(((a==2)&(b==2)&valid).sum()); fn=int(((a==2)&(b==0)&valid).sum())
            tn=int(((a==0)&(b==0)&valid).sum()); fp=int(((a==0)&(b==2)&valid).sum())
            assert tp+fn==n_on and tn+fp==n_off
            tpr=tp/n_on if n_on else np.nan
            tnr=tn/n_off if n_off else np.nan
            rows.append(dict(scope=f'R{r}' if r else 'pooled',process=p,n=len(q),exact_agreement=float(np.mean(a==b)),weighted_kappa=cohen_kappa_score(b,a,weights='quadratic') if len(np.unique(b))>1 else np.nan,degree_spearman=float(spearmanr(abs(q.delta_median),abs(q.loro_delta_median)).statistic),n_both_decisive=int(valid.sum()),n_target_on=n_on,n_target_off=n_off,tp=tp,tn=tn,fp=fp,fn=fn,coverage_both_decisive=float(valid.mean()),tpr=tpr,tnr=tnr,balanced_accuracy=(tpr+tnr)/2 if n_on and n_off else np.nan,comparison='loro_class_predicts_full_data_class; target=full_data; uncertain excluded from direction'))
    with (args.output/'rf_loro_classes_fullref_verified.csv').open('x') as f: c.to_csv(f,index=False)
    out=pd.DataFrame(rows)
    with (args.output/'rf_loro_evidence_agreement_fullref_verified.csv').open('x') as f: out.to_csv(f,index=False)
    print(out[out.scope=='pooled'].to_string(index=False))
if __name__=='__main__':main()
