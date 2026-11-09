#!/usr/bin/env python3
"""Fixed-off evaluation of frozen Full RF_S0 checkpoints; no training or writes to runs."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
V2=Path(__file__).resolve().parents[1]
os.chdir(V2);sys.path.insert(0,str(V2));os.environ.setdefault('DATA_PATH',str(V2/'data'))
from train.run_model import _build_data_loader,FlexMopexSampler
from train.local_model_handler import FlexMopexModelHandler
from train.models.mopex import mopex_step
from dmg.core.calc.metrics import Metrics
ROOT=V2/'inputs'/'trained'/'normal'/'full'/'ref'
PROC=('phen','int','snow','sub')

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
 return h.hexdigest()

def metric(pred,obs):
 pred=np.asarray(pred,dtype=np.float64);obs=np.asarray(obs,dtype=np.float64)
 ok=np.isfinite(pred)&np.isfinite(obs);p=pred[ok];o=obs[ok]
 if len(o)<2: return np.nan,np.nan,len(o)
 return float(Metrics._nse_r2(p,o,o.mean())),float(Metrics._kge(p.mean(),o.mean(),p.std(),o.std(),Metrics._corr(p,o))),len(o)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=(42,43,44),required=True)
 ap.add_argument('--output',type=Path,required=True);ap.add_argument('--n-basins',type=int,default=671)
 ap.add_argument('--batch',type=int,default=100);ap.add_argument('--device',default='cuda:0');ap.add_argument('--smoke',action='store_true')
 ap.add_argument('--compiled',action='store_true',help='use the formal torch.compile step instead of eager step')
 ap.add_argument('--diag-output',type=Path,required=True,help='new CSV for basin-wise replay diagnostics')
 ap.add_argument('--batched-endpoints',action='store_true',help='independent vectorized endpoint sensitivity check')
 a=ap.parse_args();assert not a.output.exists() and not a.diag_output.exists();assert 1<=a.n_basins<=671
 protected=[(V2/p).resolve() for p in ('data','inputs','inputs/evidence','inputs/trained','inputs/synthetic')]
 for path in (a.output.resolve(),a.diag_output.resolve()):
  if any(path==base or base in path.parents for base in protected):raise ValueError(f'output overlaps input data: {path}')
 assert a.output.resolve()!=a.diag_output.resolve()
 if a.smoke:assert a.n_basins<=4
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 run=ROOT/f'seed_{a.seed}';mf=json.loads((run/'manifest.json').read_text());ck=run/'model/best_checkpoint.pt'
 digest=sha(ck);assert digest==mf['checkpoints']['best']['sha256']
 cfg=json.loads((run/'config_snapshot.json').read_text());assert cfg['model']['phy']['name']==['FixedWeightMopex']
 assert cfg['model']['phy']['interception_semantics']=='RF_S0'
 cfg.update(device=a.device,dtype=torch.float32,mode='test',model_dir=str(run/'model'))
 cfg['test']['test_epoch']=0;cfg['observations']['data_path']=str(V2/'data/camels_dataset');cfg['observations']['gage_info']=str(V2/'data/gage_id.npy');cfg['basin_manifest']=str(V2/'data/gage_id.npy');cfg['model']['phy']['disable_compile']=not a.compiled
 assert (run/'model/normalization_statistics.json').is_file()
 m=FlexMopexModelHandler(cfg,device=a.device);dpl=next(iter(m.model_dict.values()));dpl.load_state_dict(torch.load(ck,map_location=a.device,weights_only=True),strict=True);dpl.eval();phy=dpl.phy_model
 import inspect
 assert inspect.unwrap(phy.step_fn) is mopex_step and torch.equal(phy.fixed_weight_values,torch.ones_like(phy.fixed_weight_values))
 assert bool(getattr(phy,'_torch_compile_enabled',False))==a.compiled
 ds=_build_data_loader(cfg).eval_dataset;sampler=FlexMopexSampler(cfg)
 ids=np.load(V2/'data/gage_id.npy').astype(int).reshape(-1);assert len(ids)==671
 scored=run/mf['scored_windows'][0];scored=next(p for p in (run/x for x in mf['scored_windows']) if p.name.startswith('test'))
 ref=np.load(scored/'streamflow.npy',mmap_mode='r');refobs=np.load(scored/'streamflow_obs.npy',mmap_mode='r')
 stored=json.loads((scored/'metrics.json').read_text());stored=json.loads(stored) if isinstance(stored,str) else stored
 assert ref.shape==(5110,671,1)
 rows=[];diagnostics=[];maxdiff=0.;maxmetric=0.
 for start in range(0,a.n_basins,a.batch):
  end=min(start+a.batch,a.n_basins);n=end-start
  sample=sampler.get_validation_sample(ds,start,end);sample=dpl._ensure_static_features(sample)
  with torch.inference_mode():
   pars=dpl.nn_model(sample)
   w=torch.ones((5,4),device=a.device)
   for j in range(4): w[j+1,j]=0
   assert (w[0]==1).all() and (w[1:].sum(1)==3).all()
   chunks=[];total=sample['x_phy'].shape[0];rho=cfg['delta_model']['rho'];warm=cfg['delta_model']['phy_model']['warm_up']
   assert rho==warm==365 and cfg['test']['split_dataset']
   for t in range(0,total-rho-warm+1,rho):
    window={k:v[t:t+rho+warm,...] if len(v.shape)>2 else v for k,v in sample.items()}
    if a.batched_endpoints:
     mp=phy._descale_mopex_params(pars['params']);rp=phy._descale_routing_params(pars['gamma_uh'])
     pp={k:v.repeat(5,1) for k,v in mp.items()};rr={k:v.repeat(5) for k,v in rp.items()}
     inp={'x_phy':window['x_phy'].repeat(1,5,1),'doy':window['doy'].repeat(1,5,1)}
     P,T,PET,doy,steps,_=phy._prepare_forcings(inp)
     q=phy._run_weighted_loop(P,T,PET,doy,pp,w.repeat_interleave(n,dim=0),steps,5*n)
     pred=phy._apply_routing(q.mean(-1),rr)[:,:,0].reshape(-1,5,n)
    else:
     output=[]
     for j in range(5):
      phy.fixed_weight_values.copy_(w[j]);output.append(phy(window,pars)['streamflow'][:,:,0])
     phy.fixed_weight_values.fill_(1)
     pred=torch.stack(output,dim=1)
    assert torch.isfinite(pred).all();chunks.append(pred.cpu().numpy())
  preds=np.concatenate(chunks,axis=0);assert preds.shape==(5110,5,n)
  obs=ds['target'][warm:warm+5110,start:end,0].cpu().numpy()
  refpiece=np.asarray(ref[:,start:end,0]);d=float(np.max(np.abs(preds[:,0,:]-refpiece)));maxdiff=max(maxdiff,d)
  assert refobs.shape==(total,671)
  obsref=np.asarray(refobs[warm:warm+5110,start:end]);assert obsref.shape==obs.shape and np.allclose(obs,obsref,equal_nan=True,atol=1e-6)
  for i in range(n):
   fulln,fullk,days=metric(preds[:,0,i],obs[:,i]);idx=start+i
   maxmetric=max(maxmetric,abs(fulln-stored['nse'][idx]),abs(fullk-stored['kge'][idx]))
   err=preds[:,0,i].astype(float)-refpiece[:,i].astype(float); ae=np.abs(err); scale=np.abs(refpiece[:,i].astype(float)); imax=int(np.argmax(ae)); rel=ae[scale>0.1]/scale[scale>0.1]
   diagnostics.append(dict(gauge_id=int(ids[idx]),seed=a.seed,nse_replay=fulln,kge_replay=fullk,nse_frozen=float(stored['nse'][idx]),kge_frozen=float(stored['kge'][idx]),nse_difference=fulln-float(stored['nse'][idx]),kge_difference=fullk-float(stored['kge'][idx]),mean_signed_error=float(np.mean(err)),mae=float(np.mean(ae)),rmse=float(np.sqrt(np.mean(err**2))),abs_q50=float(np.quantile(ae,.5)),abs_q95=float(np.quantile(ae,.95)),abs_q99=float(np.quantile(ae,.99)),max_abs=float(ae[imax]),max_day_index=imax,max_day_frozen_flow=float(refpiece[imax,i]),relative_q95_flow_gt_0p1=float(np.quantile(rel,.95)) if len(rel) else np.nan,mae_first_half=float(np.mean(ae[:2555])),mae_second_half=float(np.mean(ae[2555:])),n_valid=days))
   for j,p in enumerate(PROC):
    offn,offk,valid=metric(preds[:,j+1,i],obs[:,i]);assert valid==days
    rows.append(dict(gauge_id=int(ids[idx]),seed=a.seed,process=p,nse_full=fulln,kge_full=fullk,nse_off=offn,kge_off=offk,delta_nse=fulln-offn,delta_kge=fullk-offk,n_valid=valid,checkpoint_sha256=digest,config_sha256=mf['config']['config_sha256'],scored_days=5110))
  print('batch',a.seed,start,end,'full_prediction_max_abs',d,'metric_max_abs_so_far',maxmetric,flush=True)
 assert np.isfinite(maxdiff) and np.isfinite(maxmetric)
 print('REPLAY_DIAGNOSTIC',a.seed,'max_prediction_diff',maxdiff,'max_metric_diff',maxmetric,'historical_1e-4_threshold_exceeded',maxdiff>=1e-4,flush=True)
 df=pd.DataFrame(rows);assert len(df)==4*a.n_basins and np.isfinite(df[['nse_full','kge_full','nse_off','kge_off']]).all().all()
 with a.output.open('x') as f:df.to_csv(f,index=False)
 with a.diag_output.open('x') as f:pd.DataFrame(diagnostics).to_csv(f,index=False)
 print('EXECUTED_NOT_YET_ACCEPTED',a.seed,'rows',len(df),flush=True)
if __name__=='__main__':main()
