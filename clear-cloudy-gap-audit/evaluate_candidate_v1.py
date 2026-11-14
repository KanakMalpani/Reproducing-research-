"""Evaluate the already frozen candidate once under the existing controls."""
import os,sys,json,pathlib,numpy as np,torch,rasterio,hashlib
from rasterio.windows import Window
A=pathlib.Path(__file__).resolve().parent
os.chdir(A);sys.path.insert(0,str(A))
from wf_data import scenes,NORM
results=A/'results';R=results/'finetune_v1'
final=json.loads((R/'final_selection.json').read_text());checkpoint=pathlib.Path(final['checkpoint'])
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==final['checkpoint_sha256']
assert not (R/'candidate_cloud_iou_tiles.json').exists(), 'Test already evaluated'
ns={};exec((A/'cloud_iou_pilot.py').read_text().split('BINS=')[0],ns)
m=ns['m'].cuda().eval();torch.set_num_threads(3)
w=torch.load(checkpoint,map_location='cpu',weights_only=True)
m.load_state_dict({k.removeprefix('network.'):v for k,v in w.items()},strict=True);del w
@torch.inference_mode()
def predict(raw):
 x=(raw.astype(np.float32)-NORM[:,0,None,None])/NORM[:,1,None,None]
 return (m(torch.from_numpy(x)[None].cuda())[0,1]>0).cpu().numpy()
def cm(p,w,v):return np.array([(p&w&v).sum(),(p&~w&v).sum(),(~p&w&v).sum()],np.int64)
def iou(v):return float(v[0]/sum(v)) if sum(v) else None
reference=json.loads((A/'method1_tiles.json').read_text()); refmap={(r['split'],r['scene'],r['r'],r['c']):r for r in reference}
info={'val':[],'test':[]};natural=[];terrain=[];originals={}
for split in info:
 for name,s2,gt in scenes(split):
  ts=[]
  with rasterio.open(s2) as S,rasterio.open(gt) as G:
   for r in range(0,G.height-255,512):
    for c in range(0,G.width-255,512):
     win=Window(c,r,256,256);g=G.read(window=win);v=g[1]>0
     if v.mean()<.5:continue
     f=float((g[0]==2)[v].mean());ts.append((r,c,g,f));raw=S.read(list(range(1,14)),window=win)
     p=predict(raw);w=g[1]==2;cloud=g[0]==2;key=(split,name,r,c);originals[key]=p
     counts=cm(p,w,v).tolist()
     natural.append(dict(split=split,scene=name,r=r,c=c,cloud_frac=f,all=counts,clear=cm(p,w,v&~cloud).tolist(),under_cloud=cm(p,w,v&cloud).tolist()))
     item=dict(refmap[key]);item['cm']={'base':counts};terrain.append(item)
  info[split].append(dict(name=name,s2=s2,tiles=ts))
  print('CANDIDATE NATURAL',split,name,len(natural),flush=True)
 if split=='val':
  vv={group:pooled_val for group,pooled_val in [(group,iou(np.array([z['all'] for z in natural if filt(z)]).sum(0))) for group,filt in [('all',lambda z:True),('clear',lambda z:z['cloud_frac']<.01),('cloudy',lambda z:z['cloud_frac']>=.1)]]}
  assert all(abs(vv[group]-final['chosen']['validation'][group])<1e-5 for group in vv), ('Export validation failed before test',vv)
assert set(originals)==set(refmap)
(R/'candidate_cloud_iou_tiles.json').write_text(json.dumps(natural))
(R/'candidate_method1_tiles.json').write_text(json.dumps(terrain))
def pooled(rs):return iou(np.array([r['all'] for r in rs]).sum(0))
val={g:pooled([r for r in natural if r['split']=='val' and f(r)]) for g,f in [('all',lambda r:True),('clear',lambda r:r['cloud_frac']<.01),('cloudy',lambda r:r['cloud_frac']>=.1)]}
assert all(abs(val[g]-final['chosen']['validation'][g])<1e-5 for g in val), ('Calibration export mismatch',val,final['chosen']['validation'])
import builtins,runpy
from unittest.mock import patch
real_open=builtins.open
maps={'cloud_iou_tiles.json':R/'candidate_cloud_iou_tiles.json','method1_tiles.json':R/'candidate_method1_tiles.json'}
for script,outname in [('kill_test.py','kill_test_results.json'),('ndvi_comatch_analyze.py','ndvi_comatch_results.json')]:
 def redirected(file,mode='r',*args,**kwargs):
  if isinstance(file,(str,os.PathLike)):
   name=pathlib.Path(file).name
   if any(c in mode for c in 'wax+'):
    assert name==outname;file=R/('candidate_'+outname)
   elif name in maps:file=maps[name]
  return real_open(file,mode,*args,**kwargs)
 with patch('builtins.open',redirected):runpy.run_path(str(A/script),run_name='__main__')
controls=json.loads((results/'pooled_cloud_noncloud_placebo.json').read_text())
controlmap={(r['split'],r['scene']):r for r in controls['rows']};rows=[]
for split,scs in info.items():
 donors=[(s['name'],r,c,g[0]==2) for s in scs for r,c,g,f in s['tiles'] if .1<=f<=.6]
 for s in scs:
  if (split,s['name']) not in controlmap:continue
  control=controlmap[(split,s['name'])];own=[g[0]==2 for r,c,g,f in s['tiles'] if f>=.1];clear=[(r,c,g) for r,c,g,f in s['tiles'] if f<.01]
  acc={k:np.zeros(3,np.int64) for k in ('original','cloud','placebo')}
  with rasterio.open(s['s2']) as S:
   for i,(r,c,g) in enumerate(clear):
    mask=own[i%len(own)] if own else donors[(i+42)%len(donors)][3]
    raw=S.read(list(range(1,14)),window=Window(c,r,256,256)).astype(np.float32);valid=(g[1]>0)&~mask;water=g[1]==2
    acc['original']+=cm(originals[(split,s['name'],r,c)],water,valid)
    for kind,key in [('cloud','cloud_spectrum'),('placebo','noncloud_spectrum')]:
     painted=raw.copy();painted[:,mask]=np.array(control[key],np.float32)[:,None]
     acc[kind]+=cm(predict(painted),water,valid)
  scores={k:iou(v) for k,v in acc.items()};assert all(v is not None for v in scores.values())
  row=dict(split=split,scene=s['name'],n_clear_tiles=len(clear),confusions={k:v.tolist() for k,v in acc.items()},iou=scores)
  row['gap_cloud_pts']=100*(scores['original']-scores['cloud']);row['gap_placebo_pts']=100*(scores['original']-scores['placebo']);row['cloud_minus_placebo_pts']=row['gap_cloud_pts']-row['gap_placebo_pts'];rows.append(row)
  print('CANDIDATE PASTE',split,s['name'],row['gap_cloud_pts'],flush=True)
def summary(rs,key):
 g=np.array([r[key] for r in rs]);rng=np.random.default_rng(42);bs=[rng.choice(g,len(g)).mean() for _ in range(5000)]
 return dict(scenes=len(g),mean_gap_pts=float(g.mean()),ci95=np.percentile(bs,[2.5,97.5]).tolist())
paste_summary={sp:{k:summary([r for r in rows if sp=='val_test' or r['split']==sp],k) for k in ('gap_cloud_pts','gap_placebo_pts','cloud_minus_placebo_pts')} for sp in ('val_test','val','test')}
(R/'candidate_same_protocol.json').write_text(json.dumps(dict(rule='Same corrected cloud/noncloud spectra, masks and unpainted-pixel scoring as frozen audit; candidate selected only on validation.',rows=rows,summary=paste_summary),indent=2))
base=json.loads((results/'cloud_iou_gpu_tiles.json').read_text())
comparison={sp:{g:{'baseline_iou':pooled([r for r in base if r['split']==sp and f(r)]),'candidate_iou':pooled([r for r in natural if r['split']==sp and f(r)])} for g,f in [('all',lambda r:True),('clear',lambda r:r['cloud_frac']<.01),('cloudy',lambda r:r['cloud_frac']>=.1)]} for sp in ('val','test')}
paired={}
for sp in ('val','test'):
 deltas=[]
 for name in sorted({r['scene'] for r in natural if r['split']==sp}):
  filt=lambda r:r['split']==sp and r['scene']==name
  b=pooled([r for r in base if filt(r)]);p=pooled([r for r in natural if filt(r)])
  if b is not None and p is not None:deltas.append(dict(scene=name,baseline_iou=b,candidate_iou=p,delta_pts=100*(p-b)))
 paired[sp]=dict(rows=deltas,summary=summary(deltas,'delta_pts'))
report=dict(selection=final,comparison=comparison,paired_scene_comparison=paired,target_met=comparison['test']['all']['candidate_iou']>=.9,test_reused='Existing test was previously inspected for the original research audit; not a fresh blind benchmark.',no_test_selection=True)
(R/'candidate_comparison.json').write_text(json.dumps(report,indent=2))
print('FINAL CANDIDATE COMPARISON',json.dumps(report),flush=True)
