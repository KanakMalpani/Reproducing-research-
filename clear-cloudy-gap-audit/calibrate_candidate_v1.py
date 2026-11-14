"""Validation-only selection and bias calibration of separate model weights."""
import os,sys,json,pathlib,math,hashlib,collections
import numpy as np,torch,rasterio
from rasterio.windows import Window
A=pathlib.Path(__file__).resolve().parent
os.chdir(A);sys.path.insert(0,str(A))
from wf_data import NORM
R=A/'results'/'finetune_v1'; C=A/'checkpoints'/'finetune_v1'
assert (R/'calibration_rule.json').exists()
assert not (R/'final_selection.json').exists(), 'Selection already frozen'
selection=json.loads((R/'selection.json').read_text())
BASE=pathlib.Path(os.environ['WF_CKPT']);ROOT=pathlib.Path(os.environ['WF_ROOT'])
meta=[r for r in json.loads((A/'method1_tiles.json').read_text()) if r['split']=='val']
ns={};exec((A/'cloud_iou_pilot.py').read_text().split('BINS=')[0],ns)
m=ns['m'].cuda().eval();torch.set_num_threads(3)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def iou(x):return float(x[0]/sum(x)) if sum(x) else None
def cm(p,w,v):return np.array([(p&w&v).sum(),(p&~w&v).sum(),(~p&w&v).sum()],np.int64)
thresholds=[round(.2+.05*i,2) for i in range(15)]
sources=[dict(name='released',path=str(BASE),epoch=0)]
if selection['selected']:sources.append(dict(name='best_eligible',**selection['selected']))
anypath=C/'best_validation_model.pt'
anyepoch=selection['best_any_epoch']
if anypath.exists() and not any(s['epoch']==anyepoch for s in sources):sources.append(dict(name='best_validation',path=str(anypath),epoch=anyepoch))
baseline=json.loads((R/'baseline_validation.json').read_text())['metrics']
rows=[];best=None
for source in sources:
 weights=torch.load(source['path'],map_location='cpu',weights_only=True)
 if 'state_dict' in weights:weights=weights['state_dict']
 weights={k.removeprefix('network.'):v for k,v in weights.items()}
 m.load_state_dict(weights,strict=True);del weights
 counts={t:{g:np.zeros(3,np.int64) for g in ('all','clear','cloudy')} for t in thresholds}
 scene_counts={t:collections.defaultdict(lambda:np.zeros(3,np.int64)) for t in thresholds}
 for scene in sorted({r['scene'] for r in meta}):
  with rasterio.open(ROOT/'val'/'S2'/(scene+'.tif')) as S,rasterio.open(ROOT/'val'/'gt'/(scene+'.tif')) as G:
   for r in [r for r in meta if r['scene']==scene]:
    win=Window(r['c'],r['r'],256,256);raw=S.read(list(range(1,14)),window=win).astype(np.float32);g=G.read(window=win)
    x=(raw-NORM[:,0,None,None])/NORM[:,1,None,None]
    with torch.inference_mode():logits=m(torch.from_numpy(x)[None].cuda())[0,1].cpu().numpy()
    w=g[1]==2;v=g[1]>0
    group='clear' if r['cloud_frac']<.01 else 'cloudy' if r['cloud_frac']>=.1 else None
    for t in thresholds:
     a=cm(logits>math.log(t/(1-t)),w,v);counts[t]['all']+=a;scene_counts[t][scene]+=a
     if group:counts[t][group]+=a
  print('CALIBRATION VAL',source['name'],scene,flush=True)
 for t in thresholds:
  metrics={g:iou(a) for g,a in counts[t].items()}
  eligible=metrics['clear']>=baseline['clear']-.01 and metrics['cloudy']>=baseline['cloudy']-.01
  scene_mean=float(np.mean([iou(a) for a in scene_counts[t].values() if iou(a) is not None]))
  row=dict(source=source,threshold=t,validation=metrics,eligible=eligible,scene_mean_iou=scene_mean,confusions={g:a.tolist() for g,a in counts[t].items()})
  rows.append(row)
  rank=(metrics['all'],-abs(t-.5),scene_mean)
  if eligible and (best is None or rank>best[0]):best=(rank,row)
 (R/'calibration_curve.json').write_text(json.dumps(dict(baseline=baseline,rows=rows),indent=2))
assert best is not None
chosen=best[1];src=chosen['source'];t=chosen['threshold']
weights=torch.load(src['path'],map_location='cpu',weights_only=True)
if 'state_dict' in weights:weights=weights['state_dict']
weights={k.removeprefix('network.'):v.clone() for k,v in weights.items()}
original_bias=float(weights['conv_last.bias'][1]);shift=-math.log(t/(1-t))
weights['conv_last.bias'][1]+=shift
path=C/'calibrated_selected_model.pt'
torch.save({'network.'+k:v for k,v in weights.items()},path)
final=dict(chosen=chosen,checkpoint=str(path),checkpoint_sha256=sha(path),source_sha256=sha(src['path']),water_bias_before=original_bias,water_bias_shift=shift,water_bias_after=float(weights['conv_last.bias'][1]),prediction_threshold=.5,test_opened=False,target_test_iou=.9)
(R/'final_selection.json').write_text(json.dumps(final,indent=2))
print('FINAL VALIDATION SELECTION FROZEN',json.dumps(final),flush=True)
