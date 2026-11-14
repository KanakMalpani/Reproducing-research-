import os
os.environ['MPLBACKEND']='Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
import sys,json,pathlib,yaml,numpy as np,torch,rasterio
from rasterio.windows import Window
from terratorch.tasks import SemanticSegmentationTask
from terratorch.datamodules.sen1floods11 import MEANS,STDS
audit=pathlib.Path(__file__).resolve().parent
os.chdir(audit)
sys.path.insert(0,str(audit))
from wf_data import scenes
results=audit/'results'; pdir=audit/'prithvi_assets'
config=yaml.safe_load((pdir/'config.yaml').read_text())
args=config['model']['init_args']
args['model_args']['backbone_pretrained']=False
print('BUILDING PRITHVI',flush=True)
prithvi_task=SemanticSegmentationTask(**args)
weights=torch.load(pdir/'Prithvi-EO-V2-300M-TL-Sen1Floods11.pt',map_location='cpu',weights_only=True)
print('CHECKPOINT KEYS',list(weights)[:8],flush=True)
if 'state_dict' in weights: weights=weights['state_dict']
if all(k.startswith('model.') for k in weights):
    prithvi_task.load_state_dict(weights,strict=True)
else:
    prithvi_task.model.load_state_dict(weights,strict=True)
del weights
prithvi_model=prithvi_task.model.to('cuda').eval()
bands=config['data']['init_args']['bands']
means=np.array([MEANS[b] for b in bands],np.float32)[:,None,None]
stds=np.array([STDS[b] for b in bands],np.float32)[:,None,None]
@torch.inference_mode()
def predict(raw):
    image=(raw[[1,2,3,8,11,12]].astype(np.float32)/10000.-means)/stds
    output=prithvi_model(torch.from_numpy(image)[None].to('cuda'),temporal_coords=None,location_coords=None).output
    assert output.shape==(1,2,256,256), str(output.shape)
    return (output.argmax(1)[0]==1).cpu().numpy()

import builtins,runpy
from unittest.mock import patch
rule='Frozen Prithvi Sen1Floods11 checkpoint; strict full checkpoint loading; no training. 256x256 tiles, stride512, >=50% labelled pixels, identical to WF2. S2 indices [1,2,3,8,11,12], /10000 reflectance and official Sen1Floods11 means/stds; native 256px inference; no temporal/location metadata (config use_metadata=false); argmax of 2 logits. Composition: unchanged water-fraction and NDVI greedy matching and gates from committed script. Same-tile: identical corrected cloud/non-cloud spectra and donor-mask assignments from FOLLOWUP_POOLED_PLACEBO_RULE.md; unpainted valid pixels only. Aggregation: pooled confusion within scenes; unweighted scene means and5000 scene-bootstrap draws, seed0 for natural controls, seed42 for paste. Independent cross-dataset model, not a claim of WorldFloods training equivalence.'
(audit/'PRITHVI_SAME_PROTOCOL_RULE.md').write_text(rule)
def cm(p,w,v):
    return np.array([(p&w&v).sum(),(p&~w&v).sum(),(~p&w&v).sum()],np.int64)
def iou(v):
    return float(v[0]/sum(v)) if sum(v) else None
info={'val':[],'test':[]}
for split in info:
    for name,s2,gt in scenes(split):
        ts=[]
        with rasterio.open(gt) as G:
            for r in range(0,G.height-255,512):
                for c in range(0,G.width-255,512):
                    g=G.read(window=Window(c,r,256,256)); v=g[1]>0
                    if v.mean()<.5: continue
                    ts.append((r,c,g,float((g[0]==2)[v].mean())))
        info[split].append(dict(name=name,s2=s2,tiles=ts))

reference=json.loads((audit/'method1_tiles.json').read_text())
refmap={(x['split'],x['scene'],x['r'],x['c']):x for x in reference}
natural=[]; terrain=[]; originals={}
for split,scs in info.items():
    for s in scs:
        with rasterio.open(s['s2']) as S:
            for r,c,g,f in s['tiles']:
                raw=S.read(list(range(1,14)),window=Window(c,r,256,256))
                p=predict(raw); v=g[1]>0; w=g[1]==2; cloud=g[0]==2
                key=(split,s['name'],r,c); originals[key]=p
                base=cm(p,w,v).tolist()
                natural.append(dict(split=split,scene=s['name'],r=r,c=c,cloud_frac=f,all=base,clear=cm(p,w,v&~cloud).tolist(),under_cloud=cm(p,w,v&cloud).tolist()))
                item=dict(refmap[key]); item['cm']={'base':base}; terrain.append(item)
        print('PRITHVI NATURAL',split,s['name'],len(natural),flush=True)
assert set(originals)==set(refmap)
(results/'prithvi_cloud_iou_tiles.json').write_text(json.dumps(natural))
(results/'prithvi_method1_tiles.json').write_text(json.dumps(terrain))
real_open=builtins.open
maps={'cloud_iou_tiles.json':results/'prithvi_cloud_iou_tiles.json','method1_tiles.json':results/'prithvi_method1_tiles.json'}
for script,outname in [('kill_test.py','kill_test_results.json'),('ndvi_comatch_analyze.py','ndvi_comatch_results.json')]:
    def redirected(file,mode='r',*args,**kwargs):
        if isinstance(file,(str,os.PathLike)):
            name=pathlib.Path(file).name
            if any(c in mode for c in 'wax+'):
                assert name==outname
                file=results/('prithvi_'+outname)
            elif name in maps: file=maps[name]
        return real_open(file,mode,*args,**kwargs)
    with patch('builtins.open',redirected):
        runpy.run_path(str(audit/script),run_name='__main__')
    print('PRITHVI COMPOSITION SAVED',script,flush=True)
controls=json.loads((results/'pooled_cloud_noncloud_placebo.json').read_text())
controlmap={(r['split'],r['scene']):r for r in controls['rows']}
rows=[]
for split,scs in info.items():
    donors=[(s['name'],r,c,g[0]==2) for s in scs for r,c,g,f in s['tiles'] if .1<=f<=.6]
    for s in scs:
        if (split,s['name']) not in controlmap: continue
        control=controlmap[(split,s['name'])]
        own=[g[0]==2 for r,c,g,f in s['tiles'] if f>=.1]
        clear=[(r,c,g) for r,c,g,f in s['tiles'] if f<.01]
        acc={k:np.zeros(3,np.int64) for k in ('original','cloud','placebo')}
        with rasterio.open(s['s2']) as S:
            for i,(r,c,g) in enumerate(clear):
                mask=own[i%len(own)] if own else donors[(i+42)%len(donors)][3]
                raw=S.read(list(range(1,14)),window=Window(c,r,256,256)).astype(np.float32)
                valid=(g[1]>0)&~mask; water=g[1]==2
                acc['original']+=cm(originals[(split,s['name'],r,c)],water,valid)
                for kind,spec_key in [('cloud','cloud_spectrum'),('placebo','noncloud_spectrum')]:
                    painted=raw.copy(); painted[:,mask]=np.array(control[spec_key],np.float32)[:,None]
                    acc[kind]+=cm(predict(painted),water,valid)
        scores={k:iou(v) for k,v in acc.items()}
        row=dict(split=split,scene=s['name'],n_clear_tiles=len(clear),confusions={k:v.tolist() for k,v in acc.items()},iou=scores)
        assert all(v is not None for v in scores.values()), 'Undefined IoU'
        row['gap_cloud_pts']=100*(scores['original']-scores['cloud'])
        row['gap_placebo_pts']=100*(scores['original']-scores['placebo'])
        row['cloud_minus_placebo_pts']=row['gap_cloud_pts']-row['gap_placebo_pts']
        rows.append(row)
        (results/'prithvi_paste_progress.json').write_text(json.dumps(rows,indent=2))
        print('PRITHVI PASTE',split,s['name'],row['gap_cloud_pts'],row['gap_placebo_pts'],flush=True)
def summary(rs,key):
    g=np.array([r[key] for r in rs]); rng=np.random.default_rng(42)
    bs=[rng.choice(g,len(g)).mean() for _ in range(5000)]
    return dict(scenes=len(g),mean_gap_pts=float(g.mean()),ci95=np.percentile(bs,[2.5,97.5]).tolist())
summary_result={sp:{k:summary([r for r in rows if sp=='val_test' or r['split']==sp],k) for k in ('gap_cloud_pts','gap_placebo_pts','cloud_minus_placebo_pts')} for sp in ('val_test','val','test')}
(results/'prithvi_same_protocol.json').write_text(json.dumps(dict(rule=rule,model_revision=json.loads((results/'prithvi_download_manifest.json').read_text())['revision'],rows=rows,summary=summary_result),indent=2))
base=json.loads((results/'cloud_iou_gpu_tiles.json').read_text())
def pooled(rs): return iou(np.array([r['all'] for r in rs]).sum(0))
comparison={sp:{group:{'wf2_iou':pooled([r for r in base if r['split']==sp and filt(r)]),'prithvi_iou':pooled([r for r in natural if r['split']==sp and filt(r)])} for group,filt in [('all',lambda r:True),('clear',lambda r:r['cloud_frac']<.01),('cloudy',lambda r:r['cloud_frac']>=.1)]} for sp in ('val','test')}
(results/'prithvi_wf2_pooled_comparison.json').write_text(json.dumps(dict(aggregator='pixel pooled; secondary to scene-level controls',comparison=comparison),indent=2))
print('PRITHVI PASTE SUMMARY',json.dumps(summary_result),flush=True)
print('PRITHVI POOLED COMPARISON',json.dumps(comparison),flush=True)
print('PRITHVI COMPLETE',flush=True)
