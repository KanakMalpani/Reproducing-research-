import os, sys, json, pathlib, torch, subprocess
import numpy as np, rasterio
from rasterio.windows import Window
audit=pathlib.Path(__file__).resolve().parent
os.chdir(audit)
results=audit/'results'
results.mkdir(exist_ok=True)
wf_root=pathlib.Path(os.environ['WF_ROOT'])
assert wf_root.exists() and pathlib.Path(os.environ['WF_CKPT']).exists(), 'STOP: missing WF_ROOT or WF_CKPT'
sys.path.insert(0,str(audit))
from wf_data import scenes, NORM
manifest=json.loads((results/'download_manifest.json').read_text())
import numpy as np, rasterio
from rasterio.windows import Window
from wf_data import scenes, NORM
assert manifest['complete']
rule = '''Follow-up registered before new results. Frozen WF2_unetv2_all; 256px grid, stride 512, >=50% valid water labels; clear cf<0.01, donor cf>=0.1; eligible scenes >=200 clear-water pixels. Same-scene donors cycle row/col. Fallback donors are sorted within split with labelled-pixel cf in [0.1,0.6], index (i+42)%K. Corrected cloud fill: exact per-band pooled median over every GT-cloud pixel in kept scene tiles if >=1000 cloud pixels, else exact pooled validation-cloud median. Non-cloud placebo: exact per-band pooled median of valid GT-clear pixels of the same scene, or frozen pooled validation clear pixels if <1000. Identical masks for both fills; constant hard paste; score only unpainted valid pixels. This placebo measures generic constant-fill disruption, not natural cloud. Scene confusion pools, unweighted scene means, 5000 scene bootstrap draws, seed 42; original result unchanged. No tuning. Exact integer histograms compute medians without subsampling.'''
(audit/'FOLLOWUP_POOLED_PLACEBO_RULE.md').write_text(rule)
namespace = {}
exec((audit/'cloud_iou_pilot.py').read_text().split('BINS=')[0], namespace)
model = namespace['m'].to('cuda').eval()
@torch.inference_mode()
def predict(raw):
    x = (raw.astype(np.float32)-NORM[:,0,None,None])/NORM[:,1,None,None]
    return (torch.sigmoid(model(torch.from_numpy(x)[None].to('cuda')))[0,1]>0.5).cpu().numpy()
def cm(p,w,v):
    return np.array([(p&w&v).sum(),(p&~w&v).sum(),(~p&w&v).sum()],np.int64)
def iou(a):
    return float(a[0]/a.sum()) if a.sum() else None
def median_hist(h):
    n=int(h[0].sum())
    assert n>0
    c=np.cumsum(h,axis=1)
    return np.array([(np.searchsorted(row,(n-1)//2+1)+np.searchsorted(row,n//2+1))/2 for row in c],np.float32)
def add_hist(h,raw,mask):
    for band in range(13):
        vals=raw[band,mask]
        assert np.all(vals==vals.astype(np.uint16))
        h[band] += np.bincount(vals.astype(np.int64),minlength=65536)
info = {'val':[],'test':[]}
for split in info:
    for name,s2,gt in scenes(split):
        ts=[]; hc=np.zeros((13,65536),np.int64); hn=hc.copy()
        with rasterio.open(s2) as S, rasterio.open(gt) as G:
            for r in range(0,G.height-255,512):
                for c in range(0,G.width-255,512):
                    win=Window(c,r,256,256); g=G.read(window=win); v=g[1]>0
                    if v.mean()<0.5: continue
                    raw=S.read(list(range(1,14)),window=win)
                    f=float((g[0]==2)[v].mean())
                    ts.append((r,c,g,f))
                    add_hist(hc,raw,g[0]==2)
                    add_hist(hn,raw,(g[0]==1)&v)
        info[split].append(dict(name=name,s2=s2,tiles=ts,hc=hc,hn=hn))
        print('EXACT SCAN',split,name,len(ts),flush=True)
frozen_cloud=median_hist(sum(s['hc'] for s in info['val']))
frozen_clear=median_hist(sum(s['hn'] for s in info['val']))
rows=[]
for split,scs in info.items():
    donors=[(s['name'],r,c,g[0]==2) for s in scs for r,c,g,f in s['tiles'] if 0.1<=f<=0.6]
    for s in scs:
        clear=[(r,c,g) for r,c,g,f in s['tiles'] if f<0.01]
        cw=sum(int(((g[0]==1)&(g[1]==2)).sum()) for r,c,g in clear)
        if cw<200: continue
        own=[g[0]==2 for r,c,g,f in s['tiles'] if f>=0.1]
        assert own or donors
        cloud=median_hist(s['hc']) if s['hc'][0].sum()>=1000 else frozen_cloud
        noncloud=median_hist(s['hn']) if s['hn'][0].sum()>=1000 else frozen_clear
        acc={k:np.zeros(3,np.int64) for k in ('original','cloud','placebo')}
        with rasterio.open(s['s2']) as S:
            for i,(r,c,g) in enumerate(clear):
                mask=own[i%len(own)] if own else donors[(i+42)%len(donors)][3]
                raw=S.read(list(range(1,14)),window=Window(c,r,256,256)).astype(np.float32)
                valid=(g[1]>0)&~mask; water=g[1]==2
                acc['original']+=cm(predict(raw),water,valid)
                for kind,spectrum in [('cloud',cloud),('placebo',noncloud)]:
                    painted=raw.copy(); painted[:,mask]=spectrum[:,None]
                    acc[kind]+=cm(predict(painted),water,valid)
        scores={k:iou(v) for k,v in acc.items()}
        row=dict(split=split,scene=s['name'],n_clear_tiles=len(clear),clear_water_px=cw,own_mask=bool(own),own_cloud_spectrum=bool(s['hc'][0].sum()>=1000),cloud_spectrum=cloud.tolist(),noncloud_spectrum=noncloud.tolist(),confusions={k:v.tolist() for k,v in acc.items()},iou=scores)
        row['gap_cloud_pts']=100*(scores['original']-scores['cloud'])
        row['gap_placebo_pts']=100*(scores['original']-scores['placebo'])
        row['cloud_minus_placebo_pts']=row['gap_cloud_pts']-row['gap_placebo_pts']
        rows.append(row)
        (results/'pooled_placebo_progress.json').write_text(json.dumps(rows,indent=2))
        print('POOLED PLACEBO',split,s['name'],row['gap_cloud_pts'],row['gap_placebo_pts'],flush=True)
def summary(rs,key):
    g=np.array([r[key] for r in rs],float)
    rng=np.random.default_rng(42)
    bs=np.array([rng.choice(g,len(g)).mean() for _ in range(5000)])
    return dict(scenes=len(g),mean_gap_pts=float(g.mean()),ci95=np.percentile(bs,[2.5,97.5]).tolist())
summary_result={sp:{k:summary([r for r in rows if sp=='val_test' or r['split']==sp],k) for k in ('gap_cloud_pts','gap_placebo_pts','cloud_minus_placebo_pts')} for sp in ('val_test','val','test')}
output=dict(rule=rule,seed=42,device='cuda',frozen_val_cloud_spectrum=frozen_cloud.tolist(),frozen_val_clear_spectrum=frozen_clear.tolist(),rows=rows,summary=summary_result)
(results/'pooled_cloud_noncloud_placebo.json').write_text(json.dumps(output,indent=2))
print('PASTE SUMMARY',json.dumps(summary_result),flush=True)
