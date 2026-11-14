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
rule_strata = '''Registered before census/strata results. Same frozen WF2 baseline and 256px/512 grid. Brightness=sqrt(B2^2+B3^2+B4^2)/10000, per official ml4floods get_brightness at commit 192426c7286315b59a2f52c7a9cdebe589242de2; cuts .3 and .35; count GT-cloud pixels above/below each cut, with water counts, on valid water-label pixels. JRC strata: permanent JRC==3; nonpermanent JRC in {1,2}; unknown JRC==0 excluded from strata but counted. Compute binary water IoU within each spatial stratum, confusion pooled within scenes; report unweighted scene mean and 5000 scene-bootstrap CI seed42, plus pooled IoU explicitly as secondary. Nonpermanent water is a JRC-based proxy, not independently confirmed flood water. Also report raw separate per-scene IoU for Campo Galliano and Ituango South; within-scene tile bootstrap 5000 draws seed42 is descriptive and not an event interval. Never add these to existing gap means.'''
(audit/'JRC_BRIGHTNESS_SCENE_RULE.md').write_text(rule_strata)
strata_rows=[]; census_rows=[]; separate=[]
for split,scs in info.items():
    for s in scs:
        name=s['name']; jrc=wf_root/split/'PERMANENTWATERJRC'/(name+'.tif')
        assert jrc.exists(), 'STOP missing JRC raster'
        counts={k:np.zeros(3,np.int64) for k in ('permanent','nonpermanent','all')}
        buckets={str(cut):{k:dict(pixels=0,water_pixels=0) for k in ('gt_cloud_le_cut','gt_cloud_gt_cut','gt_clear_le_cut','gt_clear_gt_cut')} for cut in (0.3,0.35)}
        jrc_counts={}; tile_cm=[]
        with rasterio.open(s['s2']) as S, rasterio.open(jrc) as J:
            assert S.shape==J.shape and S.transform==J.transform and S.crs==J.crs, 'STOP: JRC alignment differs'
            for r,c,g,f in s['tiles']:
                win=Window(c,r,256,256)
                raw=S.read(list(range(1,14)),window=win).astype(np.float32)
                j=J.read(1,window=win); v=g[1]>0; wt=g[1]==2
                pred=predict(raw)
                counts['all']+=cm(pred,wt,v)
                counts['permanent']+=cm(pred,wt,v&(j==3))
                counts['nonpermanent']+=cm(pred,wt,v&np.isin(j,[1,2]))
                tile_cm.append(cm(pred,wt,v).tolist())
                for label,n in zip(*np.unique(j[v],return_counts=True)):
                    jrc_counts[str(int(label))]=jrc_counts.get(str(int(label)),0)+int(n)
                brightness=np.sqrt(np.sum(raw[[1,2,3]]**2,axis=0))/10000.
                for cut in (0.3,0.35):
                    for cloud_code,prefix in [(2,'gt_cloud'),(1,'gt_clear')]:
                        for side,bright in [('le_cut',brightness<=cut),('gt_cut',brightness>cut)]:
                            mask=v&(g[0]==cloud_code)&bright
                            b=buckets[str(cut)][prefix+'_'+side]
                            b['pixels']+=int(mask.sum()); b['water_pixels']+=int((mask&wt).sum())
        strata_rows.append(dict(split=split,scene=name,n_tiles=len(s['tiles']),jrc_pixel_counts=jrc_counts,confusions={k:v.tolist() for k,v in counts.items()},iou={k:iou(v) for k,v in counts.items()}))
        census_rows.append(dict(split=split,scene=name,cuts=buckets))
        if name.startswith('EMSR260_07CAMPOGALLIANO') or name.startswith('EMSR286_09ITUANGOSOUTH'):
            rng=np.random.default_rng(42); mat=np.array(tile_cm); values=[]
            for _ in range(5000):
                vv=iou(mat[rng.integers(0,len(mat),len(mat))].sum(0))
                if vv is not None: values.append(vv)
            separate.append(dict(split=split,scene=name,tiles=len(mat),iou=iou(counts['all']),ci95_tile_bootstrap=np.percentile(values,[2.5,97.5]).tolist() if values else None,uncertainty='within-scene tile bootstrap; descriptive; not event uncertainty',confusion=counts['all'].tolist()))
        print('JRC CENSUS',split,name,flush=True)
def stratum_summary(rs,kind):
    vals=[r['iou'][kind] for r in rs if r['iou'][kind] is not None and sum(r['confusions'][kind][::2])>=200]
    rng=np.random.default_rng(42)
    ci=np.percentile([np.mean(rng.choice(vals,len(vals))) for _ in range(5000)],[2.5,97.5]).tolist() if vals else None
    pooled=np.array([r['confusions'][kind] for r in rs]).sum(0)
    return dict(scenes=len(vals),scene_gate='>=200 GT-water pixels in stratum',scene_mean_iou=float(np.mean(vals)) if vals else None,ci95_scene_bootstrap=ci,pooled_iou_secondary=iou(pooled),pooled_confusion=pooled.tolist())
strata_summary={split:{k:stratum_summary([r for r in strata_rows if split=='val_test' or r['split']==split],k) for k in ('permanent','nonpermanent','all')} for split in ('val_test','val','test')}
census_total={str(cut):{k:{field:sum(r['cuts'][str(cut)][k][field] for r in census_rows) for field in ('pixels','water_pixels')} for k in ('gt_cloud_le_cut','gt_cloud_gt_cut','gt_clear_le_cut','gt_clear_gt_cut')} for cut in (.3,.35)}
(results/'permanent_nonpermanent_water.json').write_text(json.dumps(dict(rule=rule_strata,rows=strata_rows,summary=strata_summary),indent=2))
(results/'brightness_census.json').write_text(json.dumps(dict(rule=rule_strata,rows=census_rows,total=census_total),indent=2))
(results/'separate_scene_measurements.json').write_text(json.dumps(dict(rule=rule_strata,scenes=separate),indent=2))
print('STRATA SUMMARY',json.dumps(strata_summary),flush=True)
print('BRIGHTNESS CENSUS',json.dumps(census_total),flush=True)
print('SEPARATE SCENES',json.dumps(separate),flush=True)
