"""P2 same-tile cloud paint (see P2_CLOUD_PAINT_RULE.md). Writes p2_results.json."""
import os, sys, json, numpy as np, torch, rasterio
from rasterio.windows import Window
torch.set_num_threads(int(os.environ.get("NT",4)))
from wf_data import scenes, NORM
sys.argv=[sys.argv[0],"__none__"]; exec(open("cloud_iou_pilot.py").read().split("BINS=")[0])  # UNet m + weights
def grid(gt_path):
    with rasterio.open(gt_path) as G:
        H,W=G.shape
        for r in range(0,H-255,512):
            for c in range(0,W-255,512):
                g=G.read(window=Window(c,r,256,256))
                if (g[1]>0).mean()<0.5: continue
                v=g[1]>0; yield r,c,g,float((g[0]==2)[v].mean())
def raw(s2,r,c):
    with rasterio.open(s2) as S: return S.read(list(range(1,14)),window=Window(c,r,256,256)).astype(np.float32)
def pred(x):
    x=(x-NORM[:,0,None,None])/NORM[:,1,None,None]
    with torch.no_grad(): return torch.sigmoid(m(torch.from_numpy(x)[None]))[0,1].numpy()>0.5
def cm(p,wt,mask): return np.array([(p&wt&mask).sum(),(p&~wt&mask).sum(),(~p&wt&mask).sum()],np.int64)
info={}
for split in ("val","test"):
    S=[]
    for name,s2,gt in scenes(split):
        T=list(grid(gt)); cl=[(r,c,g) for r,c,g,f in T if f<0.01]; cd=[(r,c,g) for r,c,g,f in T if f>=0.1]
        cloudpx=[]; n=0
        for r,c,g in [(r,c,g) for r,c,g,f in T if f>0]:
            k=g[0]==2
            if k.any(): x=raw(s2,r,c); cloudpx.append(x[:,k][:,::max(1,k.sum()//2000)]); n+=int(k.sum())
        spec=np.median(np.concatenate(cloudpx,1),1) if cloudpx else None
        S.append(dict(name=name,s2=s2,clear=cl,donors=[(r,c,g[0]==2) for r,c,g in cd],spec=spec,ncloud=n,
             fr=[f for r,c,g,f in T]))
        print("scan",split,name,len(cl),len(cd),n,flush=True)
    info[split]=S
vp=[s["spec"] for s in info["val"] if s["spec"] is not None]
# frozen val spectrum: median over val GT-cloud pixels (approx: median of per-scene sampled pixels, weighted by sampling)
FROZEN=np.median(np.stack(vp),0)  # NOTE: approximation recorded in output
rows=[]
for split in ("val","test"):
    pool=[(s["name"],r,c,mk) for s in info[split] for r,c,mk in s["donors"]]
    pool=[d for d in pool if 0.1<=d[3].mean()<=0.6 or True]
    for s in info[split]:
        # selection: >=200 clear-water px over clear tiles
        cw=sum(int(((g[0]==1)&(g[1]==2)).sum()) for r,c,g in s["clear"])
        if cw<200: continue
        own=bool(s["donors"]); spec=s["spec"] if s["ncloud"]>=1000 else FROZEN
        o=np.zeros(3,np.int64); a=np.zeros(3,np.int64); pp=np.zeros(3,np.int64)
        for i,(r,c,g) in enumerate(s["clear"]):
            mk=s["donors"][i%len(s["donors"])][2] if own else pool[(i+42)%len(pool)][3]
            x=raw(s["s2"],r,c); v=g[1]>0; wt=g[1]==2
            p0=pred(x); x2=x.copy(); x2[:,mk]=spec[:,None]; p1=pred(x2)
            o+=cm(p0,wt,v&~mk); a+=cm(p1,wt,v&~mk); pp+=cm(p1,wt,v&mk)
        iou=lambda t: float(t[0]/t.sum()) if t.sum() else float("nan")
        rows.append(dict(split=split,scene=s["name"],n_clear_tiles=len(s["clear"]),clear_water_px=cw,own_mask=own,
            own_spectrum=s["ncloud"]>=1000,iou_orig_unpainted=iou(o),iou_paint_unpainted=iou(a),
            gap_pts=(iou(o)-iou(a))*100,iou_painted_px=iou(pp),painted_cm=pp.tolist(),orig_cm=o.tolist(),paint_cm=a.tolist()))
        print("P2",json.dumps(rows[-1]),flush=True)
json.dump(dict(frozen_val_spectrum=FROZEN.tolist(),rows=rows),open("p2_results.json","w"),indent=1)
