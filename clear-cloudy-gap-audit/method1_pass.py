"""One inference pass over val+test: per-tile terrain stats + confusions for each method-1 config (see SELECTION_RULE_method1.md)."""
import os, json, numpy as np, torch, rasterio
from rasterio.windows import Window
torch.set_num_threads(3)
import importlib.util, sys
from wf_data import scenes, NORM
sys.argv=[sys.argv[0],"__none__"]  # reuse UNet + weights from pilot without running its loop
src=open("cloud_iou_pilot.py").read().split("BINS=")[0]; exec(src)
CFG=[("base",0.0)]+[(f"T{a:+.1f}",a) for a in (-.4,-.3,-.2,-.1,.1,.2,.3,.4)]+[(f"G{b:+.1f}",b) for b in (-.4,-.3,-.2,-.1,.1,.2,.3,.4)]
rows=[]
for split in ("val","test"):
    for name,s2,gt in scenes(split):
        jrc=gt.replace("/gt/","/PERMANENTWATERJRC/")
        with rasterio.open(s2) as S, rasterio.open(gt) as G:
            J=rasterio.open(jrc) if os.path.exists(jrc) else None
            H,W=G.shape
            for r in range(0,H-255,512):
                for c in range(0,W-255,512):
                    w=Window(c,r,256,256); g=G.read(window=w)
                    if (g[1]>0).mean()<0.5: continue
                    raw=S.read(list(range(1,14))+[15],window=w).astype(np.float32)
                    x=(raw[:13]-NORM[:,0,None,None])/NORM[:,1,None,None]; cp=raw[13]/100.
                    with torch.no_grad(): p=torch.sigmoid(m(torch.from_numpy(x)[None]))[0].numpy()
                    pc,pw=p[0],p[1]; valid=g[1]>0; wt=g[1]==2; cl=g[0]==2
                    ndvi=(raw[7]-raw[3])/(raw[7]+raw[3]+1e-6)
                    jw=float(((J.read(1,window=w)==3)&valid).sum()/valid.sum()) if J else None
                    cm={}
                    for k,v in CFG:
                        if k[0]=="T": pred=pw>np.clip(0.5+v*cp,0.05,0.95)
                        elif k[0]=="G": pred=pw*(1-v*pc)>0.5
                        else: pred=pw>0.5
                        cm[k]=[int((pred&wt&valid).sum()),int((pred&~wt&valid).sum()),int((~pred&wt&valid).sum())]
                    rows.append(dict(split=split,scene=name,r=r,c=c,cloud_frac=float(cl[valid].mean()),water_frac=float(wt[valid].mean()),
                        ndvi_clear=float(ndvi[valid&~cl].mean()) if (valid&~cl).any() else None,jrc_perm_frac=jw,cm=cm))
            if J: J.close()
        print(split,name,len(rows),flush=True)
json.dump(rows,open("method1_tiles.json","w"))
