"""Val-only pixel census per scene: clear / thin cloud / thick cloud, and flood (water=2) pixels under each.
gt cloud band: 0 invalid, 1 clear, 2 cloud. Thin vs thick split for gt-cloud pixels uses S2 band 15
(s2cloudless cloud probability, 0-100).
v1 (thin = gt-cloud & prob<50) gave 0 thin px on val: the gt cloud mask is itself thresholded s2cloudless.
v2 (this): clear = gt-clear & prob<LO(20); thin = gt-clear & prob>=LO (haze/thin cloud the gt calls clear); thick = gt-cloud."""
import glob, os, sys, json, numpy as np, rasterio
ROOT="/home/box/worldfloods"; THR=int(os.environ.get("THR",50)); split=sys.argv[1] if len(sys.argv)>1 else "val"
rows=[]
for g in sorted(glob.glob(f"{ROOT}/{split}/gt/*.tif")):
    s=g.replace("/gt/","/S2/")
    try:
        with rasterio.open(g) as G: gt=G.read()
        with rasterio.open(s) as S: p=S.read(15)
    except Exception as e: print("skip",os.path.basename(g),e); continue
    cl,wt=gt[0],gt[1]; lab=wt>0; fl=wt==2
    LO=int(os.environ.get("LO",20))
    m={"clear":(cl==1)&(p<LO),"thin":(cl==1)&(p>=LO),"thick":cl==2}
    r=dict(scene=os.path.basename(g)[:-4],cloudprob_max=int(p.max()))
    for k,v in m.items(): r[f"{k}_px"]=int((v&lab).sum()); r[f"flood_{k}_px"]=int((v&fl).sum())
    rows.append(r); print(r,flush=True); del gt,p
tot={k:sum(r[k] for r in rows) for k in rows[0] if k.endswith("_px")}
json.dump(dict(split=split,thr=THR,totals=tot,per_scene=rows),open(f"{split}_cloud_counts.json","w"),indent=1)
print("TOTALS",tot)
