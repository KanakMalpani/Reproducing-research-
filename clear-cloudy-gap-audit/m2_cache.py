"""Cache frozen-UNet logits + cp + gt per tile to /workspace/wf-cache/<split>_*.npy (memmap-friendly). Not in git."""
import os, sys, json, numpy as np, torch, rasterio
from rasterio.windows import Window
torch.set_num_threads(3)
from wf_data import scenes, NORM
sys.argv=[sys.argv[0]]; exec(open("cloud_iou_pilot.py").read().split("BINS=")[0])
C="/workspace/wf-cache"; os.makedirs(C,exist_ok=True)
for split in (sys.argv[1:] or ["val","test","train"]) if False else os.environ.get("SPLITS","val,test,train").split(","):
    meta=[]; X=[]; Y=[]
    fx=open(f"{C}/{split}_x.bin","wb"); fy=open(f"{C}/{split}_y.bin","wb")
    for name,s2,gt in scenes(split):
        try:
          with rasterio.open(s2) as S, rasterio.open(gt) as G:
            H,W=G.shape
            for r in range(0,H-255,512):
                for c in range(0,W-255,512):
                    w=Window(c,r,256,256); g=G.read(window=w)
                    if (g[1]>0).mean()<0.5: continue
                    raw=S.read(list(range(1,14))+[15],window=w).astype(np.float32)
                    x=(raw[:13]-NORM[:,0,None,None])/NORM[:,1,None,None]
                    with torch.no_grad(): lg=m(torch.from_numpy(x)[None])[0].numpy()
                    np.stack([lg[1],lg[0],raw[13]/100.]).astype(np.float16).tofile(fx)
                    g.astype(np.int8).tofile(fy)
                    wt=g[1]==2; v=g[1]>0; cl=g[0]==2
                    meta.append(dict(scene=name,r=r,c=c,cloud_frac=float(cl[v].mean()),water_frac=float(wt[v].mean())))
        except Exception as e: print("skip",name,e,flush=True)
        print(split,name,len(meta),flush=True)
    fx.close(); fy.close(); json.dump(meta,open(f"{C}/{split}_meta.json","w"))
