"""Test label histogram (LABEL_HISTOGRAM_RULE.md)."""
import json, numpy as np, rasterio
from wf_data import scenes
PB=np.arange(0,111,10); BB=np.append(np.arange(0,10001,500),1e9)
H={k:dict(prob=np.zeros(11,np.int64),bright=np.zeros(len(BB)-1,np.int64)) for k in ("gt_cloud","gt_clear")}
per=[]
for name,s2,gt in scenes("test"):
    with rasterio.open(gt) as G: g=G.read()
    with rasterio.open(s2) as S: p=S.read(15); b=S.read([2,3,4]).astype(np.float32).mean(0)
    lab=g[1]>0
    for k,m in (("gt_cloud",lab&(g[0]==2)),("gt_clear",lab&(g[0]==1))):
        H[k]["prob"]+=np.histogram(p[m],PB)[0]; H[k]["bright"]+=np.histogram(b[m],BB)[0]
    mc=lab&(g[0]==2); per.append(dict(scene=name,cloud_px=int(mc.sum()),cloud_prob_lt50=int((p[mc]<50).sum())))
    print(per[-1],flush=True); del g,p,b
def frac(h,hi): return float(h[:hi].sum()/max(h.sum(),1))
c,k=H["gt_cloud"]["prob"],H["gt_clear"]["prob"]
out=dict(split="test",prob_bins=PB.tolist(),bright_bins=BB[:-1].tolist(),
  hist={a:{b:v.tolist() for b,v in d.items()} for a,d in H.items()},
  gt_cloud_px=int(c.sum()),gt_clear_px=int(k.sum()),
  frac_gt_cloud_prob_lt50=frac(c,5),frac_gt_cloud_prob_lt20=frac(c,2),frac_gt_clear_prob_ge20=1-frac(k,2),
  per_scene=per)
out["thin_class_inside_gt_cloud"]=out["frac_gt_cloud_prob_lt50"]>=0.05
json.dump(out,open("label_histogram_test.json","w"),indent=1)
print({x:out[x] for x in out if x.startswith(("frac","gt_","thin"))})
