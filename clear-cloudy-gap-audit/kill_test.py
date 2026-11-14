"""Kill test: same-scene clear (cf<0.01) vs cloudy (cf>=0.1) tile water IoU, scene-paired, bootstrap CI. Bar: gap >= 5 pts."""
import json, sys, numpy as np
def paired(rows, min_px=200, key="all"):
    by={}
    for x in rows: by.setdefault(x["scene"],[]).append(x)
    out=[]
    for s,rs in by.items():
        cl=[x for x in rs if x["cloud_frac"]<0.01]; cd=[x for x in rs if x["cloud_frac"]>=0.1]
        def pool(t):
            a=np.array([x[key] for x in t]).sum(0) if t else np.zeros(3); return a
        a,b=pool(cl),pool(cd)
        if a[0]+a[2]<min_px or b[0]+b[2]<min_px: continue   # need water in both
        out.append(dict(scene=s,n_clear=len(cl),n_cloudy=len(cd),iou_clear=a[0]/a.sum(),iou_cloudy=b[0]/b.sum(),gap=(a[0]/a.sum()-b[0]/b.sum())*100))
    return out
def summ(p, seed=0):
    if not p: return dict(scenes=0)
    g=np.array([x["gap"] for x in p]); rng=np.random.default_rng(seed)
    bs=[rng.choice(g,len(g)).mean() for _ in range(5000)]
    return dict(scenes=len(g),mean_gap_pts=float(g.mean()),median_gap_pts=float(np.median(g)),ci95=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],frac_scenes_clear_better=float((g>0).mean()),
                verdict=("PASS (headroom >=5 pts)" if np.percentile(bs,2.5)>=5 else "FAIL (gap CI entirely below 5 pts)" if np.percentile(bs,97.5)<5 else "INCONCLUSIVE (CI straddles 5 pts)"))
res={}
for name,f in [("test_val","cloud_iou_tiles.json")]:  # train dropped: checkpoint trained on it
    rows=json.load(open(f)); p=paired(rows)
    pc=paired(rows,key="clear")
    for sp in ("val","test"): res[sp]=dict(summary=summ(paired([x for x in rows if x["split"]==sp])))
    res[name]=dict(summary=summ(p),per_scene=p,summary_cloudy_tiles_visible_px_only=summ(pc))
json.dump(res,open("kill_test_results.json","w"),indent=1,default=float)
for k,v in res.items(): print(k,json.dumps(v["summary"]),"| visible-px:",json.dumps(v.get("summary_cloudy_tiles_visible_px_only")))
