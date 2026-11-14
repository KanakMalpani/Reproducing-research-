import json, numpy as np
R=json.load(open("p2_results.json"))["rows"]
def summ(rs):
    g=np.array([x["gap_pts"] for x in rs if np.isfinite(x["gap_pts"])])
    if not len(g): return dict(scenes=0)
    rng=np.random.default_rng(42); bs=[rng.choice(g,len(g)).mean() for _ in range(5000)]
    lo,hi=np.percentile(bs,[2.5,97.5])
    v="KILL" if g.mean()>=5 and lo>0 else "SUPPORT" if hi<5 else "INCONCLUSIVE"
    pi=[x["iou_painted_px"] for x in rs if np.isfinite(x["iou_painted_px"])]
    return dict(scenes=len(g),mean_gap_pts=float(g.mean()),median_gap_pts=float(np.median(g)),ci95=[float(lo),float(hi)],
        frac_scenes_drop=float((g>0).mean()),mean_iou_painted_px=float(np.mean(pi)) if pi else None,
        mean_iou_orig_unpainted=float(np.mean([x["iou_orig_unpainted"] for x in rs if np.isfinite(x["gap_pts"])])),verdict=v)
out=dict(headline_val_test=summ(R),val=summ([x for x in R if x["split"]=="val"]),test=summ([x for x in R if x["split"]=="test"]))
json.dump(out,open("p2_analyze.out.json","w"),indent=1); print(json.dumps(out,indent=1))
