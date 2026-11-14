import json, os, numpy as np
def ci(g):
    g=np.array([x for x in g if np.isfinite(x)])
    if not len(g): return dict(n=0)
    rng=np.random.default_rng(42); bs=[rng.choice(g,len(g)).mean() for _ in range(5000)]
    lo,hi=np.percentile(bs,[2.5,97.5]); return dict(n=len(g),mean=float(g.mean()),ci95=[float(lo),float(hi)])
R=json.load(open("p2_results.json"))["rows"]
own=[r for r in R if r["own_mask"] and r["own_spectrum"]]; rest=[r for r in R if not(r["own_mask"] and r["own_spectrum"])]
out=dict(own_cloud_only=ci([r["gap_pts"] for r in own]),complement=ci([r["gap_pts"] for r in rest]))
if os.path.exists("p2_shuffled_results.json"):
    S=json.load(open("p2_shuffled_results.json"))["rows"]
    out["real_recomputed"]=ci([r["gap_real"] for r in S]); out["shuffled"]=ci([r["gap_sh"] for r in S])
    out["paired_real_minus_shuffled"]=ci([r["gap_real"]-r["gap_sh"] for r in S])
    out["shuffled_own_cloud_only"]=ci([r["gap_sh"] for r in S if r["own_mask"] and r["own_spectrum"]])
    for t in ("real","sh"):
        for j,b in enumerate(["0-8","8-32",">32"]): out[f"edge_{t}_{b}"]=ci([r[f"gap_{t}_bin{j}"] for r in S])
json.dump(out,open("p2_followup_analyze.out.json","w"),indent=1); print(json.dumps(out,indent=1))
