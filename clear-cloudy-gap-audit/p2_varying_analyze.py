import json, numpy as np
def ci(g):
    g = np.array([x for x in g if np.isfinite(x)])
    if not len(g): return dict(n=0)
    rng = np.random.default_rng(42)
    bs = [rng.choice(g, len(g)).mean() for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=len(g), mean=float(g.mean()), ci95=[float(lo), float(hi)])
S = json.load(open("p2_varying_results.json"))["rows"]
def verdict(o, e0, e2):
    m, lo = o["mean"], o["ci95"][0]
    if lo <= 0: return "SOFTEN_RESIDUAL (CI lower <= 0)"
    if m >= 5 and e0 > abs(e2) + 2: return "STRENGTHEN_RESIDUAL"
    return "INCONCLUSIVE_HONESTY_CHECK"
out = dict(n_rows=len(S))
for k in ("soft_var", "hard_var"):
    out[k] = ci([r[f"gap_{k}"] for r in S])
    for j, b in enumerate(["0-8", "8-32", ">32"]):
        out[f"edge_{k}_{b}"] = ci([r[f"gap_{k}_bin{j}"] for r in S])
    out[f"verdict_{k}"] = verdict(out[k], out[f"edge_{k}_0-8"]["mean"], out[f"edge_{k}_>32"]["mean"])
out["hard_const_sanity"] = ci([r["gap_hard_const"] for r in S])
out["soft_var_own_cloud_only"] = ci([r["gap_soft_var"] for r in S if r["own_mask"] and r["own_spectrum"]])
out["soft_var_complement"] = ci([r["gap_soft_var"] for r in S if not (r["own_mask"] and r["own_spectrum"])])
out["verdict"] = out["verdict_soft_var"]
out["compare_to"] = dict(hard_p2="5.6 [3.0, 8.5]", soft_edge="4.4 [2.3, 6.8]", shuffled="8.1 [4.1, 12.8]")
json.dump(out, open("p2_varying_analyze.out.json", "w"), indent=1)
print(json.dumps(out, indent=1))
