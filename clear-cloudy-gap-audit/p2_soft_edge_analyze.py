import json, numpy as np
def ci(g):
    g = np.array([x for x in g if np.isfinite(x)])
    if not len(g): return dict(n=0)
    rng = np.random.default_rng(42)
    bs = [rng.choice(g, len(g)).mean() for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=len(g), mean=float(g.mean()), ci95=[float(lo), float(hi)])
S = json.load(open("p2_soft_edge_results.json"))["rows"]
out = dict(
    soft=ci([r["gap_soft"] for r in S]),
    hard_sanity=ci([r["gap_hard"] for r in S]),
    soft_own_cloud_only=ci([r["gap_soft"] for r in S if r["own_mask"] and r["own_spectrum"]]),
    soft_complement=ci([r["gap_soft"] for r in S if not (r["own_mask"] and r["own_spectrum"])]),
)
for j, b in enumerate(["0-8", "8-32", ">32"]):
    out[f"edge_soft_{b}"] = ci([r[f"gap_soft_bin{j}"] for r in S])
# Pre-declared verdict
m, lo, hi = out["soft"]["mean"], out["soft"]["ci95"][0], out["soft"]["ci95"][1]
edge0 = out["edge_soft_0-8"]["mean"]
edge_far = out["edge_soft_>32"]["mean"]
if lo <= 0:
    verdict = "SOFTEN_RESIDUAL (CI lower <= 0)"
elif m >= 5 and lo > 0 and edge0 > abs(edge_far) + 2:
    verdict = "STRENGTHEN_RESIDUAL (drop>=5, lower>0, concentrated near edge)"
else:
    verdict = "INCONCLUSIVE_HONESTY_CHECK"
out["verdict"] = verdict
out["compare_to"] = dict(hard_p2="5.6 [3.0, 8.5]", shuffled="8.1 [4.1, 12.8]")
json.dump(out, open("p2_soft_edge_analyze.out.json", "w"), indent=1)
print(json.dumps(out, indent=1))
