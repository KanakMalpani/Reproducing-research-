"""NDVI co-match: within-scene clear/cloudy tile match on |Δ water|≤0.05 AND |Δ NDVI|≤0.05.

Rule locked in NDVI_COMATCH_RULE.md (committed before this run). Extends the water-only
match in method1_analyze.py. Does not re-run P2 or change Claim A.
"""
import json
import numpy as np

R = json.load(open("method1_tiles.json"))
rng = np.random.default_rng(0)  # seed fixed first, same as method1_analyze.py
iou = lambda a: a[0] / max(sum(a), 1)


def pool(ts, k="base"):
    return np.array([t["cm"][k] for t in ts]).sum(0) if ts else np.zeros(3)


CL = lambda t: t["cloud_frac"] < 0.01
CD = lambda t: t["cloud_frac"] >= 0.1

by = {}
for t in R:
    by.setdefault(t["scene"], []).append(t)

# Paired scenes: both clear and cloudy pools have ≥200 water px (unmatched gate, same as method1)
paired = [
    s
    for s, ts in by.items()
    if (lambda a, b: a[0] + a[2] >= 200 and b[0] + b[2] >= 200)(
        pool([t for t in ts if CL(t)]), pool([t for t in ts if CD(t)])
    )
]

# Water-only match (reproduce method1 for comparison)
mg_water = []
for s in paired:
    a = [t for t in by[s] if CL(t)]
    b = [t for t in by[s] if CD(t)]
    used = set()
    ma, mb = [], []
    for t in sorted(b, key=lambda t: t["water_frac"]):
        cand = [(abs(u["water_frac"] - t["water_frac"]), i) for i, u in enumerate(a) if i not in used]
        if cand and min(cand)[0] <= 0.05:
            i = min(cand)[1]
            used.add(i)
            ma.append(a[i])
            mb.append(t)
    pa, pb = pool(ma), pool(mb)
    if pa[0] + pa[2] >= 200 and pb[0] + pb[2] >= 200:
        mg_water.append(dict(scene=s, pairs=len(mb), gap=100 * (iou(pa) - iou(pb))))


def ndvi_ok(t):
    return t.get("ndvi_clear") is not None


# NDVI co-match
mg = []
for s in paired:
    a = [t for t in by[s] if CL(t) and ndvi_ok(t)]
    b = [t for t in by[s] if CD(t) and ndvi_ok(t)]
    used = set()
    ma, mb = [], []
    for t in sorted(b, key=lambda t: t["water_frac"]):
        cand = []
        for i, u in enumerate(a):
            if i in used:
                continue
            dw = abs(u["water_frac"] - t["water_frac"])
            dn = abs(u["ndvi_clear"] - t["ndvi_clear"])
            if dw <= 0.05 and dn <= 0.05:
                cand.append((dw, i))
        if cand:
            i = min(cand)[1]
            used.add(i)
            ma.append(a[i])
            mb.append(t)
    pa, pb = pool(ma), pool(mb)
    if pa[0] + pa[2] >= 200 and pb[0] + pb[2] >= 200:
        mg.append(
            dict(
                scene=s,
                pairs=len(mb),
                gap=100 * (iou(pa) - iou(pb)),
                mean_abs_dw=float(np.mean([abs(x["water_frac"] - y["water_frac"]) for x, y in zip(ma, mb)])) if mb else None,
                mean_abs_dn=float(np.mean([abs(x["ndvi_clear"] - y["ndvi_clear"]) for x, y in zip(ma, mb)])) if mb else None,
            )
        )


def boot(v):
    v = np.array(v)
    bs = [rng.choice(v, len(v)).mean() for _ in range(5000)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


gaps = [x["gap"] for x in mg]
mean = float(np.mean(gaps)) if gaps else None
ci = boot(gaps) if len(mg) > 1 else None

if mean is None or ci is None:
    verdict = "DROP"
elif mean >= 5.0 and ci[0] > 0:
    verdict = "PASS"
elif ci[0] <= 0:
    verdict = "DROP"  # CI lower ≤0 -> drop matched-cloud-gap framing
else:
    verdict = "INCONCLUSIVE"  # mean < 5 but CI lower > 0; do not register matched claim

out = dict(
    rule="NDVI_COMATCH_RULE.md",
    seed=0,
    n_bootstrap=5000,
    tolerances=dict(abs_water_frac=0.05, abs_ndvi=0.05),
    min_water_px=200,
    n_paired_scenes_unmatched_gate=len(paired),
    water_only=dict(
        n_scenes=len(mg_water),
        per_scene=mg_water,
        mean_gap=float(np.mean([x["gap"] for x in mg_water])) if mg_water else None,
        # note: water-only CI not re-bootstrapped here with a fresh rng consume; see method1_results.json
        reference_mean_ci="8.85 [-0.66, 17.78] from method1_results.json",
    ),
    ndvi_comatch=dict(
        n_scenes=len(mg),
        per_scene=mg,
        mean_gap=mean,
        ci95=ci,
        verdict=verdict,
        scenes_surviving_vs_water_only=dict(
            water_only_n=len(mg_water),
            ndvi_comatch_n=len(mg),
            dropped_scenes=[x["scene"] for x in mg_water if x["scene"] not in {y["scene"] for y in mg}],
            retained_scenes=[x["scene"] for x in mg],
        ),
    ),
)

json.dump(out, open("ndvi_comatch_results.json", "w"), indent=1)
print("water_only n", len(mg_water), [round(x["gap"], 2) for x in mg_water])
print("ndvi_comatch n", len(mg))
for x in mg:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()})
print("mean", None if mean is None else round(mean, 3), "ci", None if ci is None else [round(c, 3) for c in ci])
print("verdict", verdict)
