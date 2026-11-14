"""P2 soft-edge / own-scene-mean spectrum paste (P2_SOFT_EDGE_RULE.md). Writes p2_soft_edge_results.json."""
import os, sys, json, numpy as np, torch, rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt
torch.set_num_threads(int(os.environ.get("NT", 4)))
from wf_data import scenes, NORM
sys.argv = [sys.argv[0], "__none__"]; exec(open("cloud_iou_pilot.py").read().split("BINS=")[0])  # UNet m + weights

def grid(gt_path):
    with rasterio.open(gt_path) as G:
        H, W = G.shape
        for r in range(0, H - 255, 512):
            for c in range(0, W - 255, 512):
                g = G.read(window=Window(c, r, 256, 256))
                if (g[1] > 0).mean() < 0.5: continue
                v = g[1] > 0
                yield r, c, g, float((g[0] == 2)[v].mean())

def raw(s2, r, c):
    with rasterio.open(s2) as S:
        return S.read(list(range(1, 14)), window=Window(c, r, 256, 256)).astype(np.float32)

def pred(x):
    x = (x - NORM[:, 0, None, None]) / NORM[:, 1, None, None]
    with torch.no_grad():
        return torch.sigmoid(m(torch.from_numpy(x)[None]))[0, 1].numpy() > 0.5

def cm(p, wt, mask):
    return np.array([(p & wt & mask).sum(), (p & ~wt & mask).sum(), (~p & wt & mask).sum()], np.int64)

def soft_alpha(mk):
    # Locked: gaussian_filter(M.astype(float32), sigma=2.0, truncate=3.0)
    return gaussian_filter(mk.astype(np.float32), sigma=2.0, truncate=3.0)

info = {}
for split in ("val", "test"):
    S = []
    for name, s2, gt in scenes(split):
        T = list(grid(gt))
        cl = [(r, c, g) for r, c, g, f in T if f < 0.01]
        cd = [(r, c, g) for r, c, g, f in T if f >= 0.1]
        cloudpx = []; n = 0
        for r, c, g in [(r, c, g) for r, c, g, f in T if f > 0]:
            k = g[0] == 2
            if k.any():
                x = raw(s2, r, c)
                cloudpx.append(x[:, k][:, ::max(1, k.sum() // 2000)])
                n += int(k.sum())
        # P2 uses median for frozen fallback; soft-edge own-spectrum uses MEAN (rule).
        mean_spec = np.mean(np.concatenate(cloudpx, 1), 1) if cloudpx else None
        med_spec = np.median(np.concatenate(cloudpx, 1), 1) if cloudpx else None
        S.append(dict(name=name, s2=s2, clear=cl,
                      donors=[(r, c, g[0] == 2) for r, c, g in cd],
                      mean_spec=mean_spec, med_spec=med_spec, ncloud=n))
        print("scan", split, name, len(cl), len(cd), n, flush=True)
    info[split] = S

# Frozen val spectrum: same construction as P2 (median of per-scene val medians) for fallback.
vp = [s["med_spec"] for s in info["val"] if s["med_spec"] is not None]
FROZEN = np.median(np.stack(vp), 0)

BINS = [(0, 8), (8, 32), (32, 1e9)]
iou = lambda t: float(t[0]) / t.sum() if t.sum() else float("nan")
rows = []
for split in ("val", "test"):
    pool = [(s["name"], r, c, mk) for s in info[split] for r, c, mk in s["donors"]]
    for s in info[split]:
        cw = sum(int(((g[0] == 1) & (g[1] == 2)).sum()) for r, c, g in s["clear"])
        if cw < 200: continue
        own = bool(s["donors"])
        own_spec = s["ncloud"] >= 1000
        # Soft-edge: scene MEAN when own; else FROZEN. Hard sanity: scene MEDIAN when own (P2).
        soft_spec = s["mean_spec"] if own_spec else FROZEN
        hard_spec = s["med_spec"] if own_spec else FROZEN
        keys = ["o_soft", "a_soft", "o_hard", "a_hard"]
        keys += [f"{m}_soft{j}" for m in ("o", "a") for j in range(3)]
        acc = {k: np.zeros(3, np.int64) for k in keys}
        for i, (r, c, g) in enumerate(s["clear"]):
            mk = s["donors"][i % len(s["donors"])][2] if own else pool[(i + 42) % len(pool)][3]
            x = raw(s["s2"], r, c); v = g[1] > 0; wt = g[1] == 2
            p0 = pred(x)
            # Soft paste
            A = soft_alpha(mk)
            x_soft = (1.0 - A)[None] * x + A[None] * soft_spec[:, None, None]
            p_soft = pred(x_soft.astype(np.float32))
            # Hard paste (sanity vs P2)
            x_hard = x.copy(); x_hard[:, mk] = hard_spec[:, None]
            p_hard = pred(x_hard)
            unp = v & ~mk
            acc["o_soft"] += cm(p0, wt, unp); acc["a_soft"] += cm(p_soft, wt, unp)
            acc["o_hard"] += cm(p0, wt, unp); acc["a_hard"] += cm(p_hard, wt, unp)
            d = edt(~mk) if mk.any() else np.full(mk.shape, 1e9)
            for j, (lo, hi) in enumerate(BINS):
                bm = unp & (d > lo) & (d <= hi)
                acc[f"o_soft{j}"] += cm(p0, wt, bm)
                acc[f"a_soft{j}"] += cm(p_soft, wt, bm)
        row = dict(split=split, scene=s["name"], own_mask=own, own_spectrum=own_spec,
                   gap_soft=(iou(acc["o_soft"]) - iou(acc["a_soft"])) * 100,
                   gap_hard=(iou(acc["o_hard"]) - iou(acc["a_hard"])) * 100)
        for j in range(3):
            row[f"gap_soft_bin{j}"] = (iou(acc[f"o_soft{j}"]) - iou(acc[f"a_soft{j}"])) * 100
        rows.append(row)
        print("SOFT", json.dumps(row), flush=True)

out = dict(rule="P2_SOFT_EDGE_RULE.md",
           soft_kernel=dict(kind="gaussian_filter", sigma=2.0, truncate=3.0),
           spectrum="scene_mean_if_ncloud>=1000_else_frozen_val_median_of_medians",
           frozen_val_spectrum=FROZEN.tolist(),
           rows=rows)
json.dump(out, open("p2_soft_edge_results.json", "w"), indent=1)
print("fin p2_soft_edge", len(rows), flush=True)
