"""P2 spatially varying spectrum paste (P2_VARYING_SPECTRUM_RULE.md). Writes p2_varying_results.json."""
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
                      mean_spec=mean_spec, med_spec=med_spec, ncloud=n,
                      pool=np.concatenate(cloudpx, 1) if cloudpx else None))
        print("scan", split, name, len(cl), len(cd), n, flush=True)
    info[split] = S

# Frozen val spectrum: same construction as P2 (median of per-scene val medians) for fallback.
vp = [s["med_spec"] for s in info["val"] if s["med_spec"] is not None]
FROZEN = np.median(np.stack(vp), 0)
FROZEN_POOL = np.concatenate([s['pool'] for s in info['val'] if s['pool'] is not None], 1)
RNG = np.random.default_rng(42)

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
        vpool = s["pool"] if own_spec else FROZEN_POOL
        hard_spec = s["med_spec"] if own_spec else FROZEN
        keys = ["o", "a_sv", "a_hv", "a_hc"] + [f"{m}{j}" for m in ("o", "sv", "hv") for j in range(3)]
        acc = {k: np.zeros(3, np.int64) for k in keys}
        for i, (r, c, g) in enumerate(s["clear"]):
            mk = s["donors"][i % len(s["donors"])][2] if own else pool[(i + 42) % len(pool)][3]
            x = raw(s["s2"], r, c); v = g[1] > 0; wt = g[1] == 2
            p0 = pred(x)
            F = vpool[:, RNG.integers(0, vpool.shape[1], size=mk.shape)].astype(np.float32)
            A = soft_alpha(mk)
            p_sv = pred(((1.0 - A)[None] * x + A[None] * F).astype(np.float32))
            xh = x.copy(); xh[:, mk] = F[:, mk]; p_hv = pred(xh)
            xc = x.copy(); xc[:, mk] = hard_spec[:, None]; p_hc = pred(xc)
            unp = v & ~mk
            acc["o"] += cm(p0, wt, unp); acc["a_sv"] += cm(p_sv, wt, unp)
            acc["a_hv"] += cm(p_hv, wt, unp); acc["a_hc"] += cm(p_hc, wt, unp)
            d = edt(~mk) if mk.any() else np.full(mk.shape, 1e9)
            for j, (lo, hi) in enumerate(BINS):
                bm = unp & (d > lo) & (d <= hi)
                acc[f"o{j}"] += cm(p0, wt, bm)
                acc[f"sv{j}"] += cm(p_sv, wt, bm); acc[f"hv{j}"] += cm(p_hv, wt, bm)
        G = lambda k: (iou(acc["o"]) - iou(acc[k])) * 100
        row = dict(split=split, scene=s["name"], own_mask=own, own_spectrum=own_spec, pool_n=int(vpool.shape[1]),
                   gap_soft_var=G("a_sv"), gap_hard_var=G("a_hv"), gap_hard_const=G("a_hc"))
        for j in range(3):
            row[f"gap_soft_var_bin{j}"] = (iou(acc[f"o{j}"]) - iou(acc[f"sv{j}"])) * 100
            row[f"gap_hard_var_bin{j}"] = (iou(acc[f"o{j}"]) - iou(acc[f"hv{j}"])) * 100
        rows.append(row)
        print("VAR", json.dumps(row), flush=True)

out = dict(rule="P2_VARYING_SPECTRUM_RULE.md", seed=42,
           soft_kernel=dict(kind="gaussian_filter", sigma=2.0, truncate=3.0),
           spectrum="per_pixel_sample_scene_pool_if_ncloud>=1000_else_frozen_val_pool",
           frozen_val_spectrum=FROZEN.tolist(),
           rows=rows)
json.dump(out, open("p2_varying_results.json", "w"), indent=1)
print("fin p2_varying", len(rows), flush=True)
