"""WorldFloods v2 loader: windowed 256x256 tiles from S2 (first 13 bands) + gt (cloud, water)."""
import glob, os, numpy as np, rasterio
from rasterio.windows import Window
ROOT = os.environ.get("WF_ROOT", "/home/box/worldfloods")
# ml4floods SENTINEL2_NORMALIZATION (mean, std) for B1..B12
NORM = np.array([[3787.06,2634.44],[3758.07,2794.10],[3238.08,2549.49],[3418.90,2811.78],[3450.23,2776.93],
 [4030.95,2632.14],[4164.17,2657.43],[3981.96,2500.48],[4226.75,2589.29],[1868.30,1820.90],
 [399.39,761.36],[2391.66,1500.03],[1790.32,1241.98]], np.float32)
def scenes(split):
    out=[]
    for g in sorted(glob.glob(f"{ROOT}/{split}/gt/*.tif")):
        s=g.replace("/gt/","/S2/")
        if os.path.exists(s):
            try: rasterio.open(s).close(); out.append((os.path.basename(g)[:-4], s, g))
            except Exception: pass
    return out
def tiles(s2_path, gt_path, size=256, stride=512):
    with rasterio.open(s2_path) as S, rasterio.open(gt_path) as G:
        H, W = G.shape
        for r in range(0, H-size+1, stride):
            for c in range(0, W-size+1, stride):
                w = Window(c, r, size, size)
                gt = G.read(window=w)  # (2,h,w): cloud {0 inv,1 clear,2 cloud}, water {0 inv,1 land,2 water}
                if (gt[1] > 0).mean() < 0.5: continue
                x = S.read(list(range(1, 14)), window=w).astype(np.float32)
                x = (x - NORM[:, 0, None, None]) / NORM[:, 1, None, None]
                yield r, c, x, gt
