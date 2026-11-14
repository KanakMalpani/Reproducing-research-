"""C2S-MS-only raster reader and fixed pre-inference eligibility census."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

A = Path(__file__).resolve().parent
R = A / 'results'
ROOT = Path(os.environ.get('C2SMS_ROOT', '/content/c2sms_data'))
BANDS = ('B2', 'B3', 'B4', 'B8A', 'B11', 'B12', 'B8')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

@lru_cache(maxsize=4)
def read_chip(prefix):
    base = ROOT / prefix
    with rasterio.open(base / 'B2.tif') as ref:
        assert ref.shape == (512, 512) and ref.count == 1
        crs, transform, bounds = ref.crs, ref.transform, ref.bounds
    raw, shapes = [], {}
    for band in BANDS:
        with rasterio.open(base / (band + '.tif')) as src:
            assert src.count == 1 and src.crs == crs
            assert np.allclose(src.bounds, bounds, rtol=0, atol=1e-6), 'STOP band extent mismatch'
            assert src.dtypes[0] == 'uint16', 'STOP unexpected imagery dtype'
            shapes[band] = list(src.shape)
            raw.append(src.read(1, out_shape=(512, 512), resampling=Resampling.nearest))
    labels = []
    for name, codes in [('LabelWater', [0, 1, 255]), ('LabelCloud', [0, 1, 2, 255])]:
        with rasterio.open(base / (name + '.tif')) as src:
            assert src.count == 1 and src.shape == (512, 512)
            assert src.crs == crs and src.transform == transform, 'STOP label alignment mismatch'
            x = src.read(1)
            assert np.isin(x, codes).all(), 'STOP unexpected ' + name + ' code'
            labels.append(x)
    return np.stack(raw), labels[0], labels[1], shapes

def masks(raw, water, cloud):
    image = np.all(raw[:6] > 0, axis=0)
    cloud_valid = image & np.isin(cloud, [0, 1])
    score = cloud_valid & np.isin(water, [0, 1])
    return image, cloud_valid, score

def add_hist(hist, raw, mask):
    for i in range(6):
        hist[i] += np.bincount(raw[i][mask], minlength=65536)

def median_hist(hist):
    n = int(hist[0].sum())
    assert n >= 1000, 'STOP insufficient C2S-MS spectrum pixels'
    cs = np.cumsum(hist, axis=1)
    return [float((np.searchsorted(row, (n - 1) // 2 + 1) + np.searchsorted(row, n // 2 + 1)) / 2) for row in cs]

def prepare():
    inventory = json.loads((R / 'c2sms_floods_input_inventory.json').read_text())
    manifest = json.loads((R / 'c2sms_floods_download_manifest.json').read_text())
    assert manifest['complete'] and ROOT.exists()
    tiles, histograms, census, geometry = [], {}, {}, {band: Counter() for band in BANDS}
    for number, prefix in enumerate(inventory['chip_folders']):
        event = prefix.split('/')[2]
        chip = prefix.rsplit('/', 1)[-1]
        raw, water, cloud, shapes = read_chip(prefix)
        for band, shape in shapes.items():
            geometry[band]['x'.join(map(str, shape))] += 1
        image, cloud_valid, score = masks(raw, water, cloud)
        if event not in histograms:
            histograms[event] = {k: np.zeros((6, 65536), np.int64) for k in ('cloud', 'land')}
            census[event] = {str(k): dict(pixels=0, image_valid_pixels=0, water_known_pixels=0, water_pixels=0, water_nodata_pixels=0) for k in (0, 1, 2, 255)}
        add_hist(histograms[event]['cloud'], raw, image & (cloud == 1))
        add_hist(histograms[event]['land'], raw, image & (cloud == 0) & (water == 0))
        for code, group in census[event].items():
            mask = cloud == int(code)
            group['pixels'] += int(mask.sum())
            group['image_valid_pixels'] += int((mask & image).sum())
            group['water_known_pixels'] += int((mask & image & np.isin(water, [0, 1])).sum())
            group['water_pixels'] += int((mask & image & (water == 1)).sum())
            group['water_nodata_pixels'] += int((mask & image & (water == 255)).sum())
        for r in (0, 256):
            for c in (0, 256):
                sl = np.s_[r:r + 256, c:c + 256]
                v, cv = score[sl], cloud_valid[sl]
                g, w, x = cloud[sl], water[sl], raw[:, r:r + 256, c:c + 256]
                reasons = []
                if cv.mean() < .5:
                    reasons.append('less_than_half_image_and_cloud_label_valid')
                if v.mean() < .5:
                    reasons.append('less_than_half_joint_water_cloud_label_valid')
                clear = v & (g == 0) & (x[6] > 0)
                ndvi = (x[6].astype(np.float32) - x[2].astype(np.float32)) / (x[6].astype(np.float32) + x[2].astype(np.float32) + 1e-6)
                tiles.append(dict(tile_id=prefix + '/' + str(r) + '/' + str(c), dataset='C2S-MS Floods', event=event,
                                  scene=event, source_acquisition=chip.rsplit('_', 1)[0], chip=chip, prefix=prefix, r=r, c=c,
                                  eligible=not reasons, exclusions=reasons, image_cloud_valid_pixels=int(cv.sum()),
                                  scoreable_pixels=int(v.sum()), gt_water_pixels=int(((w == 1) & v).sum()),
                                  cloud_frac=float(((g == 1) & cv).sum() / cv.sum()) if cv.any() else None,
                                  water_frac=float(((w == 1) & v).sum() / v.sum()) if v.any() else None,
                                  ndvi_clear=float(ndvi[clear].mean()) if clear.any() else None))
        if (number + 1) % 50 == 0:
            print('C2SMS LABEL PREFLIGHT', number + 1, '/', len(inventory['chip_folders']), flush=True)
    pooled = {kind: sum(h[kind] for h in histograms.values()) for kind in ('cloud', 'land')}
    frozen = {kind: median_hist(h) for kind, h in pooled.items()}
    fills = {}
    for event, groups in histograms.items():
        fills[event] = {}
        for kind, hist in groups.items():
            count = int(hist[0].sum())
            fills[event][kind] = dict(pixels=count, source='own_event' if count >= 1000 else 'pooled_C2S_MS',
                                     spectrum=median_hist(hist) if count >= 1000 else frozen[kind])
    assert tiles and any(t['eligible'] for t in tiles), 'STOP no eligible C2S-MS tiles'
    return dict(dataset='C2S-MS Floods', doi='10.34911/rdnt.oz32gz', rule='C2SMS_FROZEN_CONTROL_RULE.md',
                rule_sha256=digest(A / 'C2SMS_FROZEN_CONTROL_RULE.md'), input_inventory_sha256=digest(R / 'c2sms_floods_input_inventory.json'),
                chip_folders=len(inventory['chip_folders']), event_folders=len(histograms),
                event_definition='scene equals event folder; source acquisitions retained separately',
                source_acquisitions=len({(t['event'], t['source_acquisition']) for t in tiles}),
                tile_candidates=len(tiles), eligible_tiles=sum(t['eligible'] for t in tiles),
                exclusion_counts=dict(Counter(reason for t in tiles for reason in t['exclusions'])),
                geometry={k: dict(v) for k, v in geometry.items()}, label_coverage_by_event=census,
                pooled_frozen_spectra=frozen, fills=fills, tiles=tiles)

if __name__ == '__main__':
    output = R / 'c2sms_floods_label_preflight.json'
    assert not output.exists(), 'STOP preflight already exists'
    obj = prepare()
    with output.open('x') as f:
        json.dump(obj, f, indent=2)
    print('C2SMS LABEL PREFLIGHT COMPLETE', json.dumps({k: obj[k] for k in ('chip_folders', 'event_folders', 'source_acquisitions', 'tile_candidates', 'eligible_tiles', 'exclusion_counts', 'geometry')}), flush=True)
