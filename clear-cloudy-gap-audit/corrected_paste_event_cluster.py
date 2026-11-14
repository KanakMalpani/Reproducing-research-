"""Cluster existing corrected-paste gaps, preserving stored point estimates."""
import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np

A = Path(__file__).resolve().parent
R = A / 'results'
OUT = R / 'corrected_paste_event_cluster_2026_11_15.json'
SOURCES = {'WF2_UNet': R / 'pooled_cloud_noncloud_placebo.json',
           'Prithvi_Sen1Floods11': R / 'prithvi_same_protocol.json'}
assert (A / 'CORRECTED_PASTE_EVENT_RULE.md').exists(), 'STOP missing rule'
assert not OUT.exists(), 'STOP output already exists; do not replace JSON'
assert all(p.exists() for p in SOURCES.values()), 'STOP missing input'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
hashes = {model: digest(p) for model, p in SOURCES.items()}
data = {model: json.loads(p.read_text()) for model, p in SOURCES.items()}
keys = [[(r['split'], r['scene']) for r in obj['rows']] for obj in data.values()]
assert keys[0] == keys[1], 'STOP model scene keys/order differ'
stats = ('gap_cloud_pts', 'gap_placebo_pts', 'cloud_minus_placebo_pts')
output = {}
for model, blob in data.items():
    output[model] = {}
    for split in ('val_test', 'val', 'test'):
        rows = [r for r in blob['rows'] if split == 'val_test' or r['split'] == split]
        buckets = {}
        for i, row in enumerate(rows):
            m = re.match(r'^EMSR\d+', row['scene'])
            cid = m.group(0) if m else row['scene']
            buckets.setdefault(cid, []).append(i)
        cluster_keys = list(buckets)
        output[model][split] = {}
        for stat in stats:
            stored = blob['summary'][split][stat]
            assert stored['scenes'] == len(rows), 'STOP scene count mismatch'
            values = np.asarray([r[stat] for r in rows], dtype=float)
            assert np.isfinite(values).all(), 'STOP nonfinite gap'
            rng = np.random.default_rng(42)
            ordinary = [float(rng.choice(values, len(values)).mean()) for _ in range(5000)]
            scene_ci = np.percentile(ordinary, [2.5, 97.5])
            assert np.allclose(scene_ci, stored['ci95'], rtol=0, atol=1e-6), 'STOP scene interval mismatch'
            rng = np.random.default_rng(42)
            draws = []
            for _ in range(5000):
                picked = rng.choice(len(cluster_keys), len(cluster_keys), replace=True)
                ids = [i for k in picked for i in buckets[cluster_keys[k]]]
                draws.append(float(values[ids].mean()))
            cluster_ci = np.percentile(draws, [2.5, 97.5])
            if all(len(ids) == 1 for ids in buckets.values()):
                assert np.allclose(cluster_ci, scene_ci, rtol=0, atol=1e-9)
            output[model][split][stat] = dict(
                scenes=len(rows), activations=len(buckets),
                scene_mean_pts_stored=stored['mean_gap_pts'],
                scene_bootstrap_ci95_stored=stored['ci95'],
                activation_cluster_bootstrap_ci95=cluster_ci.tolist(),
                cluster_members={k: [rows[i]['scene'] for i in ids] for k, ids in buckets.items()},
                stored_scene_interval_reproduced=True)
            print(model, split, stat, 'stored mean', stored['mean_gap_pts'], 'cluster CI', cluster_ci.tolist(), flush=True)
assert all(digest(p) == hashes[model] for model, p in SOURCES.items())
result = dict(rule='CORRECTED_PASTE_EVENT_RULE.md', seed=42, bootstrap_draws=5000,
              point_estimates='Copied verbatim from source summaries; no new scene or event-weighted point estimate',
              input_sha256=hashes, input_unchanged=True, python=platform.python_version(),
              numpy=np.__version__, results=output)
with OUT.open('x') as f:
    json.dump(result, f, indent=2)
print('CORRECTED EVENT FOLLOWUP COMPLETE', OUT, flush=True)
