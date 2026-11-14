"""Frozen C2S-MS Prithvi inference plus all three locked controls."""
import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml
from terratorch.tasks import SemanticSegmentationTask
from terratorch.datamodules.sen1floods11 import MEANS, STDS

A = Path(__file__).resolve().parent
sys.path.insert(0, str(A))
from c2sms_data import read_chip, masks, digest
from c2sms_controls import cm, iou, composition, aggregate, DATASET, DOI
R = A / 'results'
OUTPUTS = ['c2sms_floods_prithvi_composition.json', 'c2sms_floods_prithvi_same_tile.json',
           'c2sms_floods_prithvi_aggregator.json', 'c2sms_floods_prithvi_tiles.json']
assert not any((R / name).exists() for name in OUTPUTS), 'STOP completed output already exists'
pre = json.loads((R / 'c2sms_floods_label_preflight.json').read_text())
manifest = json.loads((R / 'c2sms_floods_download_manifest.json').read_text())
assert manifest['complete'] and pre['dataset'] == DATASET
assert pre['rule_sha256'] == digest(A / 'C2SMS_FROZEN_CONTROL_RULE.md'), 'STOP rule changed after preflight'
checkpoint = A / 'prithvi_assets' / 'Prithvi-EO-V2-300M-TL-Sen1Floods11.pt'
before = digest(checkpoint)
assert before == '76eed77d8bd543ae441308b80e8408502243a871cb8a338ddc8220ed96dfc270'
cfg = yaml.safe_load((A / 'prithvi_assets' / 'config.yaml').read_text())
args = cfg['model']['init_args']
args['model_args']['backbone_pretrained'] = False
torch.set_num_threads(3)
torch.manual_seed(0)
task = SemanticSegmentationTask(**args)
weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
if 'state_dict' in weights:
    weights = weights['state_dict']
if all(k.startswith('model.') for k in weights):
    task.load_state_dict(weights, strict=True)
else:
    task.model.load_state_dict(weights, strict=True)
del weights
model = task.model.cuda().eval()
model.requires_grad_(False)
assert not model.training
bands = cfg['data']['init_args']['bands']
means = np.asarray([MEANS[b] for b in bands], np.float32)[:, None, None]
stds = np.asarray([STDS[b] for b in bands], np.float32)[:, None, None]

@torch.inference_mode()
def predict(raws):
    x = np.stack([(raw.astype(np.float32) / 10000. - means) / stds for raw in raws])
    output = model(torch.from_numpy(x).cuda(), temporal_coords=None, location_coords=None).output
    assert output.shape == (len(raws), 2, 256, 256), str(output.shape)
    assert torch.isfinite(output).all(), 'STOP nonfinite frozen-model logits'
    return (output.argmax(1).cpu().numpy() == 1)

def tile_data(row):
    raw, water, cloud, _ = read_chip(row['prefix'])
    r, c = row['r'], row['c']
    return raw[:, r:r + 256, c:c + 256], water[r:r + 256, c:c + 256], cloud[r:r + 256, c:c + 256]

start = time.time()
tiles = [row for row in pre['tiles'] if row['eligible']]
events = sorted(pre['fills'])
natural, original = [], {}
for offset in range(0, len(tiles), 2):
    batch = tiles[offset:offset + 2]
    inputs = [tile_data(t) for t in batch]
    predictions = predict([x[0][:6] for x in inputs])
    for row, (raw, water, cloud), p in zip(batch, inputs, predictions):
        _, _, valid = masks(raw, water, cloud)
        w = water == 1
        record = dict(row, all=cm(p, w, valid).tolist(), clear=cm(p, w, valid & (cloud == 0)).tolist(),
                      under_cloud=cm(p, w, valid & (cloud == 1)).tolist())
        natural.append(record)
        original[row['tile_id']] = np.packbits(p)
    if (offset + len(batch)) % 100 == 0 or offset + len(batch) == len(tiles):
        print('C2SMS FROZEN NATURAL', offset + len(batch), '/', len(tiles), 'seconds', round(time.time() - start, 1), flush=True)

composition_result = composition(natural, events)
donors = sorted([t for t in tiles if .1 <= t['cloud_frac'] <= .6], key=lambda r: r['tile_id'])
same_rows = []
for event in events:
    clear = [t for t in tiles if t['event'] == event and t['cloud_frac'] < .01]
    own = [t for t in donors if t['event'] == event]
    acc = {kind: np.zeros(3, np.int64) for kind in ('original', 'cloud', 'land')}
    common_pixels, common_water = 0, 0
    assignments = []
    for i, row in enumerate(clear):
        if not donors:
            break
        donor = own[i % len(own)] if own else donors[(i + 42) % len(donors)]
        _, _, donor_cloud = tile_data(donor)
        mask = donor_cloud == 1
        raw, water, cloud = tile_data(row)
        image, _, _ = masks(raw, water, cloud)
        valid = image & np.isin(water, [0, 1]) & (cloud == 0) & ~mask
        w = water == 1
        common_pixels += int(valid.sum())
        common_water += int((valid & w).sum())
        p = np.unpackbits(original[row['tile_id']]).reshape(256, 256).astype(bool)
        acc['original'] += cm(p, w, valid)
        painted = []
        for kind in ('cloud', 'land'):
            x = raw[:6].astype(np.float32).copy()
            x[:, mask] = np.asarray(pre['fills'][event][kind]['spectrum'], np.float32)[:, None]
            painted.append(x)
        predictions = predict(painted)
        for kind, p in zip(('cloud', 'land'), predictions):
            acc[kind] += cm(p, w, valid)
        assignments.append(dict(target=row['tile_id'], donor=donor['tile_id'], own_event=bool(own),
                                painted_pixels=int(mask.sum()), common_scoreable_pixels=int(valid.sum())))
    scores = {k: iou(v) for k, v in acc.items()}
    reasons = []
    if not clear:
        reasons.append('no_clear_tiles')
    if not donors:
        reasons.append('no_eligible_C2S_MS_cloud_donor_masks')
    if common_water < 200:
        reasons.append('common_unpainted_surface_less_than_200_GT_water_pixels')
    if any(value is None for value in scores.values()):
        reasons.append('undefined_common_surface_IoU')
    row = dict(dataset=DATASET, scene=event, event=event, clear_tiles=len(clear), eligible=not reasons,
               exclusions=reasons, common_scoreable_pixels=common_pixels, common_GT_water_pixels=common_water,
               confusions={k: v.tolist() for k, v in acc.items()}, iou=scores, fills=pre['fills'][event], assignments=assignments,
               gap_cloud_pts=100 * (scores['original'] - scores['cloud']) if not reasons else None,
               gap_land_pts=100 * (scores['original'] - scores['land']) if not reasons else None)
    row['cloud_minus_land_pts'] = row['gap_cloud_pts'] - row['gap_land_pts'] if not reasons else None
    same_rows.append(row)
    print('C2SMS CONSTRUCTED FILL', event, 'eligible', row['eligible'], 'seconds', round(time.time() - start, 1), flush=True)

aggregator = aggregate(natural, events, composition_result, same_rows)
aggregator['source_acquisitions_observed'] = pre['source_acquisitions']
aggregator['eligible_windows'] = len(tiles)
metadata = dict(model_id=manifest['model_id'], revision=manifest['model_revision'], checkpoint_sha256=before,
                frozen=True, training=False, native_inference_shape=[256, 256], precision='float32',
                input_bands=['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'], inference_batch_size=2,
                input_scale='DN/10000 then official Sen1Floods11 means/stds', prediction='two-class argmax; no metadata or TTA')
composition_result['model'] = metadata
same_result = dict(dataset=DATASET, doi=DOI, model=metadata, rule='C2SMS_FROZEN_CONTROL_RULE.md',
                   caption='Constructed constant fill on clear C2S-MS tiles, scored on unpainted pixels; it is not natural cloud.',
                   spectra='C2S-MS only; exact per-band medians; noncloud land requires LabelCloud0 and LabelWater0',
                   rows=same_rows, summary=aggregator['same_tile'])
assert digest(checkpoint) == before, 'STOP frozen checkpoint changed'
aggregator['checkpoint_unchanged'] = True
aggregator['runtime_seconds'] = time.time() - start
outputs = [composition_result, same_result, aggregator, dict(dataset=DATASET, doi=DOI, model=metadata, rows=natural)]
for name, obj in zip(OUTPUTS, outputs):
    with (R / name).open('x') as f:
        json.dump(obj, f, indent=2)
np.savez_compressed(R / 'c2sms_floods_original_predictions.npz', tile_ids=np.asarray(list(original)),
                    packed_predictions=np.stack(list(original.values())))
print('C2SMS THREE CONTROLS COMPLETE', json.dumps(dict(composition=aggregator['composition'], same_tile=aggregator['same_tile'],
                                                     water_accuracy=aggregator['water_accuracy']['summary'])), flush=True)
