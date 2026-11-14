"""Fixed composition matching and explicitly event-folder aggregation."""
import numpy as np

DATASET = 'C2S-MS Floods'
DOI = '10.34911/rdnt.oz32gz'

def cm(pred, water, valid):
    return np.array([(pred & water & valid).sum(), (pred & ~water & valid).sum(), (~pred & water & valid).sum()], np.int64)

def total(rows, field='all'):
    return np.asarray([r[field] for r in rows], np.int64).sum(0) if rows else np.zeros(3, np.int64)

def iou(counts):
    return float(counts[0] / sum(counts)) if sum(counts) else None

def bootstrap(values, seed):
    if not values:
        return dict(scene_count=0, scene_mean=None, ci95=None, reason='no eligible event-folder scenes')
    x = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    ci = np.percentile([rng.choice(x, len(x)).mean() for _ in range(5000)], [2.5, 97.5]).tolist()
    return dict(scene_count=len(x), scene_mean=float(x.mean()), ci95=ci,
                uncertainty='event-folder scene bootstrap; 5000 draws; seed' + str(seed),
                limitation='one event: degenerate descriptive interval' if len(x) == 1 else None)

def group_contrast(event, clear, cloudy):
    a, b = total(clear), total(cloudy)
    ia, ib = iou(a), iou(b)
    reasons = []
    if a[0] + a[2] < 200:
        reasons.append('clear_group_less_than_200_GT_water_pixels')
    if b[0] + b[2] < 200:
        reasons.append('cloudy_group_less_than_200_GT_water_pixels')
    if ia is None or ib is None:
        reasons.append('undefined_group_IoU')
    return dict(scene=event, event=event, clear_tiles=len(clear), cloudy_tiles=len(cloudy),
                clear_confusion=a.tolist(), cloudy_confusion=b.tolist(), clear_iou=ia, cloudy_iou=ib,
                eligible=not reasons, exclusions=reasons,
                gap_pts=100 * (ia - ib) if not reasons else None)

def matched(clear, cloudy, ndvi=False):
    missing = dict(clear_missing_ndvi=0, cloudy_missing_ndvi=0)
    if ndvi:
        missing = dict(clear_missing_ndvi=sum(r['ndvi_clear'] is None for r in clear),
                       cloudy_missing_ndvi=sum(r['ndvi_clear'] is None for r in cloudy))
        clear = [r for r in clear if r['ndvi_clear'] is not None]
        cloudy = [r for r in cloudy if r['ndvi_clear'] is not None]
    used, left, right, pairs = set(), [], [], []
    for t in sorted(cloudy, key=lambda r: r['water_frac']):
        candidates = []
        for i, u in enumerate(clear):
            if i in used:
                continue
            dw = abs(t['water_frac'] - u['water_frac'])
            dn = abs(t['ndvi_clear'] - u['ndvi_clear']) if ndvi else 0.
            if dw <= .05 and (not ndvi or dn <= .05):
                candidates.append((dw + dn, i, dw, dn))
        if not candidates:
            continue
        _, i, dw, dn = min(candidates)
        used.add(i)
        left.append(clear[i])
        right.append(t)
        pairs.append(dict(clear=clear[i]['tile_id'], cloudy=t['tile_id'], abs_water_fraction_difference=dw,
                          abs_ndvi_difference=dn if ndvi else None))
    accounting = dict(**missing, available_cloudy_tiles=len(cloudy), matched_pairs=len(pairs),
                      discarded_cloudy_tiles=len(cloudy) - len(pairs), clear_tiles_used=len(used), pairs=pairs)
    return left, right, accounting

def composition(natural, events):
    out = {k: [] for k in ('unmatched', 'water_fraction_match', 'water_fraction_plus_ndvi_match')}
    for event in events:
        rows = [r for r in natural if r['event'] == event]
        clear = [r for r in rows if r['cloud_frac'] < .01]
        cloudy = [r for r in rows if r['cloud_frac'] >= .1]
        out['unmatched'].append(group_contrast(event, clear, cloudy))
        for kind, ndvi in [('water_fraction_match', False), ('water_fraction_plus_ndvi_match', True)]:
            a, b, accounting = matched(clear, cloudy, ndvi)
            record = group_contrast(event, a, b)
            record['matching'] = accounting
            out[kind].append(record)
    result = {}
    for kind, rows in out.items():
        kept = [r for r in rows if r['eligible']]
        a = np.asarray([r['clear_confusion'] for r in kept]).sum(0) if kept else np.zeros(3)
        b = np.asarray([r['cloudy_confusion'] for r in kept]).sum(0) if kept else np.zeros(3)
        result[kind] = dict(rows=rows, summary=bootstrap([r['gap_pts'] for r in kept], 0),
                            pixel_pooled_secondary_gap_pts=100 * (iou(a) - iou(b)) if iou(a) is not None and iou(b) is not None else None)
    return dict(dataset=DATASET, doi=DOI, scene_definition='event folder; within-event matching',
                cloud_groups='LabelCloud0/1 only; clear tile fraction<.01, cloudy>=.1; shadows/no-data excluded',
                water_fraction_tolerance=.05, ndvi_tolerance=.05, minimum_group_GT_water_pixels=200,
                contrasts=result)

def aggregate(natural, events, composition_result, same_rows):
    event_accuracy = []
    for event in events:
        rs = [r for r in natural if r['event'] == event]
        counts = total(rs)
        eligible = counts[0] + counts[2] >= 200 and iou(counts) is not None
        event_accuracy.append(dict(event=event, scene=event, eligible=bool(eligible), tiles=len(rs),
                                   confusion=counts.tolist(), water_iou=iou(counts)))
    accuracy = bootstrap([r['water_iou'] for r in event_accuracy if r['eligible']], 0)
    accuracy['pixel_pooled_water_iou_secondary'] = iou(total(natural))
    same = {key: bootstrap([r[key] for r in same_rows if r['eligible']], 42)
            for key in ('gap_cloud_pts', 'gap_land_pts', 'cloud_minus_land_pts')}
    return dict(dataset=DATASET, doi=DOI, primary_aggregator='unweighted mean of eligible event-folder scene values',
                scene_definition='one event folder; not one chip or one source acquisition',
                bootstrap_unit='event folder; all dependent chip/windows retained within event',
                event_folders=len(events), composition={k: v['summary'] for k, v in composition_result['contrasts'].items()},
                same_tile=same, water_accuracy=dict(rows=event_accuracy, summary=accuracy),
                event_weighted_mean=None, event_weighted_mean_reason='primary scene unit already is an event folder; not a separately estimated point',
                pixel_pool_role='secondary pooled confusion; never substituted for scene mean',
                worldfloods_results_modified=False)
