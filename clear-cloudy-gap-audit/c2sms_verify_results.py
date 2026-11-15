"""Verify exported records without rerunning inference or estimating intervals."""
import hashlib
import json
from pathlib import Path
import subprocess

A = Path(__file__).resolve().parent
REPO = A.parent
BASELINE = 'e79db6acacae62c405ccd3daf2d1d12ea343bc41'
R = A / 'results'
def read(name):
    obj = json.loads((R / name).read_text())
    assert obj['dataset'] == 'C2S-MS Floods'
    assert obj['doi'] == '10.34911/rdnt.oz32gz'
    return obj
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

pre = read('c2sms_floods_label_preflight.json')
manifest = read('c2sms_floods_download_manifest.json')
comp = read('c2sms_floods_prithvi_composition.json')
same = read('c2sms_floods_prithvi_same_tile.json')
agg = read('c2sms_floods_prithvi_aggregator.json')
tiles = read('c2sms_floods_prithvi_tiles.json')['rows']
prov = read('c2sms_floods_runtime_provenance.json')
rule_bytes = (A/'C2SMS_FROZEN_CONTROL_RULE.md').read_bytes()
assert pre['rule_sha256'] in {hashlib.sha256(rule_bytes).hexdigest(), hashlib.sha256(rule_bytes.replace(b'\r\n', b'\n')).hexdigest()}
inventory_bytes = (R/'c2sms_floods_input_inventory.json').read_bytes()
assert pre['input_inventory_sha256'] in {hashlib.sha256(inventory_bytes).hexdigest(), hashlib.sha256(inventory_bytes.replace(b'\r\n', b'\n')).hexdigest()}
assert manifest['complete'] and len(manifest['files']) == 8100
assert len(tiles) == pre['eligible_tiles'] == agg['eligible_windows']
assert len({t['tile_id'] for t in tiles}) == len(tiles)
assert agg['checkpoint_unchanged'] and comp['model']['frozen'] and not comp['model']['training']
for name, digest in prov['script_sha256'].items():
    blob = (A/name).read_bytes()
    assert digest in {hashlib.sha256(blob).hexdigest(), hashlib.sha256(blob.replace(b'\r\n', b'\n')).hexdigest()}, name
for t in tiles:
    assert all(t['all'][i] == t['clear'][i] + t['under_cloud'][i] for i in range(3))
    assert t['all'][0] + t['all'][2] == t['gt_water_pixels']
    assert sum(t['all']) <= t['scoreable_pixels']
for kind, result in comp['contrasts'].items():
    kept = [r for r in result['rows'] if r['eligible']]
    assert len(kept) == result['summary']['scene_count']
    assert result['summary'] == agg['composition'][kind]
    for row in result['rows']:
        if 'matching' not in row:
            continue
        pairs = row['matching']['pairs']
        assert len(pairs) == row['matching']['matched_pairs'] == row['clear_tiles'] == row['cloudy_tiles']
        assert len({p['clear'] for p in pairs}) == len(pairs)
        assert len({p['cloudy'] for p in pairs}) == len(pairs)
        assert all(p['abs_water_fraction_difference'] <= .05 for p in pairs)
        if 'ndvi' in kind:
            assert all(p['abs_ndvi_difference'] <= .05 for p in pairs)
for row in same['rows']:
    for counts in row['confusions'].values():
        assert counts[0]+counts[2] == row['common_GT_water_pixels']
        assert sum(counts) <= row['common_scoreable_pixels']
    if row['eligible']:
        assert abs(row['gap_cloud_pts']-row['gap_land_pts']-row['cloud_minus_land_pts']) < 1e-10
assert same['summary'] == agg['same_tile']
tracked = subprocess.check_output(['git','-C',str(REPO),'ls-tree','-r','--name-only',BASELINE],text=True).splitlines()
unchanged = 0
for name in tracked:
    if name.endswith('.json'):
        old = subprocess.check_output(['git','-C',str(REPO),'show',BASELINE+':'+name])
        local = (REPO/name).read_bytes()
        assert old.replace(b'\r\n',b'\n') == local.replace(b'\r\n',b'\n'), name
        unchanged += 1
print('PASS: export integrity, fixed matching, common-surface water counts, identical summaries, checkpoint unchanged; preserved',unchanged,'tracked JSON files.')
print('PASS: C2S-MS records checked against fixed pre-results baseline', BASELINE)
