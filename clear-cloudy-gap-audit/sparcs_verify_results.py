"""Record integrity checks only; no inference or interval estimation."""
from pathlib import Path
import hashlib,json,subprocess

A=Path(__file__).resolve().parent;R=A/'results';BASELINE='a1d44e2'
def load(name):
 o=json.loads((R/name).read_text());assert o['dataset']=='Landsat 8 SPARCS' and o['doi']=='10.5066/F7FB5146',name
 return o
def hashes(p):
 b=p.read_bytes();return {hashlib.sha256(b).hexdigest(),hashlib.sha256(b.replace(b'\r\n',b'\n')).hexdigest()}
p=load('sparcs_label_preflight.json');m=load('sparcs_input_manifest.json');c=load('sparcs_prithvi_composition.json');s=load('sparcs_prithvi_same_tile.json');a=load('sparcs_prithvi_aggregator.json');t=load('sparcs_prithvi_tiles.json');v=load('sparcs_runtime_provenance.json')
assert p['rule_sha256'] in hashes(A/'SPARCS_FROZEN_CONTROL_RULE.md')
assert p['manifest_sha256'] in hashes(R/'sparcs_input_manifest.json')
assert m['complete'] and m['chip_count']==80 and p['chip_count']==80
assert len(t['rows'])==p['eligible_tiles']==a['eligible_windows']
assert len({r['tile_id'] for r in t['rows']})==len(t['rows'])
assert a['checkpoint_unchanged'] and c['model']['frozen'] and not c['model']['training']
for name,h in v['script_sha256'].items():assert h in hashes(A/name),name
assert v['shared_controls_sha256'] in hashes(A/'c2sms_controls.py')
for row in t['rows']:
 assert row['all'][0]+row['all'][2]==row['gt_water_pixels']
 assert sum(row['all'])<=row['scoreable_pixels']
for kind,result in c['contrasts'].items():
 assert result['summary']==a['composition'][kind]
 assert result['summary']['scene_count']==sum(r['eligible'] for r in result['rows'])
 for row in result['rows']:
  if 'matching' not in row:continue
  pairs=row['matching']['pairs']
  assert len(pairs)==row['matching']['matched_pairs']==row['clear_tiles']==row['cloudy_tiles']
  assert len({r['clear'] for r in pairs})==len(pairs)==len({r['cloudy'] for r in pairs})
  assert all(r['abs_water_fraction_difference']<=.05 for r in pairs)
  if 'ndvi' in kind:assert all(r['abs_ndvi_difference']<=.05 for r in pairs)
assert s['summary']==a['same_tile']
for row in s['rows']:
 for counts in row['confusions'].values():
  assert counts[0]+counts[2]==row['common_GT_water_pixels']
  assert sum(counts)<=row['common_scoreable_pixels']
 if row['eligible']:assert abs(row['gap_cloud_pts']-row['gap_land_pts']-row['cloud_minus_land_pts'])<1e-10
paths=subprocess.check_output(['git','-C',str(A.parent),'ls-tree','-r','--name-only',BASELINE],text=True).splitlines();count=0
for name in paths:
 if name.endswith('.json'):
  old=subprocess.check_output(['git','-C',str(A.parent),'show',BASELINE+':'+name])
  assert old.replace(b'\r\n',b'\n')==(A.parent/name).read_bytes().replace(b'\r\n',b'\n'),name
  count+=1
print('PASS: source/input hashes, eligible counts, confusion conservation, fixed unique matching, identical paired scoring surfaces, copied summaries and frozen checkpoint records.')
print('PASS:',count,'previously tracked JSON files unchanged versus',BASELINE)
