import os, sys, json, pathlib, torch, subprocess
import numpy as np, rasterio
from rasterio.windows import Window
audit=pathlib.Path(__file__).resolve().parent
os.chdir(audit)
results=audit/'results'
results.mkdir(exist_ok=True)
wf_root=pathlib.Path(os.environ['WF_ROOT'])
assert wf_root.exists() and pathlib.Path(os.environ['WF_CKPT']).exists(), 'STOP: missing WF_ROOT or WF_CKPT'
sys.path.insert(0,str(audit))
from wf_data import scenes, NORM
manifest=json.loads((results/'download_manifest.json').read_text())
assert manifest['complete'], 'STOP: missing download'
subprocess.run([sys.executable,'-m','pip','install','-q','rasterio'],check=True)
sys.path.insert(0,str(audit))
import numpy as np, rasterio
from wf_data import scenes
print('SCENES', {sp:len(scenes(sp)) for sp in ('val','test')},flush=True)
assert len(scenes('val'))==16 and len(scenes('test'))==18
os.environ['OUT'] = 'results/cloud_iou_gpu'
source = (audit/'cloud_iou_pilot.py').read_text()
source = source.replace('m.eval()', "m.eval(); m.to('cuda')")
source = source.replace('m(torch.from_numpy(x)[None])', "m(torch.from_numpy(x)[None].to('cuda'))")
source = source.replace('[0].numpy()', '[0].cpu().numpy()')
saved_argv = sys.argv[:]
sys.argv = ['cloud_iou_pilot.py','test','val']
try:
    exec(compile(source,'cloud_iou_pilot_gpu','exec'),{'__name__':'__main__'})
finally:
    sys.argv = saved_argv
new = json.loads((results/'cloud_iou_gpu_tiles.json').read_text())
ref = json.loads((audit/'cloud_iou_tiles.json').read_text())
key = lambda x:(x['split'],x['scene'],x['r'],x['c'])
a,b = {key(x):x for x in new},{key(x):x for x in ref}
comparison = {'tiles':len(a),'reference_tiles':len(b),'same_tile_keys':set(a)==set(b)}
if comparison['same_tile_keys']:
    changes = [sum(abs(a[k][v][i]-b[k][v][i]) for v in ('all','clear','under_cloud') for i in range(3)) for k in a]
    comparison['tiles_with_count_differences'] = sum(x>0 for x in changes)
    comparison['max_absolute_count_difference'] = max(changes)
(results/'gpu_inference_comparison.json').write_text(json.dumps(comparison,indent=2))
print('INFERENCE COMPARISON',json.dumps(comparison),flush=True)