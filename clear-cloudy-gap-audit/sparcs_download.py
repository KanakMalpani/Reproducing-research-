"""Download and verify official SPARCS inputs in Colab; no inference."""
from pathlib import Path
import hashlib,json,zipfile,urllib.request,time

A=Path(__file__).resolve().parent
R=A/'results';R.mkdir(exist_ok=True)
ROOT=Path('/content/sparcs_data');ROOT.mkdir(exist_ok=True)
OUT=R/'sparcs_input_manifest.json'
assert not OUT.exists(),'STOP manifest exists'
URL='https://landsat.usgs.gov/cloud-validation/sparcs/l8cloudmasks.zip'
Z=ROOT/'l8cloudmasks.zip'
assert not Z.exists(),'STOP archive exists; inspect before resuming'
start=time.time()
with urllib.request.urlopen(URL,timeout=120) as response,Z.open('xb') as f:
 expected=response.headers.get('Content-Length');n=0
 while True:
  b=response.read(8*1024*1024)
  if not b:break
  f.write(b);n+=len(b)
  if n%(128*1024*1024)==0:print('SPARCS DOWNLOAD',n,flush=True)
assert not expected or n==int(expected),'STOP incomplete archive'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
records=[]
with zipfile.ZipFile(Z) as z:
 assert z.testzip() is None,'STOP ZIP CRC failure'
 for name in z.namelist():
  target=(ROOT/name).resolve();assert target.is_relative_to(ROOT.resolve()),'STOP unsafe archive path'
 z.extractall(ROOT)
 for p in sorted(ROOT.rglob('*')):
  if p.is_file() and p!=Z:records.append(dict(path=str(p),relative=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
data=sorted(ROOT.rglob('*_data.tif'))
assert len(data)==80,('STOP unexpected subscene count',len(data))
chips=[]
for p in data:
 stem=p.name[:-len('_data.tif')]
 mask=p.with_name(stem+'_mask.png')
 mtls=[q for q in p.parent.iterdir() if q.name.lower()==(stem+'_mtl.txt').lower()]
 assert mask.exists() and len(mtls)==1,('STOP missing mask/MTL',stem)
 chips.append(dict(chip=stem,event=stem.rsplit('_',1)[0],data=str(p),mask=str(mask),mtl=str(mtls[0])))
obj=dict(dataset='Landsat 8 SPARCS',doi='10.5066/F7FB5146',url=URL,archive_bytes=n,archive_sha256=sha(Z),integrity='HTTPS download; Content-Length where supplied; all ZIP CRCs; observed SHA256, not publisher-certified hash',complete=True,chip_count=len(chips),acquisition_count=len({c['event'] for c in chips}),chips=chips,files=records,elapsed_seconds=time.time()-start)
with OUT.open('x') as f:json.dump(obj,f,indent=2)
print('SPARCS INPUTS COMPLETE',obj['chip_count'],obj['acquisition_count'],n,obj['archive_sha256'],flush=True)
