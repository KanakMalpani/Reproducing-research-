"""Download only public validation/test audit inputs; stop on any failed download."""
import os,json,pathlib,urllib.request,urllib.parse,shutil,hashlib
root=pathlib.Path(__file__).resolve().parent
results=root/'results'; results.mkdir(exist_ok=True)
wf_root=pathlib.Path(os.environ['WF_ROOT'])
wf_ckpt=pathlib.Path(os.environ['WF_CKPT'])
def meta(url):
    with urllib.request.urlopen(url,timeout=120) as r: return json.load(r)
def fetch(repo,revision,item,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists() or target.stat().st_size!=item['size']:
        with urllib.request.urlopen(repo+'/resolve/'+revision+'/'+urllib.parse.quote(item['rfilename'],safe='/'),timeout=120) as src,target.open('wb') as dst: shutil.copyfileobj(src,dst)
    assert target.stat().st_size==item['size'], 'Incomplete download'
    h=hashlib.sha256()
    with target.open('rb') as f:
        for chunk in iter(lambda:f.read(8388608),b''): h.update(chunk)
    if item.get('lfs',{}).get('sha256'): assert h.hexdigest()==item['lfs']['sha256']
    return dict(path=str(target),bytes=target.stat().st_size,sha256=h.hexdigest())
dataset_revision='1f3faa2989e69930ac31d5c0a79fd224461123f8'
model_revision='3f52941fbb01b6f66eed71449b7586d6e168e035'
data=meta('https://huggingface.co/api/datasets/isp-uv-es/WorldFloodsv2/revision/'+dataset_revision+'?blobs=true')
model=meta('https://huggingface.co/api/models/isp-uv-es/ml4floods/revision/'+model_revision+'?blobs=true')
selected=[x for x in data['siblings'] if x['rfilename'].split('/')[0] in ('val','test') and x['rfilename'].split('/')[1] in ('S2','gt','PERMANENTWATERJRC') and x['rfilename'].endswith('.tif')]
manifest=dict(dataset_sha=data['sha'],checkpoint_sha=model['sha'],files=[],complete=False)
try:
    checkpoint=next(x for x in model['siblings'] if x['rfilename']=='models/WF2_unetv2_all/model.pt')
    manifest['files'].append(fetch('https://huggingface.co/isp-uv-es/ml4floods',model['sha'],checkpoint,wf_ckpt))
    for i,item in enumerate(selected):
        manifest['files'].append(fetch('https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2',data['sha'],item,wf_root/item['rfilename']))
        print(i+1,len(selected),item['rfilename'],flush=True)
    manifest['complete']=True
finally:
    (results/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
prid='ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11'
prrev='91ce9d38086a80b078a192b374df758b8855b732'
pr=meta('https://huggingface.co/api/models/'+prid+'/revision/'+prrev+'?blobs=true')
pm=dict(model_id=prid,revision=pr['sha'],files=[],complete=False)
try:
    for name in ('config.yaml','requirements.txt','Prithvi-EO-V2-300M-TL-Sen1Floods11.pt'):
        item=next(x for x in pr['siblings'] if x['rfilename']==name)
        pm['files'].append(fetch('https://huggingface.co/'+prid,pr['sha'],item,root/'prithvi_assets'/name))
    pm['complete']=True
finally:
    (results/'prithvi_download_manifest.json').write_text(json.dumps(pm,indent=2))
print('All required downloads verified')
