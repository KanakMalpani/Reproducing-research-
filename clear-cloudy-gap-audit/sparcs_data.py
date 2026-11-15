"""Fixed Landsat calibration, manual-label preflight, and window reader."""
from pathlib import Path
from functools import lru_cache
from collections import Counter
import hashlib,json,re
import numpy as np
import rasterio
from PIL import Image

A=Path(__file__).resolve().parent;R=A/'results'
DATASET='Landsat 8 SPARCS';DOI='10.5066/F7FB5146'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metadata(path):
 text=Path(path).read_text()
 def number(name):
  found=re.findall(r'^\s*'+re.escape(name)+r'\s*=\s*([-+\d.Ee]+)\s*$',text,re.M)
  assert len(found)==1,('STOP missing/ambiguous calibration',name,path)
  return float(found[0])
 sun=number('SUN_ELEVATION');assert 0<sun<=90
 return dict(sun_elevation=sun,mult=[number('REFLECTANCE_MULT_BAND_'+str(i)) for i in range(2,8)],
             add=[number('REFLECTANCE_ADD_BAND_'+str(i)) for i in range(2,8)])

@lru_cache(maxsize=2)
def read_chip(data,mask,mtl):
 cal=metadata(mtl)
 with rasterio.open(data) as ds:
  assert (ds.height,ds.width)==(1000,1000) and ds.count>=7,('STOP unexpected geometry',data,ds.shape,ds.count)
  dn=ds.read([2,3,4,5,6,7]);assert dn.dtype==np.uint16
  geometry=dict(width=ds.width,height=ds.height,bands=ds.count,crs=str(ds.crs),transform=list(ds.transform),descriptions=ds.descriptions)
 with Image.open(mask) as im:labels=np.asarray(im)
 assert labels.shape==(1000,1000) and set(np.unique(labels)).issubset(range(7)),('STOP unexpected labels',mask)
 x=(dn.astype(np.float32)*np.asarray(cal['mult'],np.float32)[:,None,None]+np.asarray(cal['add'],np.float32)[:,None,None])/np.float32(np.sin(np.deg2rad(cal['sun_elevation'])))
 image=np.all(dn>0,axis=0)&np.all(np.isfinite(x),axis=0)
 return x,labels,image,cal,geometry

def tile_data(row):
 x,g,image,_,_=read_chip(row['data'],row['mask'],row['mtl'])
 r,c=row['r'],row['c'];h=min(256,1000-r);w=min(256,1000-c)
 out=np.zeros((6,256,256),np.float32);lab=np.full((256,256),255,np.uint8);valid=np.zeros((256,256),bool)
 out[:,:h,:w]=x[:,r:r+h,c:c+w];lab[:h,:w]=g[r:r+h,c:c+w];valid[:h,:w]=image[r:r+h,c:c+w]
 return out,lab,valid

def prepare():
 manifest=json.loads((R/'sparcs_input_manifest.json').read_text());assert manifest['complete']
 tiles=[];fills={};coverage={};calibrations={};pooled={k:[] for k in ['cloud','land']}
 groups={}
 for chip in manifest['chips']:groups.setdefault(chip['event'],[]).append(chip)
 for event,chips in sorted(groups.items()):
  spectra={k:[] for k in pooled};coverage[event]={str(i):0 for i in range(7)}
  for chip in sorted(chips,key=lambda x:x['chip']):
   x,g,image,cal,geo=read_chip(chip['data'],chip['mask'],chip['mtl'])
   calibrations[chip['chip']]=dict(calibration=cal,geometry=geo)
   for code in range(7):coverage[event][str(code)]+=int((image&(g==code)).sum())
   for kind,code in [('cloud',5),('land',4)]:
    values=x[:,image&(g==code)].copy();spectra[kind].append(values);pooled[kind].append(values)
   for r in (0,256,512,768):
    for c in (0,256,512,768):
     row=dict(chip,dataset=DATASET,scene=event,tile_id=chip['chip']+'/'+str(r)+'/'+str(c),r=r,c=c)
     y,l,v=tile_data(row);cv=v&np.isin(l,[2,4,5,6]);score=v&np.isin(l,[2,4,6]);water=np.isin(l,[2,6])
     den=y[3]+y[2];nd=score&(den>1e-6)
     reasons=[]
     if cv.mean()<.5:reasons.append('less_than_half_image_and_cloud_label_valid')
     if score.mean()<.5:reasons.append('less_than_half_visible_water_label_valid')
     row.update(eligible=not reasons,exclusions=reasons,cloud_frac=float((cv&(l==5)).sum()/cv.sum()) if cv.any() else None,
                water_frac=float((score&water).sum()/score.sum()) if score.any() else None,
                ndvi_clear=float(((y[3]-y[2])/(den+1e-6))[nd].mean()) if nd.any() else None,
                scoreable_pixels=int(score.sum()),gt_water_pixels=int((score&water).sum()),cloud_label_pixels=int(cv.sum()))
     tiles.append(row)
  fills[event]={}
  for kind,arrays in spectra.items():
   values=np.concatenate(arrays,axis=1);n=values.shape[1]
   fills[event][kind]=dict(pixels=n,source='own_acquisition' if n>=1000 else 'pooled_SPARCS',spectrum=np.median(values,axis=1).tolist() if n>=1000 else None)
  print('SPARCS PREFLIGHT',event,'windows',len(tiles),flush=True)
 frozen={}
 for kind,arrays in pooled.items():
  values=np.concatenate(arrays,axis=1);assert values.shape[1]>=1000,'STOP insufficient pooled fill pixels'
  frozen[kind]=np.median(values,axis=1,overwrite_input=True).tolist()
  del values
 for event,classes in fills.items():
  for kind,obj in classes.items():
   if obj['spectrum'] is None:obj['spectrum']=frozen[kind]
 return dict(dataset=DATASET,doi=DOI,rule_sha256=digest(A/'SPARCS_FROZEN_CONTROL_RULE.md'),manifest_sha256=digest(R/'sparcs_input_manifest.json'),
             chip_count=len(manifest['chips']),acquisition_count=len(groups),tile_candidates=len(tiles),eligible_tiles=sum(t['eligible'] for t in tiles),
             exclusion_counts=dict(Counter(e for t in tiles for e in t['exclusions'])),label_coverage_by_acquisition=coverage,
             calibration_by_chip=calibrations,pooled_frozen_spectra=frozen,fills=fills,tiles=tiles,
             score='visible manual water/flooded versus land only; cloud/shadow/snow/padding excluded; not hidden-water accuracy')

if __name__=='__main__':
 out=R/'sparcs_label_preflight.json';assert not out.exists(),'STOP preflight exists'
 obj=prepare()
 with out.open('x') as f:json.dump(obj,f,indent=2)
 print('SPARCS PREFLIGHT COMPLETE',json.dumps({k:obj[k] for k in ['chip_count','acquisition_count','tile_candidates','eligible_tiles','exclusion_counts']}),flush=True)
