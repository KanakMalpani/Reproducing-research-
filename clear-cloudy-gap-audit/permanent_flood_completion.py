"""Complete the supplied JRC stratification with separate code1 flood proxy and code2 seasonal water."""
import os,sys,pathlib,json,hashlib,numpy as np,torch,rasterio
from rasterio.windows import Window
A=pathlib.Path(__file__).resolve().parent;os.chdir(A);sys.path.insert(0,str(A))
from wf_data import scenes,NORM
R=A/'results';ROOT=pathlib.Path(os.environ['WF_ROOT']);BASE=pathlib.Path(os.environ['WF_CKPT'])
rule='Frozen released WF2. Original256px grid stride512 and >=50% valid GT. JRC YearlyHistory codes0=no observations,1=not water,2=seasonal,3=permanent. Separate binary-water IoUs on code3 permanent, code1 historically-nonwater flood proxy, code2 seasonal, code0 unknown; never label code0 flood. All/clear/cloud pixels reported. Pool confusion within scene; unweighted scene means with >=200 GT-water gate and5000 scene-bootstrap draws seed42. Also report secondary pooled IoU and GT-water coverage. JRC is used to build released GT, so permanent-water scores are label-dependent. Nonpermanent water is not independently confirmed flood; whole-image genuine flood split remains unidentifiable without labels for unknown regions.'
assert BASE.exists() and ROOT.exists()
for sp in ('val','test'):
 for name,s2,gt in scenes(sp):assert pathlib.Path(s2).exists() and pathlib.Path(gt).exists() and (ROOT/sp/'PERMANENTWATERJRC'/(name+'.tif')).exists(), 'STOP missing required path'
(A/'PERMANENT_FLOOD_COMPLETION_RULE.md').write_text(rule+chr(10)+'Sources: https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_YearlyHistory and ml4floods/data/create_gt.py, ee_download.py commit192426c7286315b59a2f52c7a9cdebe589242de2.')
ns={};exec((A/'cloud_iou_pilot.py').read_text().split('BINS=')[0],ns);m=ns['m'].cuda().eval()
before=hashlib.sha256(BASE.read_bytes()).hexdigest();torch.set_num_threads(3)
def cm(p,w,v):return np.array([(p&w&v).sum(),(p&~w&v).sum(),(~p&w&v).sum()],np.int64)
def iou(x):return float(x[0]/sum(x)) if sum(x) else None
names={0:'unknown',1:'flood_proxy_historically_nonwater',2:'seasonal',3:'permanent'};rows=[]
for sp in ('val','test'):
 for name,s2,gt in scenes(sp):
  counts={kind:{region:np.zeros(3,np.int64) for region in ('all','clear','cloud')} for kind in names.values()};coverage={kind:dict(valid_pixels=0,gt_water_pixels=0) for kind in names.values()};n=0
  with rasterio.open(s2) as S,rasterio.open(gt) as G,rasterio.open(ROOT/sp/'PERMANENTWATERJRC'/(name+'.tif')) as J:
   assert S.shape==G.shape==J.shape and S.transform==G.transform==J.transform and S.crs==G.crs==J.crs
   for r in range(0,G.height-255,512):
    for c in range(0,G.width-255,512):
     win=Window(c,r,256,256);g=G.read(window=win);v=g[1]>0
     if v.mean()<.5:continue
     raw=S.read(list(range(1,14)),window=win).astype(np.float32);j=J.read(1,window=win);assert np.isin(j[v],[0,1,2,3]).all()
     x=(raw-NORM[:,0,None,None])/NORM[:,1,None,None]
     with torch.inference_mode():p=(m(torch.from_numpy(x)[None].cuda())[0,1]>0).cpu().numpy()
     w=g[1]==2;n+=1
     for code,kind in names.items():
      mask=v&(j==code);coverage[kind]['valid_pixels']+=int(mask.sum());coverage[kind]['gt_water_pixels']+=int((mask&w).sum())
      for region,reg in [('all',v),('clear',g[0]==1),('cloud',g[0]==2)]:counts[kind][region]+=cm(p,w,mask&reg)
  rows.append(dict(split=sp,scene=name,tiles=n,coverage=coverage,confusions={k:{g:a.tolist() for g,a in groups.items()} for k,groups in counts.items()}));print('PERMANENT FLOOD COMPLETION',sp,name,flush=True)
def summarize(rs,kind,region):
 mat=np.array([r['confusions'][kind][region] for r in rs]);vals=[iou(a) for a in mat if a[0]+a[2]>=200];rng=np.random.default_rng(42)
 return dict(pooled_confusion=mat.sum(0).tolist(),pooled_iou_secondary=iou(mat.sum(0)),eligible_scenes=len(vals),scene_mean_iou=float(np.mean(vals)) if vals else None,ci95_scene_bootstrap=np.percentile([np.mean(rng.choice(vals,len(vals))) for _ in range(5000)],[2.5,97.5]).tolist() if vals else None)
summary={sp:{k:{g:summarize([r for r in rows if sp=='val_test' or r['split']==sp],k,g) for g in ('all','clear','cloud')} for k in names.values()} for sp in ('val_test','val','test')}
coverage={k:{field:sum(r['coverage'][k][field] for r in rows) for field in ('valid_pixels','gt_water_pixels')} for k in names.values()}
water=sum(x['gt_water_pixels'] for x in coverage.values());unknown_fraction=coverage['unknown']['gt_water_pixels']/water
assert hashlib.sha256(BASE.read_bytes()).hexdigest()==before
out=dict(rule=rule,rows=rows,summary=summary,coverage=coverage,unknown_gt_water_fraction=unknown_fraction,full_genuine_flood_split='not identifiable from supplied JRC0 pixels; additional independent water-history/flood labels required',released_checkpoint_sha256=before,released_checkpoint_unchanged=True)
(R/'permanent_flood_completion.json').write_text(json.dumps(out,indent=2));print('JRC COMPLETION SUMMARY',json.dumps(dict(coverage=coverage,unknown_gt_water_fraction=unknown_fraction,summary=summary['val_test'])),flush=True)
