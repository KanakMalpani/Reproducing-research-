"""Frozen SPARCS evaluation; experimental computation runs only in Colab."""
import os
os.environ['MPLBACKEND']='Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import torch,yaml
from terratorch.tasks import SemanticSegmentationTask
from terratorch.datamodules.sen1floods11 import MEANS,STDS
from sparcs_data import A,R,tile_data,digest,DATASET,DOI

OLD=Path('/content/c2sms_run/clear-cloudy-gap-audit')
assert digest(OLD/'c2sms_controls.py')=='097905c19e0c246e4f5dcabb08f265dc64972102878a43de6e7f7c0c6ee8ad8d'
sys.path.insert(0,str(OLD))
import c2sms_controls as ctl
ctl.DATASET=DATASET;ctl.DOI=DOI
OUTPUTS=['sparcs_prithvi_composition.json','sparcs_prithvi_same_tile.json','sparcs_prithvi_aggregator.json','sparcs_prithvi_tiles.json']
assert not any((R/n).exists() for n in OUTPUTS),'STOP completed result exists'
pre=json.loads((R/'sparcs_label_preflight.json').read_text())
assert pre['rule_sha256']==digest(A/'SPARCS_FROZEN_CONTROL_RULE.md')
assert pre['manifest_sha256']==digest(R/'sparcs_input_manifest.json')
checkpoint=OLD/'prithvi_assets/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt'
before=digest(checkpoint)
assert before=='76eed77d8bd543ae441308b80e8408502243a871cb8a338ddc8220ed96dfc270'
config=OLD/'prithvi_assets/config.yaml'
assert digest(config)=='ca2fa45885b85afac9400c57f7ca3b711526cbad3af285f8b67de74ba2db404a'
cfg=yaml.safe_load(config.read_text());args=cfg['model']['init_args'];args['model_args']['backbone_pretrained']=False
torch.manual_seed(0);torch.set_num_threads(3)
task=SemanticSegmentationTask(**args)
weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
if 'state_dict' in weights:weights=weights['state_dict']
if all(k.startswith('model.') for k in weights):task.load_state_dict(weights,strict=True)
else:task.model.load_state_dict(weights,strict=True)
del weights
model=task.model.cuda().eval();model.requires_grad_(False)
bands=cfg['data']['init_args']['bands']
means=np.asarray([MEANS[b] for b in bands],np.float32)[:,None,None]
stds=np.asarray([STDS[b] for b in bands],np.float32)[:,None,None]

@torch.inference_mode()
def predict(raws):
 x=np.stack([(raw-means)/stds for raw in raws])
 output=model(torch.from_numpy(x).cuda(),temporal_coords=None,location_coords=None).output
 assert output.shape==(len(raws),2,256,256) and torch.isfinite(output).all(),'STOP invalid logits'
 return output.argmax(1).cpu().numpy()==1

start=time.time();tiles=[t for t in pre['tiles'] if t['eligible']];events=sorted(pre['fills']);natural=[];original={}
for offset in range(0,len(tiles),2):
 batch=tiles[offset:offset+2];inputs=[tile_data(t) for t in batch];predictions=predict([x[0] for x in inputs])
 for row,(x,g,image),p in zip(batch,inputs,predictions):
  valid=image&np.isin(g,[2,4,6]);water=np.isin(g,[2,6])
  natural.append(dict(row,all=ctl.cm(p,water,valid).tolist()))
  original[row['tile_id']]=np.packbits(p)
 if (offset+len(batch))%100==0 or offset+len(batch)==len(tiles):print('SPARCS NATURAL',offset+len(batch),'/',len(tiles),'seconds',round(time.time()-start,1),flush=True)
comp=ctl.composition(natural,events)
comp.update(scene_definition='original acquisition; within-acquisition matching',cloud_groups='manual cloud5 versus clear2/4/6; shadow0/1 and snow3 excluded; fraction<.01 clear, >=.1 cloudy',water_scoring='visible manual2/6 water versus4 land only; no hidden-water labels')
donors=sorted([t for t in tiles if .1<=t['cloud_frac']<=.6],key=lambda t:t['tile_id']);same=[]
for event in events:
 clear=[t for t in tiles if t['event']==event and t['cloud_frac']<.01];own=[t for t in donors if t['event']==event]
 acc={k:np.zeros(3,np.int64) for k in ['original','cloud','land']};pixels=water_pixels=0;assignments=[]
 for i,row in enumerate(clear):
  if not donors:break
  donor=own[i%len(own)] if own else donors[(i+42)%len(donors)]
  _,dg,_=tile_data(donor);mask=dg==5
  x,g,image=tile_data(row);valid=image&np.isin(g,[2,4,6])&~mask;water=np.isin(g,[2,6])
  pixels+=int(valid.sum());water_pixels+=int((valid&water).sum())
  p=np.unpackbits(original[row['tile_id']]).reshape(256,256).astype(bool);acc['original']+=ctl.cm(p,water,valid)
  painted=[]
  for kind in ['cloud','land']:
   y=x.copy();y[:,mask]=np.asarray(pre['fills'][event][kind]['spectrum'],np.float32)[:,None];painted.append(y)
  for kind,p in zip(['cloud','land'],predict(painted)):acc[kind]+=ctl.cm(p,water,valid)
  assignments.append(dict(target=row['tile_id'],donor=donor['tile_id'],own_acquisition=bool(own),painted_pixels=int(mask.sum()),common_scoreable_pixels=int(valid.sum())))
 scores={k:ctl.iou(v) for k,v in acc.items()};reasons=[]
 if not clear:reasons.append('no_clear_tiles')
 if not donors:reasons.append('no_eligible_SPARCS_cloud_donor_masks')
 if water_pixels<200:reasons.append('common_unpainted_surface_less_than_200_GT_water_pixels')
 if any(v is None for v in scores.values()):reasons.append('undefined_common_surface_IoU')
 row=dict(dataset=DATASET,scene=event,event=event,clear_tiles=len(clear),eligible=not reasons,exclusions=reasons,common_scoreable_pixels=pixels,common_GT_water_pixels=water_pixels,
          confusions={k:v.tolist() for k,v in acc.items()},iou=scores,fills=pre['fills'][event],assignments=assignments,
          gap_cloud_pts=100*(scores['original']-scores['cloud']) if not reasons else None,gap_land_pts=100*(scores['original']-scores['land']) if not reasons else None)
 row['cloud_minus_land_pts']=row['gap_cloud_pts']-row['gap_land_pts'] if not reasons else None;same.append(row)
 print('SPARCS FILL',event,'eligible',row['eligible'],'seconds',round(time.time()-start,1),flush=True)
agg=ctl.aggregate(natural,events,comp,same)
agg['acquisitions']=agg.pop('event_folders');agg['primary_aggregator']='unweighted mean of eligible original-acquisition scene values'
agg['scene_definition']='original Landsat acquisition, not subscene/window';agg['bootstrap_unit']='original acquisition; all dependent windows retained'
agg['event_weighted_mean_reason']='not estimated; no flood-event identifiers are supplied'
agg.update(eligible_windows=len(tiles),runtime_seconds=time.time()-start,checkpoint_unchanged=digest(checkpoint)==before)
assert agg['checkpoint_unchanged'],'STOP checkpoint changed'
metadata=dict(model_id='ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11',revision='91ce9d38086a80b078a192b374df758b8855b732',checkpoint_sha256=before,frozen=True,training=False,
              precision='float32',batch_size=2,model_normalization='unchanged official Sen1Floods11 means/stds',landsat_bands=[2,3,4,5,6,7],input='calibrated TOA reflectance; native30m; no harmonization/resampling',prediction='argmax; no metadata/TTA')
comp['model']=metadata
same_result=dict(dataset=DATASET,doi=DOI,model=metadata,caption='Constructed constant fill on clear SPARCS windows, scored on unpainted visible labelled pixels; not natural cloud.',rows=same,summary=agg['same_tile'])
def clarify(obj):
 if isinstance(obj,dict):return {k:clarify(v) for k,v in obj.items()}
 if isinstance(obj,list):return [clarify(v) for v in obj]
 if isinstance(obj,str):return obj.replace('event-folder','acquisition').replace('event folder','acquisition')
 return obj
outputs=[comp,same_result,agg,dict(dataset=DATASET,doi=DOI,model=metadata,rows=natural)]
for name,obj in zip(OUTPUTS,outputs):
 with (R/name).open('x') as f:json.dump(clarify(obj),f,indent=2)
np.savez_compressed(R/'sparcs_original_predictions.npz',tile_ids=np.asarray(list(original)),packed_predictions=np.stack(list(original.values())))
print('SPARCS THREE CONTROLS COMPLETE',json.dumps(clarify(dict(composition=agg['composition'],same_tile=agg['same_tile'],accuracy=agg['water_accuracy']['summary']))),flush=True)
