import os
os.environ['MPLBACKEND']='Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
import sys,json,pathlib,yaml,numpy as np,torch,rasterio
from rasterio.windows import Window
from terratorch.tasks import SemanticSegmentationTask
from terratorch.datamodules.sen1floods11 import MEANS,STDS
audit=pathlib.Path(__file__).resolve().parent
os.chdir(audit)
sys.path.insert(0,str(audit))
from wf_data import scenes
results=audit/'results'; pdir=audit/'prithvi_assets'
config=yaml.safe_load((pdir/'config.yaml').read_text())
args=config['model']['init_args']
args['model_args']['backbone_pretrained']=False
print('BUILDING PRITHVI',flush=True)
prithvi_task=SemanticSegmentationTask(**args)
weights=torch.load(pdir/'Prithvi-EO-V2-300M-TL-Sen1Floods11.pt',map_location='cpu',weights_only=True)
print('CHECKPOINT KEYS',list(weights)[:8],flush=True)
if 'state_dict' in weights: weights=weights['state_dict']
if all(k.startswith('model.') for k in weights):
    prithvi_task.load_state_dict(weights,strict=True)
else:
    prithvi_task.model.load_state_dict(weights,strict=True)
del weights
prithvi_model=prithvi_task.model.to('cuda').eval()
bands=config['data']['init_args']['bands']
means=np.array([MEANS[b] for b in bands],np.float32)[:,None,None]
stds=np.array([STDS[b] for b in bands],np.float32)[:,None,None]
@torch.inference_mode()
def predict(raw):
    image=(raw[[1,2,3,8,11,12]].astype(np.float32)/10000.-means)/stds
    output=prithvi_model(torch.from_numpy(image)[None].to('cuda'),temporal_coords=None,location_coords=None).output
    assert output.shape==(1,2,256,256), str(output.shape)
    return (output.argmax(1)[0]==1).cpu().numpy()

name,s2,gt=scenes('val')[0]
with rasterio.open(s2) as S:
    raw=S.read(list(range(1,14)),window=Window(0,0,256,256))
pred=predict(raw)
print('PRITHVI SMOKE PASSED',pred.shape,int(pred.sum()),flush=True)
