"""Fine-tune a separate WF2 candidate; never modify the released checkpoint."""
import os,sys,json,pathlib,time,math,random,hashlib,collections
os.environ.setdefault('GDAL_CACHEMAX','64')
import numpy as np,torch,torch.nn.functional as F,rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
AUDIT=pathlib.Path(__file__).resolve().parent
os.chdir(AUDIT); sys.path.insert(0,str(AUDIT))
from wf_data import NORM
RUN=AUDIT/'results'/'finetune_v1'; CKPTS=AUDIT/'checkpoints'/'finetune_v1'
CKPTS.mkdir(parents=True,exist_ok=True)
ROOT=pathlib.Path(os.environ['WF_ROOT']); BASE=pathlib.Path(os.environ['WF_CKPT'])
assert json.loads((RUN/'training_download_manifest.json').read_text())['complete']
assert not (RUN/'selection.json').exists(), 'Completed run exists; do not overwrite'
SEED=42; EPOCHS=16; STEPS=512; BATCH=8
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(3)
config=dict(seed=SEED,epochs=EPOCHS,steps_per_epoch=STEPS,batch_size=BATCH,encoder_lr=5e-6,decoder_lr=5e-5,weight_decay=1e-4,water_bce_weight=.5,water_dice_weight=.5,cloud_bce_weight=.1,patience=4,min_delta=.0001,water_threshold=.5,selection='maximum eligible validation pooled water IoU; clear and cloudy each >=baseline minus1 point',target_test_iou=.9,train_only=True,training_stride=256,evaluation_stride=512,precision='AMP fp16 training; fp32 validation',augmentation='random right-angle rotations/flips; gain0.9..1.1 and per-band Gaussian offset std20DN',sampler='inverse scene tile-count times (1+2*cloudy+mixed-water); replacement',training_events='exclude evaluation EMSR activations')
(RUN/'config.json').write_text(json.dumps(config,indent=2))
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(8388608),b''): h.update(c)
    return h.hexdigest()
base_digest=digest(BASE)
manifest=json.loads((RUN/'rule.json').read_text())['data_preflight']
index=[]
for scene in manifest['eligible_names']:
    s2=ROOT/'train'/'S2'/(scene+'.tif'); gt=ROOT/'train'/'gt'/(scene+'.tif')
    assert s2.exists() and gt.exists(), 'STOP missing training path'
    with rasterio.open(s2) as S,rasterio.open(gt) as G:
        assert S.count>=13 and G.count==2 and S.shape==G.shape and S.transform==G.transform
        for r in range(0,G.height-255,256):
            for c in range(0,G.width-255,256):
                g=G.read(window=Window(c,r,256,256)); v=g[1]>0
                if v.mean()<.5: continue
                index.append(dict(split='train',scene=scene,r=r,c=c,cloud_frac=float((g[0]==2)[v].mean()),water_frac=float((g[1]==2)[v].mean())))
    if len(index)%500<20: print('TRAIN INDEX',scene,len(index),flush=True)
assert index, 'No eligible training tiles'
(RUN/'train_index.json').write_text(json.dumps(index))
val_index=[r for r in json.loads((AUDIT/'method1_tiles.json').read_text()) if r['split']=='val']
assert len(val_index)==615
assert not ({r['scene'] for r in index}&{r['scene'] for r in val_index})
print('DATA READY',len(index),'train tiles',len(val_index),'validation tiles',flush=True)
class Tiles(Dataset):
    def __init__(self,meta,augment=False): self.meta=meta; self.augment=augment; self.cache=collections.OrderedDict()
    def __len__(self): return len(self.meta)
    def handles(self,row):
        key=(row['split'],row['scene'])
        if key not in self.cache:
            self.cache[key]=(rasterio.open(ROOT/key[0]/'S2'/(key[1]+'.tif')),rasterio.open(ROOT/key[0]/'gt'/(key[1]+'.tif')))
            if len(self.cache)>8:
                _,old=self.cache.popitem(last=False)
                for f in old: f.close()
        self.cache.move_to_end(key); return self.cache[key]
    def __getitem__(self,i):
        row=self.meta[i]; S,G=self.handles(row); win=Window(row['c'],row['r'],256,256)
        raw=S.read(list(range(1,14)),window=win).astype(np.float32); g=G.read(window=win).astype(np.int64)
        if self.augment:
            k=np.random.randint(4); raw=np.rot90(raw,k,axes=(1,2)); g=np.rot90(g,k,axes=(1,2))
            if np.random.rand()<.5: raw=raw[:,:,::-1]; g=g[:,:,::-1]
            if np.random.rand()<.5: raw=raw[:,::-1,:]; g=g[:,::-1,:]
            gain=np.random.uniform(.9,1.1); offset=np.random.normal(0,20,(13,1,1)).astype(np.float32)
            raw=np.maximum(raw*gain+offset,0)
        x=np.ascontiguousarray((raw-NORM[:,0,None,None])/NORM[:,1,None,None])
        return torch.from_numpy(x),torch.from_numpy(np.ascontiguousarray(g)),i
    def close(self):
        for pair in self.cache.values():
            for f in pair: f.close()
        self.cache.clear()
def worker_seed(worker_id):
    seed=torch.initial_seed()%2**32; np.random.seed(seed); random.seed(seed)
def confusion(p,w,v): return np.array([(p&w&v).sum(),(p&~w&v).sum(),(~p&w&v).sum()],np.int64)
def pooled(rows,which):
    selected=[r['all'] for r in rows if which=='all' or (which=='clear' and r['cloud_frac']<.01) or (which=='cloudy' and r['cloud_frac']>=.1)]
    a=np.array(selected).sum(0) if selected else np.zeros(3)
    return float(a[0]/a.sum()) if a.sum() else None
val_dataset=Tiles(val_index)
val_loader=DataLoader(val_dataset,batch_size=4,shuffle=False,num_workers=2,pin_memory=True,worker_init_fn=worker_seed)
ns={}; exec((AUDIT/'cloud_iou_pilot.py').read_text().split('BINS=')[0],ns)
model=ns['m'].to('cuda'); initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
@torch.inference_mode()
def evaluate():
    model.eval(); rows=[]
    for x,g,ids in val_loader:
        pred=(model(x.to('cuda',non_blocking=True))[:,1]>0).cpu().numpy(); gs=g.numpy()
        for j,idx in enumerate(ids.tolist()):
            t=val_index[idx]; v=gs[j,1]>0; w=gs[j,1]==2; cloud=gs[j,0]==2
            rows.append(dict(split='val',scene=t['scene'],r=t['r'],c=t['c'],cloud_frac=t['cloud_frac'],all=confusion(pred[j],w,v).tolist(),clear=confusion(pred[j],w,v&~cloud).tolist(),under_cloud=confusion(pred[j],w,v&cloud).tolist()))
    return {k:pooled(rows,k) for k in ('all','clear','cloudy')},rows
baseline,baseline_rows=evaluate()
(RUN/'baseline_validation.json').write_text(json.dumps(dict(metrics=baseline,rows=baseline_rows),indent=2))
print('BASELINE VALIDATION',json.dumps(baseline),flush=True)
encoder=[]; decoder=[]
for name,p in model.named_parameters(): (encoder if name.startswith('dconv_down') else decoder).append(p)
opt=torch.optim.AdamW([dict(params=encoder,lr=config['encoder_lr']),dict(params=decoder,lr=config['decoder_lr'])],weight_decay=config['weight_decay'])
def lr_scale(step):
    if step<128: return max(.1,(step+1)/128)
    progress=(step-128)/max(EPOCHS*STEPS-128,1)
    return .1+.9*.5*(1+math.cos(math.pi*progress))
schedule=torch.optim.lr_scheduler.LambdaLR(opt,lr_scale)
scaler=torch.amp.GradScaler('cuda',init_scale=1024)
scene_counts=collections.Counter(r['scene'] for r in index)
weights=torch.tensor([(1+2*(r['cloud_frac']>=.1)+(0.01<r['water_frac']<.8))/scene_counts[r['scene']] for r in index],dtype=torch.double)
train_dataset=Tiles(index,augment=True)
best_eligible=baseline['all']; best_any=-1.; selected=None; best_any_epoch=None; stale=0; history=[]
def save_weights(path): torch.save({'network.'+k:v.detach().cpu() for k,v in model.state_dict().items()},path)
start=time.time()
for epoch in range(EPOCHS):
    generator=torch.Generator().manual_seed(SEED+epoch)
    sampler=WeightedRandomSampler(weights,num_samples=STEPS*BATCH,replacement=True,generator=generator)
    loader=DataLoader(train_dataset,batch_size=BATCH,sampler=sampler,num_workers=2,pin_memory=True,drop_last=True,worker_init_fn=worker_seed,generator=generator)
    model.train(); losses=[]; epoch_start=time.time()
    for step,(x,g,_) in enumerate(loader):
        x=x.to('cuda',non_blocking=True); g=g.to('cuda',non_blocking=True)
        valid=(g[:,1]>0).float(); target=(g[:,1]==2).float(); cv=(g[:,0]>0).float(); ct=(g[:,0]==2).float()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16): logits=model(x)
        water=logits[:,1].float(); cloud_logits=logits[:,0].float()
        bce=(F.binary_cross_entropy_with_logits(water,target,reduction='none')*valid).sum()/valid.sum().clamp_min(1)
        probs=water.sigmoid()*valid
        dice=1-(2*(probs*target).sum()+1)/(probs.sum()+(target*valid).sum()+1)
        cloud_loss=(F.binary_cross_entropy_with_logits(cloud_logits,ct,reduction='none')*cv).sum()/cv.sum().clamp_min(1)
        loss=.5*bce+.5*dice+.1*cloud_loss
        assert torch.isfinite(loss), 'Non-finite training loss'
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        scaler.step(opt); scaler.update(); schedule.step(); losses.append(float(loss.detach()))
        if (step+1)%64==0: print('TRAIN STEP',epoch+1,step+1,'loss',float(np.mean(losses)),'seconds',round(time.time()-epoch_start,1),flush=True)
    metrics,rows=evaluate()
    eligible=metrics['clear']>=baseline['clear']-.01 and metrics['cloudy']>=baseline['cloudy']-.01
    change=max(float((model.state_dict()[k].detach().cpu()-v).abs().max()) for k,v in initial.items())
    record=dict(epoch=epoch+1,training_loss=float(np.mean(losses)),validation=metrics,eligible=eligible,max_weight_change=change,seconds=time.time()-epoch_start)
    history.append(record)
    assert change>0, 'Weights did not update'
    if metrics['all']>best_any:
        best_any=metrics['all']; best_any_epoch=epoch+1
        save_weights(CKPTS/'best_validation_model.pt')
    if eligible and metrics['all']>best_eligible+config['min_delta']:
        best_eligible=metrics['all']; selected=dict(epoch=epoch+1,path=str(CKPTS/'best_eligible_model.pt'),validation=metrics)
        save_weights(CKPTS/'best_eligible_model.pt'); stale=0
        (RUN/'selected_validation_tiles.json').write_text(json.dumps(rows))
    else: stale+=1
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),scaler=scaler.state_dict(),epoch=epoch+1,history=history,config=config),CKPTS/'last_training_state.pt')
    (RUN/'history.json').write_text(json.dumps(dict(config=config,baseline=baseline,epochs=history),indent=2))
    print('EPOCH RESULT',json.dumps(record),'BEST ELIGIBLE',best_eligible,flush=True)
    if stale>=config['patience']: print('EARLY STOP',flush=True); break
assert digest(BASE)==base_digest, 'Released checkpoint changed'
selection=dict(run='finetune_v1',selected=selected,baseline_validation=baseline,best_any_validation_iou=best_any,best_any_epoch=best_any_epoch,best_any_weights=str(CKPTS/'best_validation_model.pt'),test_opened=False,epochs_completed=len(history),released_checkpoint_sha256=base_digest,released_checkpoint_unchanged=True,training_seconds=time.time()-start,target=.90)
(RUN/'selection.json').write_text(json.dumps(selection,indent=2))
print('TRAINING COMPLETE',json.dumps(selection),flush=True)
