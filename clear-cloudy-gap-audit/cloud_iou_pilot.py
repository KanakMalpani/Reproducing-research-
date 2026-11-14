"""Pilot: how does the published WF2_unetv2_all water IoU degrade with cloud cover? CPU, 3 threads."""
import os, sys, json, numpy as np, torch, torch.nn as nn
torch.set_num_threads(3)
from wf_data import scenes, tiles
def dc(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.ReLU(True),nn.Conv2d(o,o,3,padding=1),nn.ReLU(True))
class UNet(nn.Module):  # identical to ml4floods.models.architectures.unets.UNet
    def __init__(s,n=13,k=2):
        super().__init__(); s.dconv_down1,s.dconv_down2,s.dconv_down3,s.dconv_down4=dc(n,64),dc(64,128),dc(128,256),dc(256,512)
        s.maxpool=nn.MaxPool2d(2); s.upsample=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        s.dconv_up3,s.dconv_up2,s.dconv_up1=dc(768,256),dc(384,128),dc(192,64); s.conv_last=nn.Conv2d(64,k,1)
    def forward(s,x):
        c1=s.dconv_down1(x);c2=s.dconv_down2(s.maxpool(c1));c3=s.dconv_down3(s.maxpool(c2));x=s.dconv_down4(s.maxpool(c3))
        x=s.dconv_up3(torch.cat([s.upsample(x),c3],1));x=s.dconv_up2(torch.cat([s.upsample(x),c2],1))
        return s.conv_last(s.dconv_up1(torch.cat([s.upsample(x),c1],1)))
m=UNet(); sd=torch.load(os.environ.get("WF_CKPT","/workspace/wf-models/model.pt"),map_location="cpu")
m.load_state_dict({k.replace("network.",""):v for k,v in sd.items()}); m.eval()
BINS=[0,0.01,0.1,0.3,0.6,1.01]; rows=[]
for split in sys.argv[1:] or ["test","val"]:
    for name,s2,gt in scenes(split):
        for r,c,x,g in tiles(s2,gt):
            with torch.no_grad(): p=torch.sigmoid(m(torch.from_numpy(x)[None]))[0].numpy()
            pw=p[1]>0.5; valid=g[1]>0; wt=g[1]==2; cl=g[0]==2
            cf=float(cl[valid].mean())
            def cm(mask): return [int((pw&wt&mask).sum()),int((pw&~wt&mask).sum()),int((~pw&wt&mask).sum())]
            rows.append(dict(split=split,scene=name,r=r,c=c,cloud_frac=cf,all=cm(valid),clear=cm(valid&~cl),under_cloud=cm(valid&cl)))
        print(split,name,len(rows),flush=True)
P=os.environ.get("OUT","cloud_iou"); json.dump(rows,open(P+"_tiles.json","w"))
def iou(rs,k):
    t=np.array([x[k] for x in rs]).sum(0) if rs else np.zeros(3); return float(t[0]/max(t.sum(),1))
rng=np.random.default_rng(0); out=[]
for lo,hi in zip(BINS[:-1],BINS[1:]):
    rs=[x for x in rows if lo<=x["cloud_frac"]<hi]; sc=sorted({x["scene"] for x in rs})
    boot=[]
    for _ in range(500):
        pick=rng.choice(sc,len(sc)) if sc else []
        boot.append(iou([x for s in pick for x in rs if x["scene"]==s],"all"))
    out.append(dict(bin=f"[{lo},{min(hi,1)})",tiles=len(rs),scenes=len(sc),iou_all=iou(rs,"all"),
        ci95=[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))] if sc else None,
        iou_clear_px=iou(rs,"clear"),iou_under_cloud_px=iou(rs,"under_cloud")))
from scipy.stats import spearmanr
per=[(x["cloud_frac"],x["all"][0]/max(sum(x["all"]),1)) for x in rows if x["all"][0]+x["all"][2]>100]
rho=spearmanr(*zip(*per)) if len(per)>3 else None
res=dict(ckpt="isp-uv-es/ml4floods WF2_unetv2_all",tiles=len(rows),bins=out,spearman_cloudfrac_vs_tile_iou=[float(rho[0]),float(rho[1])] if rho else None)
json.dump(res,open(P+"_results.json","w"),indent=1); print(json.dumps(res,indent=1))
