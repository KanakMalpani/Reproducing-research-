"""Method 2: train cloud-aware residual adapter on cached train tiles; select on val; score test once (SELECTION_RULE_method2.md)."""
import json, os, time, numpy as np, torch, torch.nn as nn
torch.set_num_threads(3); torch.manual_seed(0); np.random.seed(0)
C="/workspace/wf-cache"; T=256*256
def load(s):
    meta=json.load(open(f"{C}/{s}_meta.json")); n=len(meta)
    return meta, np.memmap(f"{C}/{s}_x.bin",np.float16,"r",shape=(n,3,256,256)), np.memmap(f"{C}/{s}_y.bin",np.int8,"r",shape=(n,2,256,256))
OUT="m2_results.json"; res=json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res,open(OUT,"w"),indent=1)
class A(nn.Module):
    def __init__(s,h,cp):
        super().__init__(); s.cp=cp; s.n=nn.Sequential(nn.Conv2d(3 if cp else 2,h,3,padding=1),nn.ReLU(),nn.Conv2d(h,h,3,padding=1),nn.ReLU(),nn.Conv2d(h,1,1))
        nn.init.zeros_(s.n[-1].weight); nn.init.zeros_(s.n[-1].bias)
    def forward(s,x): return x[:,0]+s.n(x if s.cp else x[:,:2])[:,0]
def cms(model,X,Y,meta,pred_fn=None):
    out=[]
    for i in range(len(meta)):
        x=torch.from_numpy(np.asarray(X[i],np.float32))[None]; y=np.asarray(Y[i])
        with torch.no_grad(): lw=(model(x)[0] if model else x[0,0]).numpy()
        p=pred_fn(x[0].numpy()) if pred_fn else lw>0
        v=y[1]>0; w=y[1]==2
        out.append([int((p&w&v).sum()),int((p&~w&v).sum()),int((~p&w&v).sum())])
    return out
iou=lambda a:a[0]/max(sum(a),1)
def pool(cm,meta,f): a=np.array([c for c,t in zip(cm,meta) if f(t)]); return a.sum(0) if len(a) else np.zeros(3)
CL=lambda t:t["cloud_frac"]<0.01; CD=lambda t:t["cloud_frac"]>=0.1
def sc(cm,meta): return [100*iou(pool(cm,meta,CL)),100*iou(pool(cm,meta,CD))]
def matched(cm,meta):
    by={}
    for c,t in zip(cm,meta): by.setdefault(t["scene"],[]).append((t,c))
    gaps=[]; cd_all=np.zeros(3)
    for s,L in by.items():
        a=[x for x in L if CL(x[0])]; b=[x for x in L if CD(x[0])]; used=set(); ma=[]; mb=[]
        for t in sorted(b,key=lambda x:x[0]["water_frac"]):
            cand=[(abs(u[0]["water_frac"]-t[0]["water_frac"]),i) for i,u in enumerate(a) if i not in used]
            if cand and min(cand)[0]<=0.05: i=min(cand)[1]; used.add(i); ma.append(a[i][1]); mb.append(t[1])
        if not mb: continue
        pa,pb=np.sum(ma,0),np.sum(mb,0)
        if pa[0]+pa[2]>=200 and pb[0]+pb[2]>=200: gaps.append(100*(iou(pa)-iou(pb))); cd_all=cd_all+pb
    rng=np.random.default_rng(0); bs=[rng.choice(gaps,len(gaps)).mean() for _ in range(5000)] if len(gaps)>1 else [np.nan]
    return dict(scenes=len(gaps),gap=float(np.mean(gaps)) if gaps else None,ci95=[float(np.nanpercentile(bs,2.5)),float(np.nanpercentile(bs,97.5))],matched_cloudy_iou=100*iou(cd_all))
mv,Xv,Yv=load("val"); mt,Xt,Yt=load("test"); mtr,Xtr,Ytr=load("train")
if "val_base" not in res: res["val_base"]=sc(cms(None,Xv,Yv,mv),mv); res["n_train_tiles"]=len(mtr); save()
# exploratory threshold widening, val only
if "explore_T_val" not in res:
    res["explore_T_val"]={str(a):sc(cms(None,Xv,Yv,mv,lambda x,a=a:1/(1+np.exp(-x[0]))>np.clip(0.5+a*x[2],0.05,0.95)),mv) for a in (0.5,0.6,0.8,1.0,1.5)}; save()
bce=nn.BCEWithLogitsLoss(reduction="none")
for h in (8,16):
    for cp in (True,False):
        k=f"h{h}_cp{int(cp)}"
        if k in res.get("val",{}): continue
        torch.manual_seed(0); net=A(h,cp); opt=torch.optim.Adam(net.parameters(),1e-3); t0=time.time()
        for ep in range(4):
            idx=np.random.default_rng(ep).permutation(len(mtr)); L=0
            for b in range(0,len(idx),8):
                j=np.sort(idx[b:b+8]); x=torch.from_numpy(np.asarray(Xtr[j],np.float32)); y=torch.from_numpy(np.asarray(Ytr[j,1],np.int64))
                v=(y>0).float(); l=(bce(net(x),(y==2).float())*v).sum()/v.sum().clamp(min=1)
                opt.zero_grad(); l.backward(); opt.step(); L+=l.item()
            print(k,"ep",ep,round(L,2),round(time.time()-t0),"s",flush=True)
        torch.save(net.state_dict(),f"{C}/adapter_{k}.pt"); net.eval()
        res.setdefault("val",{})[k]=sc(cms(net,Xv,Yv,mv),mv); save(); print(k,res["val"][k],flush=True)
vb=res["val_base"]; ok=[k for k,v in res["val"].items() if v[0]>=vb[0]-1 and v[1]>vb[1]]
sel=max(ok,key=lambda k:res["val"][k][1]) if ok else None; res["selected"]=sel
if sel and "test" not in res:
    h=int(sel[1:sel.index("_")]); cp=sel.endswith("1"); net=A(h,cp); net.load_state_dict(torch.load(f"{C}/adapter_{sel}.pt")); net.eval()
    cb,cs=cms(None,Xt,Yt,mt),cms(net,Xt,Yt,mt); b,s=sc(cb,mt),sc(cs,mt)
    sn=sorted({t["scene"] for t in mt if CD(t)}); rng=np.random.default_rng(0)
    def d(scs):
        A_=np.zeros(3);B_=np.zeros(3)
        for s_ in scs:
            for c1,c2,t in zip(cb,cs,mt):
                if t["scene"]==s_ and CD(t): A_+=c1;B_+=c2
        return 100*(iou(B_)-iou(A_))
    bs=[d(rng.choice(sn,len(sn))) for _ in range(2000)]; ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
    res["test"]=dict(baseline=b,selected=s,delta_cloudy=s[1]-b[1],ci95_cloudy=ci,delta_clear=s[0]-b[0],
        verdict="PASS" if s[1]-b[1]>=5 and ci[0]>0 and s[0]-b[0]>=-1 else "FAIL",
        matched_baseline=matched(cb,mt),matched_selected=matched(cs,mt))
elif not sel: res["test"]="not opened: no config beat baseline on val"
save(); print(json.dumps({k:v for k,v in res.items()},indent=1,default=str))
