"""Terrain check (step 1) + method-1 selection on val / single test score (step 2)."""
import json, numpy as np
R=json.load(open("method1_tiles.json")); rng=np.random.default_rng(0)
iou=lambda a: a[0]/max(sum(a),1)
def pool(ts,k="base"): return np.array([t["cm"][k] for t in ts]).sum(0) if ts else np.zeros(3)
CL=lambda t:t["cloud_frac"]<0.01; CD=lambda t:t["cloud_frac"]>=0.1
out={}
# ---- step 1: terrain ----
by={}
for t in R: by.setdefault(t["scene"],[]).append(t)
paired=[s for s,ts in by.items() if (lambda a,b: a[0]+a[2]>=200 and b[0]+b[2]>=200)(pool([t for t in ts if CL(t)]),pool([t for t in ts if CD(t)]))]
terr=[]; mg=[]
for s in paired:
    a=[t for t in by[s] if CL(t)]; b=[t for t in by[s] if CD(t)]
    f=lambda ts,k: float(np.mean([t[k] for t in ts if t[k] is not None])) if any(t[k] is not None for t in ts) else None
    terr.append(dict(scene=s,n_clear=len(a),n_cloudy=len(b),wf_clear=f(a,"water_frac"),wf_cloudy=f(b,"water_frac"),
        ndvi_clear=f(a,"ndvi_clear"),ndvi_cloudy=f(b,"ndvi_clear"),jrc_clear=f(a,"jrc_perm_frac"),jrc_cloudy=f(b,"jrc_perm_frac"),
        gap=100*(iou(pool(a))-iou(pool(b)))))
    # matched: each cloudy tile -> nearest unused clear tile by water_frac, |diff|<=0.05
    used=set(); ma=[]; mb=[]
    for t in sorted(b,key=lambda t:t["water_frac"]):
        cand=[(abs(u["water_frac"]-t["water_frac"]),i) for i,u in enumerate(a) if i not in used]
        if cand and min(cand)[0]<=0.05: i=min(cand)[1]; used.add(i); ma.append(a[i]); mb.append(t)
    pa,pb=pool(ma),pool(mb)
    if pa[0]+pa[2]>=200 and pb[0]+pb[2]>=200: mg.append(dict(scene=s,pairs=len(mb),gap=100*(iou(pa)-iou(pb))))
def boot(v): v=np.array(v); bs=[rng.choice(v,len(v)).mean() for _ in range(5000)]; return [float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
out["terrain"]=dict(per_scene=terr,matched=mg,matched_mean_gap=float(np.mean([x["gap"] for x in mg])) if mg else None,
    matched_ci95=boot([x["gap"] for x in mg]) if len(mg)>1 else None)
# ---- step 2: selection on val ----
cfgs=list(R[0]["cm"]); V=[t for t in R if t["split"]=="val"]; T=[t for t in R if t["split"]=="test"]
def sc(ts,k): return 100*iou(pool([t for t in ts if CL(t)],k)),100*iou(pool([t for t in ts if CD(t)],k))
vb=sc(V,"base"); val={k:sc(V,k) for k in cfgs}
ok=[k for k in cfgs if k!="base" and val[k][0]>=vb[0]-1.0 and val[k][1]>vb[1]]
sel=max(ok,key=lambda k:(round(val[k][1],6),-abs(float(k[1:])))) if ok else None
out["val"]=dict(baseline=vb,configs=val,selected=sel)
if sel:
    tb,ts_=sc(T,"base"),sc(T,sel)
    sn=sorted({t["scene"] for t in T if CD(t)})
    def dcd(scs):
        ts=[t for s in scs for t in T if t["scene"]==s and CD(t)]; return 100*(iou(pool(ts,sel))-iou(pool(ts,"base")))
    bs=[dcd(rng.choice(sn,len(sn))) for _ in range(5000)]
    d_cd=ts_[1]-tb[1]; d_cl=ts_[0]-tb[0]; ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
    out["test"]=dict(baseline=tb,selected=ts_,delta_cloudy=d_cd,ci95_cloudy=ci,delta_clear=d_cl,cloudy_scenes=len(sn),
        verdict="PASS" if d_cd>=5 and ci[0]>0 and d_cl>=-1 else "FAIL")
json.dump(out,open("method1_results.json","w"),indent=1)
for x in terr: print({k:(round(v,3) if isinstance(v,float) else v) for k,v in x.items()})
print("matched",mg,out["terrain"]["matched_mean_gap"],out["terrain"]["matched_ci95"])
print("VAL base",vb); [print(k,[round(v,2) for v in val[k]]) for k in cfgs]; print("SEL",sel); print("TEST",out.get("test"))
