"""Replicate the keystroke classifier: compact CoAtNet-style model, 80/10/10 split.
Epochs are chosen on the validation split only. The test split is scored once per run, after training stops.
Credit: adapted from Eshwar Potturi's replication, https://github.com/eshwarpotturi/keystroke-replication (scripts/train.py).
Changes: nc and keys read from the npz, the best model is saved to models/<name>.pt, and a per-key report goes to logs/train_report.json.
Architecture, augmentation, learning rate and split are unchanged.
usage: python train.py <train_set> <test_set> <seed> [epochs] [split]"""
import sys,json,math,random,numpy as np,torch,torch.nn as nn
import os; torch.set_num_threads(int(os.environ.get('THREADS','2')))
from common import log,mel_fn,feats,CoAtLite
tr_name,te_name,seed=sys.argv[1],sys.argv[2],int(sys.argv[3]); EPOCHS=int(sys.argv[4]) if len(sys.argv)>4 else 120
SPLIT=sys.argv[5] if len(sys.argv)>5 else "random"
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def load(name):
    d=np.load(f"data/{name}.npz"); X=torch.tensor(d["X"],dtype=torch.float32); return X,torch.tensor(d["y"]),d["pos"],int(d["sr"]),[str(k) for k in d["keys"]]
def split(n_pos,pos,seed,mode):
    idx=np.arange(len(pos)); rng=np.random.RandomState(seed)
    if mode=="random":
        rng.shuffle(idx); a,b=int(.8*len(idx)),int(.9*len(idx)); return idx[:a],idx[a:b],idx[b:]
    # "ordered": first 20 presses of each key train, next 2 val, last 3 test (never mixes early and late presses)
    return idx[pos<20],idx[(pos>=20)&(pos<22)],idx[pos>=22]
Xtr_all,ytr_all,ptr,sr_tr,keys=load(tr_name); Xte_all,yte_all,pte,sr_te,_=load(te_name)
nc=len(keys)
mtr,mte=mel_fn(sr_tr),mel_fn(sr_te)
if tr_name==te_name:
    a,b,c=split(len(ptr),ptr,seed,SPLIT); Xtr,ytr,Xva,yva,Xte,yte=Xtr_all[a],ytr_all[a],Xtr_all[b],ytr_all[b],Xtr_all[c],ytr_all[c]
else:  # transfer: train/val from the training medium, test = every stroke of the other medium
    a,b,_=split(len(ptr),ptr,seed,SPLIT); Xtr,ytr,Xva,yva=Xtr_all[a],ytr_all[a],Xtr_all[b],ytr_all[b]; Xte,yte=Xte_all,yte_all; c=None
log(f"train {tr_name} test {te_name} split {SPLIT} seed {seed}: train {len(ytr)} val {len(yva)} test {len(yte)} classes {nc}")
def aug(x,mel):
    sh=torch.randint(0,int(x.shape[1]*0.3)+1,(x.shape[0],)); x=torch.stack([torch.roll(r,int(s)) for r,s in zip(x,sh)])
    M=mel(x).unsqueeze(1)[...,:48]
    for _ in range(2):
        f=np.random.randint(0,7); f0=np.random.randint(0,64-f); M[:,:,f0:f0+f]=M.mean()
        t=np.random.randint(0,7); t0=np.random.randint(0,M.shape[3]-t); M[:,:,:,t0:t0+t]=M.mean()
    return M
mu,sd=None,None
Mva,Mte=feats(Xva,mtr),feats(Xte,mte)
with torch.no_grad(): mu,sd=feats(Xtr,mtr).mean(),feats(Xtr,mtr).std()
norm=lambda M:(M-mu)/sd
model=CoAtLite(nc=nc); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=0.05)
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=2e-3,total_steps=EPOCHS*math.ceil(len(ytr)/32))
best,best_state,hist=-1,None,[]
def evaluate(M,y):
    model.eval();
    with torch.no_grad(): return (model(norm(M)).argmax(1)==y).float().mean().item()
for ep in range(1,EPOCHS+1):
    model.train(); perm=torch.randperm(len(ytr)); tl=0
    for i in range(0,len(perm),32):
        j=perm[i:i+32]; loss=nn.functional.cross_entropy(model(norm(aug(Xtr[j],mtr))),ytr[j],label_smoothing=0.1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl+=loss.item()*len(j)
    va=evaluate(Mva,yva); hist.append((ep,round(tl/len(ytr),4),round(va,4)))
    if va>best: best,best_ep,best_state=va,ep,{k:v.clone() for k,v in model.state_dict().items()}
    if ep%10==0 or ep==1: log(f"epoch {ep}/{EPOCHS} loss {tl/len(ytr):.3f} val {va:.3f} best_val {best:.3f}@{best_ep}")
model.load_state_dict(best_state); test=evaluate(Mte,yte)
torch.save({"state":best_state,"mu":mu,"sd":sd,"keys":keys,"sr":sr_tr},f"models/{tr_name}.pt")
model.eval()
with torch.no_grad(): P=torch.softmax(model(norm(Mte)),1)
pred=P.argmax(1).numpy(); yt=yte.numpy(); conf=P.max(1).values.numpy(); top3=P.topk(3,1).indices.numpy()
np.savez_compressed(f"data/pred_{tr_name}_{te_name}_{SPLIT}_{seed}.npz",y=yt,pred=pred,conf=conf,top3=top3)   # test scored ONCE, with the epoch chosen on validation
log(f"RESULT train={tr_name} test={te_name} split={SPLIT} seed={seed} best_epoch={best_ep} val={best:.4f} test={test:.4f}")
# per-key accuracy and most-confused pairs (unordered: a<->b counts both directions)
per_key={k:{"right":int(((yt==i)&(pred==i)).sum()),"n":int((yt==i).sum())} for i,k in enumerate(keys)}
pairs={}
for t,p in zip(yt,pred):
    if t!=p: kk=tuple(sorted((keys[t],keys[p]))); pairs[kk]=pairs.get(kk,0)+1
conf_pairs=[{"pair":list(k),"count":v} for k,v in sorted(pairs.items(),key=lambda kv:-kv[1])][:5]
log("per-key accuracy:"); [log(f"  {k}: {v['right']}/{v['n']}") for k,v in per_key.items()]
log("most confused pairs:",[(f"{c['pair'][0]}<->{c['pair'][1]}",c["count"]) for c in conf_pairs])
test_items=[{"true":keys[t],"pred":keys[p],"conf":round(float(cf),3),"top3":[keys[x] for x in t3],"pos":int(ps)}
            for t,p,cf,t3,ps in zip(yt,pred,conf,top3,(ptr[c] if c is not None else pte))]
json.dump({"train":tr_name,"test_set":te_name,"split":SPLIT,"seed":seed,"epochs":EPOCHS,"best_epoch":best_ep,"val":best,
           "test_accuracy":test,"n_test_presses":len(yte),"n_keys":nc,"keys":keys,"per_key":per_key,"confused_pairs":conf_pairs,
           "test_items":test_items,"hist":hist},open("logs/train_report.json","w"),indent=1)
