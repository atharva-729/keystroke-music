"""Isolate keystrokes with the authors' own isolator (from the dataset README). Logs a timestamp on every line.
Credit: adapted from Eshwar Potturi's replication, https://github.com/eshwarpotturi/keystroke-replication (scripts/isolate.py).
Changes: phone recordings (MBPWavs) only, reads ../sources/ksd, saves the key list, keeps the first 25 strokes if more are found."""
import json,numpy as np,librosa
from common import KEYS,CLIP,log,isolator,search
X,y,fid,info,thr=[],[],[],{},{}
for i,k in enumerate(KEYS):
    s,sr=librosa.load(f"../sources/ksd/MBPWavs/{k.upper()}.wav",sr=None)  # letter files are uppercase, labels lowercase
    s=s[int(1*sr):]
    st,prom=search(s,sr,25)
    st=st[:25]
    info[k]=len(st); thr[k]=round(prom,4)
    for j,x in enumerate(st): X.append(x);y.append(i);fid.append(j)
    log("phone",k,"found",len(st),"thr",round(prom,4),f"{i+1}/36")
np.savez("data/phone.npz",X=np.array([np.pad(x,(0,CLIP-len(x))) for x in X]),y=np.array(y),pos=np.array(fid),sr=sr,keys=np.array(list(KEYS)))
bad={k:v for k,v in info.items() if v!=25}
log("phone","total strokes",len(X),"keys not at 25:",bad)
json.dump({"strokes":info,"threshold":thr,"total":len(X),"not_25":bad,"sr":int(sr)},open("logs/isolate_report.json","w"),indent=1)
