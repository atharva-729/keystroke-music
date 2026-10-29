"""Build data/demo.json: stitch held-out clicks (presses 23-25, never trained on) into phrases, then isolate and
classify the stitched audio the way an eavesdropper would (no known clip positions are given to the model).
usage: python scripts/build_demo.py"""
import json,base64,numpy as np,librosa,torch
from common import log,search,isolator_onsets,mel_fn,feats,CoAtLite,CLIP
PHRASES=["password123","hello world","straive","open the pod bay doors","qwerty","zoom call 2026"]
BEFORE=2400          # the click sits this many samples into each clip (the isolator's window start)
OUT_SR=16000
rng=np.random.RandomState(0)
d=np.load("data/phone.npz"); X,y,pos,sr=d["X"],d["y"],d["pos"],int(d["sr"]); keys=[str(k) for k in d["keys"]]
held={k:[X[(y==i)&(pos==p)][0] for p in sorted(set(pos[(y==i)&(pos>=22)]))] for i,k in enumerate(keys)}  # last three by position
ck=torch.load("models/phone.pt",weights_only=False)
model=CoAtLite(nc=len(ck["keys"])); model.load_state_dict(ck["state"]); model.eval()
mel=mel_fn(ck["sr"]); assert ck["sr"]==sr and ck["keys"]==keys
def classify(strokes):
    S=torch.tensor(np.array([np.pad(s,(0,CLIP-len(s))) for s in strokes]),dtype=torch.float32)
    with torch.no_grad(): P=torch.softmax(model((feats(S,mel)-ck["mu"])/ck["sd"]),1)
    return P.numpy()
def stitch(text,lo,hi):
    """Place one held-out clip per character; click onsets are lo..hi s apart, a space adds 1.0 s."""
    used={k:0 for k in keys}; chars=[c for c in text if c!=" "]; onsets=[]; t=0.5; gap_pending=0.0
    for idx,c in enumerate(text):
        if c==" ": gap_pending+=1.0; continue
        if onsets: t+=rng.uniform(lo,hi)+gap_pending
        gap_pending=0.0; onsets.append(int(t*sr))
    track=np.zeros(onsets[-1]+CLIP+int(0.5*sr),dtype=np.float32)
    for c,o in zip(chars,onsets):
        clip=held[c][used[c]%len(held[c])]; used[c]+=1
        track[o-BEFORE:o-BEFORE+CLIP]+=clip           # overlapping tails add together
    return track,chars,onsets
phrases,report=[],[]
for text in PHRASES:
    iso=True; gaps=(0.45,0.60)
    for attempt in range(2):
        track,chars,true_on=stitch(text,*gaps)
        (strokes,found_on),thr=search(track,sr,len(chars),iso=isolator_onsets)
        if len(strokes)==len(chars): break
        log(f"'{text}': isolator found {len(strokes)} of {len(chars)} at gaps {gaps}; retrying wider"); gaps=(0.60,0.75)
    if len(strokes)!=len(chars):
        log(f"'{text}': still {len(strokes)} of {len(chars)}; falling back to known onsets for this phrase")
        iso=False; found_on=true_on; strokes=[track[o-BEFORE:o-BEFORE+CLIP] for o in true_on]
    drift=[abs(f-t)/sr for f,t in zip(found_on,true_on)]   # sanity: detected onsets should sit on the real clicks
    P=classify(strokes); pred=P.argmax(1)
    presses=[{"t":round(o/sr,3),"true":c,"pred":keys[p],"conf":round(float(P[i,p]),3),
              "top3":[[keys[j],round(float(P[i,j]),3)] for j in np.argsort(-P[i])[:3]]}
             for i,(o,c,p) in enumerate(zip(found_on,chars,pred))]
    right=sum(p["true"]==p["pred"] for p in presses)
    play=track/np.abs(track).max()*0.9                       # normalised for listening only
    a16=np.clip(librosa.resample(play,orig_sr=sr,target_sr=OUT_SR)*32767,-32768,32767).astype("<i2")
    phrases.append({"text":text,"sr":OUT_SR,"audio":base64.b64encode(a16.tobytes()).decode(),"isolated_from_audio":iso,
                    "duration":round(len(a16)/OUT_SR,3),"presses":presses})
    report.append({"text":text,"right":right,"n":len(chars),"isolated_from_audio":iso,"gaps":gaps,"threshold":round(thr,4),"max_onset_drift_s":round(max(drift),3)})
    log(f"'{text}': {right}/{len(chars)} right, isolated_from_audio={iso}, gaps={gaps}, max onset drift {max(drift)*1000:.0f} ms, "
        f"read as '{''.join(p['pred'] for p in presses)}'")
tr=json.load(open("logs/train_report.json"))
R=sum(r["right"] for r in report); N=sum(r["n"] for r in report)
demo={"test_accuracy":tr["test_accuracy"],"n_keys":tr["n_keys"],"n_test_presses":tr["n_test_presses"],
      "phrase_accuracy":R/N,"phrase_right":R,"phrase_presses":N,"per_key":tr["per_key"],"confused_pairs":tr["confused_pairs"],
      "test_items":tr["test_items"],"phrases":phrases}
json.dump(demo,open("data/demo.json","w"),separators=(",",":"))
json.dump({"phrases":report,"phrase_accuracy":R/N,"test_accuracy":tr["test_accuracy"]},open("logs/demo_report.json","w"),indent=1)
log(f"phrase accuracy {R}/{N} = {R/N:.3f} vs test accuracy {tr['test_accuracy']:.3f}")
