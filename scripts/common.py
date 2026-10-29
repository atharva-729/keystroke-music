# Credit: isolator, mel features and model copied unchanged from Eshwar Potturi's replication
# (https://github.com/eshwarpotturi/keystroke-replication, scripts/isolate.py and scripts/train.py),
# which follows "A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards" (arXiv 2308.01074).
# Moved into one module so isolate.py, train.py and build_demo.py share exactly the same code.
import time,numpy as np,librosa,torch,torch.nn as nn
KEYS="1234567890qwertyuiopasdfghjklzxcvbnm"
CLIP=14400
def log(*a): print(time.strftime("%Y-%m-%dT%H:%M:%S"),*a,flush=True)
def isolator(signal,sr,size,scan,before,after,threshold):
    strokes=[]
    fft=librosa.stft(signal,n_fft=size,hop_length=scan)
    energy=np.abs(np.sum(fft,axis=0)).astype(float)
    peaks=np.where(energy>threshold)[0]
    prev_end=sr*0.1*(-1)
    for p in peaks:
        ts=(p*scan)+size//2
        if ts>prev_end+0.1*sr and ts-before>=0 and ts+after<=len(signal):
            strokes.append(signal[ts-before:ts+after]); prev_end=ts+after
    return strokes
def isolator_onsets(signal,sr,size,scan,before,after,threshold):
    """Same logic as isolator(), but also returns each stroke's onset sample (needed to time the notes)."""
    strokes,onsets=[],[]
    fft=librosa.stft(signal,n_fft=size,hop_length=scan)
    energy=np.abs(np.sum(fft,axis=0)).astype(float)
    peaks=np.where(energy>threshold)[0]
    prev_end=sr*0.1*(-1)
    for p in peaks:
        ts=(p*scan)+size//2
        if ts>prev_end+0.1*sr and ts-before>=0 and ts+after<=len(signal):
            strokes.append(signal[ts-before:ts+after]); onsets.append(ts); prev_end=ts+after
    return strokes,onsets
def search(signal,sr,target,iso=isolator):
    """The authors' auto-threshold search: nudge the threshold until exactly `target` strokes are found."""
    prom,step,st=0.06,0.005,[]
    for it in range(2000):
        st=iso(signal,sr,48,24,2400,12000,prom)
        n=len(st[0]) if isinstance(st,tuple) else len(st)
        if n==target: break
        prom+= step if n>target else -step
        step*=0.99
        if prom<=0: break
    return st,prom
def mel_fn(sr):
    fb=torch.tensor(librosa.filters.mel(sr=sr,n_fft=1536,n_mels=64,fmin=100,fmax=2500),dtype=torch.float32)
    win=torch.hann_window(1024)
    def f(x):  # x: (B,T)
        S=torch.stft(x,1536,hop_length=300,win_length=1024,window=win,return_complex=True,center=True).abs()**2
        M=torch.matmul(fb,S)  # (B,64,frames)
        return torch.log2(M+1e-10)
    return f
def feats(X,mel):
    M=mel(X); return M.unsqueeze(1)[...,:48]
class MB(nn.Module):
    def __init__(s,i,o,stride):
        super().__init__(); m=i*2
        s.f=nn.Sequential(nn.BatchNorm2d(i),nn.Conv2d(i,m,1),nn.GELU(),nn.Conv2d(m,m,3,stride,1,groups=m),nn.GELU(),nn.Conv2d(m,o,1))
        s.sc=nn.Sequential(nn.MaxPool2d(stride) if stride>1 else nn.Identity(),nn.Conv2d(i,o,1) if i!=o else nn.Identity())
    def forward(s,x): return s.sc(x)+s.f(x)
class Att(nn.Module):
    def __init__(s,d,heads=4):
        super().__init__(); s.n1=nn.LayerNorm(d); s.a=nn.MultiheadAttention(d,heads,batch_first=True); s.n2=nn.LayerNorm(d)
        s.m=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d))
    def forward(s,x):
        h=s.n1(x); x=x+s.a(h,h,h,need_weights=False)[0]; return x+s.m(s.n2(x))
class CoAtLite(nn.Module):  # conv stem -> 2 MBConv stages -> 2 transformer stages (the CoAtNet recipe, narrower)
    def __init__(s,nc=36,w=(24,48,96,128)):
        super().__init__()
        s.stem=nn.Sequential(nn.Conv2d(1,w[0],3,2,1),nn.GELU(),nn.Conv2d(w[0],w[0],3,1,1))
        s.s1=nn.Sequential(MB(w[0],w[1],2),MB(w[1],w[1],1)); s.s2=nn.Sequential(MB(w[1],w[2],2),MB(w[2],w[2],1))
        s.proj=nn.Conv2d(w[2],w[3],1); s.pos=nn.Parameter(torch.zeros(1,48,w[3])); s.t=nn.Sequential(Att(w[3]),Att(w[3]))
        s.head=nn.Sequential(nn.LayerNorm(w[3]),nn.Dropout(0.3),nn.Linear(w[3],nc))
    def forward(s,x):
        x=s.proj(s.s2(s.s1(s.stem(x)))); B,C,H,W=x.shape; t=x.flatten(2).transpose(1,2)
        return s.head(s.t(t+s.pos).mean(1))
