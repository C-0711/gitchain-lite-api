#!/usr/bin/env python3
"""Duden-Korrektur des CNN-Wort-Readouts: fuellt die Luecken mit 356k deutschen Vollformen.
Gesetz (wie glypher duden.py): im Duden -> behalten; sonst EINDEUTIGER Kandidat gleicher
Laenge mit 1-2 Abweichungen -> korrigieren; sonst roh lassen. Misst Wort-Accuracy pro Typ."""
import glob, os, json, re
import numpy as np, cv2, fitz, torch, torch.nn as nn

BASE = os.path.expanduser("~/C-Force/Textlayer")
GH, GW = 32, 24; SC = 300/72.0

def norm(c):
    ys, xs = np.where(c > 0)
    if len(xs) == 0: return None
    cc = c[ys.min():ys.max()+1, xs.min():xs.max()+1].astype(np.float32); h, w = cc.shape
    s = min(GH/h, GW/w); nh, nw = max(1,int(h*s)), max(1,int(w*s)); r = cv2.resize(cc,(nw,nh))
    o = np.zeros((GH,GW),np.float32); o[(GH-nh)//2:(GH-nh)//2+nh,(GW-nw)//2:(GW-nw)//2+nw]=r; return o
def rc(cg):
    if cg.size==0: return np.zeros((GH,GW),np.float32)
    r=cv2.resize(cg,(GW,GH)).astype(np.float32); mn,mx=float(r.min()),float(r.max()); return (r-mn)/(mx-mn+1e-6)

d = np.load(os.path.expanduser("~/C-Force/glyph_v3.npz"), allow_pickle=True)
classes = np.array(sorted(set(d["y"].tolist())))
class CNN(nn.Module):
    def __init__(s,K):
        super().__init__(); s.c=nn.Sequential(nn.Conv2d(2,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.MaxPool2d(2))
        s.f=nn.Sequential(nn.Flatten(),nn.Linear(128*4*3,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,K))
    def forward(s,x): return s.f(s.c(x))
net=CNN(len(classes)).cuda(); net.load_state_dict(torch.load(os.path.expanduser("~/C-Force/glyph_cnn.pt"))); net.eval()

# Duden laden, nach Laenge indizieren
DUD = [w.strip() for w in open(os.path.expanduser("~/glyph-tester/duden_de.txt"), encoding="utf-8", errors="replace") if w.strip()]
dset = set(DUD)
by_len = {}
for w in DUD: by_len.setdefault(len(w), []).append(w)
print(f"Duden: {len(DUD)} Vollformen", flush=True)

def lev(a, b, cap=2):
    """Levenshtein (Einfuegen/Loeschen/Ersetzen) mit frueher Abbruch bei > cap."""
    la, lb = len(a), len(b)
    if abs(la - lb) > cap: return cap + 1
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb; rowmin = cur[0]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + (a[i-1] != b[j-1]))
            if cur[j] < rowmin: rowmin = cur[j]
        if rowmin > cap: return cap + 1               # ganze Zeile ueber cap -> raus
        prev = cur
    return prev[lb]

def correct(pred):
    """Duden schlaegt vor via Levenshtein (fehlende/falsche Buchstaben). Bekannt -> roh;
    sonst EINDEUTIG bester Kandidat (Distanz <=2, Laenge +-2) -> ersetzen; sonst roh."""
    if pred in dset or not pred.isalpha():
        return pred
    best = []
    for dl in (0, -1, 1, -2, 2):                       # auch fehlende/zusaetzliche Buchstaben
        for w in by_len.get(len(pred) + dl, []):
            dd = lev(pred, w, cap=2)
            if dd <= 2:
                best.append((dd, w))
    if not best: return pred
    best.sort()
    if len(best) == 1 or best[0][0] < best[1][0]:      # eindeutig bester Abstand
        return best[0][1]
    return pred

rng = np.random.default_rng(711)
types = [t for t in sorted(os.listdir(BASE)) if os.path.isdir(os.path.join(BASE,t))]
tot_raw = tot_cor = tot_n = 0
for t in types:
    pdfs = [p for p in sorted(glob.glob(f"{BASE}/{t}/*.pdf")) if "".join(pg.get_text() for pg in fitz.open(p)).strip()]
    k = max(1, int(round(len(pdfs)*0.15))); held = set(rng.permutation(pdfs)[:k].tolist())
    tw_, pw_ = [], []
    for p in held:
        for page in fitz.open(p):
            pm=page.get_pixmap(dpi=300); g=np.frombuffer(pm.samples,np.uint8).reshape(pm.h,pm.w,pm.n)[:,:,0]; ink=(g<160).astype(np.uint8)
            chars=[]
            for b in page.get_text("rawdict")["blocks"]:
                for l in b.get("lines",[]):
                    for s in l.get("spans",[]):
                        for c in s.get("chars",[]):
                            if c["c"].strip() and len(c["c"])==1: chars.append((c["c"],[v*SC for v in c["bbox"]]))
            for w in page.get_text("words"):
                tw=w[4].strip()
                if not (2<=len(tw)<=40): continue
                wb=[v*SC for v in w[:4]]
                wc=[(ch,bb) for ch,bb in chars if bb[0]>=wb[0]-2 and bb[2]<=wb[2]+2 and (bb[1]+bb[3])/2>=wb[1]-2 and (bb[1]+bb[3])/2<=wb[3]+2]
                if not wc: continue
                wc.sort(key=lambda x:x[1][0]); feats=[]
                for ch,bb in wc:
                    x0,y0,x1,y1=max(0,int(bb[0])),max(0,int(bb[1])),int(bb[2])+1,int(bb[3])+1
                    nf=norm(ink[y0:y1,x0:x1])
                    if nf is None: feats=None; break
                    feats.append(np.stack([nf,rc(g[y0:y1,x0:x1])]))
                if not feats: continue
                pr=net(torch.tensor(np.array(feats),device="cuda")).argmax(1).cpu().numpy()
                tw_.append(tw); pw_.append("".join(classes[i] for i in pr))
    raw=np.mean([a==b for a,b in zip(tw_,pw_)])
    cor=np.mean([correct(b)==a for a,b in zip(tw_,pw_)])
    n=len(pw_); tot_raw+=raw*n; tot_cor+=cor*n; tot_n+=n
    print(f"  {t:32s} roh {100*raw:5.1f}%  ->  +Duden {100*cor:5.1f}%  (n={n})", flush=True)
print(f"\nGESAMT Wort-Readout: roh {100*tot_raw/tot_n:.1f}%  ->  +Duden {100*tot_cor/tot_n:.1f}%")
