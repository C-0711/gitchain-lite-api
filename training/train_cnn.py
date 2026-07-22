#!/usr/bin/env python3
"""CNN-Glyph-Reader: nutzt die 2D-Struktur (32x24 ink + 32x24 grau = 2 Kanaele), die der
flache MLP wegwirft. Dokument-weiser Split (ehrlich), pro Belegtyp. Ziel 97%."""
import sys, numpy as np, torch, torch.nn as nn

d = np.load(sys.argv[1], allow_pickle=True)
X, y, dt, did = d["X"].astype(np.float32), d["y"], d["dt"], d["did"]
GH, GW = 32, 24
img = np.stack([X[:, :GH*GW].reshape(-1, GH, GW), X[:, GH*GW:2*GH*GW].reshape(-1, GH, GW)], 1)  # (N,2,32,24)
classes = np.array(sorted(set(y.tolist()))); cidx = {c: i for i, c in enumerate(classes.tolist())}
yi = np.array([cidx[c] for c in y.tolist()], np.int64)

rng = np.random.default_rng(711)
docs = np.array([f"{t}|{i}" for t, i in zip(dt, did)]); uniq = np.array(sorted(set(docs.tolist())))
test = set()
for t in sorted(set(dt.tolist())):
    dd = [u for u in uniq if u.startswith(t + "|")]; k = max(1, int(round(len(dd)*0.15)))
    for u in rng.permutation(dd)[:k]: test.add(u)
is_te = np.array([x in test for x in docs]); tr = ~is_te
dev = "cuda"; K = len(classes)
Xtr = torch.tensor(img[tr]); ytr = torch.tensor(yi[tr])          # CPU-resident
Xte = torch.tensor(img[is_te]); yte = torch.tensor(yi[is_te])
print(f"{tr.sum()} train / {is_te.sum()} test; {K} Klassen", flush=True)

class CNN(nn.Module):
    def __init__(s):
        super().__init__()
        s.c = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 16x12
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 8x6
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))  # 4x3
        s.f = nn.Sequential(nn.Flatten(), nn.Linear(128*4*3, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, K))
    def forward(s, x): return s.f(s.c(x))

net = CNN().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
bs = 2048
def aug(x):  # leichte Jitter/Noise gegen die Render->Scan-Luecke
    n = torch.randn_like(x) * 0.05
    return (x + n).clamp(0, 1)
for ep in range(60):
    net.train(); p = torch.randperm(len(Xtr))
    for i in range(0, len(p), bs):
        idx = p[i:i+bs].cpu()
        xb = aug(Xtr[idx].to(dev)); yb = ytr[idx].to(dev)
        opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
    sched.step()
    if (ep+1) % 10 == 0:
        net.eval()
        with torch.no_grad():
            cor = 0
            for i in range(0, len(Xte), 2048):
                cor += (net(Xte[i:i+2048].to(dev)).argmax(1).cpu() == yte[i:i+2048]).sum().item()
        print(f"  CNN epoch {ep+1:2d}  test-acc {100*cor/len(Xte):.2f}%", flush=True)
net.eval()
with torch.no_grad():
    pred = np.concatenate([net(Xte[i:i+2048].to(dev)).argmax(1).cpu().numpy() for i in range(0, len(Xte), 2048)])
yt = yte.cpu().numpy(); dte = dt[is_te]
print(f"\nCNN: Gesamt-Readout {100*(pred==yt).mean():.2f}%")
for t in sorted(set(dte.tolist())):
    m = dte == t
    print(f"    {t:32s} {100*(pred[m]==yt[m]).mean():.2f}%  (n={int(m.sum())})")
torch.save(net.state_dict(), sys.argv[2])
print(f"gespeichert -> {sys.argv[2]}")
