#!/usr/bin/env python3
"""Der FINALE Multi-ASCII-MLP fuer ALLE Belegtypen: lernt ALLES (kein Holdout) aus den
auto-gelabelten Glyphen aller 5 Typen, im glyph_clf.npz-Format (W1,b1,W2,b2,classes) —
Drop-in fuer den Glypher-from_image-Pfad. Das ist der 'final read'-Leser, der aus dem
gesamten korrigierten Wissen destilliert wird. Usage: train_final_mlp.py <384d-trainset.npz> <out.npz>"""
import sys, numpy as np, torch, torch.nn as nn

d = np.load(sys.argv[1], allow_pickle=True)
X, y = d["X"].astype(np.float32), d["y"]
classes = np.array(sorted(set(y.tolist())))
cidx = {c: i for i, c in enumerate(classes.tolist())}
yi = np.array([cidx[c] for c in y.tolist()], np.int64)
assert X.shape[1] == 384, f"erwarte 384-d (24x16 norm), habe {X.shape[1]}"
print(f"{len(X)} Glyphen, {len(classes)} ASCII-Klassen, ALLE Belegtypen -> ein MLP", flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
K = len(classes)
Xt = torch.tensor(X, device=dev); yt = torch.tensor(yi, device=dev)
# glyph_clf-Architektur: 384 -> 512 -> K (eine versteckte Schicht, ReLU) — exakt das Format
W1 = torch.zeros(512, 384, device=dev, requires_grad=True)
b1 = torch.zeros(512, device=dev, requires_grad=True)
W2 = torch.zeros(K, 512, device=dev, requires_grad=True)
b2 = torch.zeros(K, device=dev, requires_grad=True)
nn.init.kaiming_uniform_(W1); nn.init.kaiming_uniform_(W2)
opt = torch.optim.Adam([W1, b1, W2, b2], lr=1e-3, weight_decay=1e-5)
lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
def fwd(x): return torch.relu(x @ W1.T + b1) @ W2.T + b2
bs = 4096
for ep in range(60):
    p = torch.randperm(len(Xt), device=dev)
    for i in range(0, len(p), bs):
        idx = p[i:i+bs]; opt.zero_grad()
        lossf(fwd(Xt[idx]), yt[idx]).backward(); opt.step()
    if (ep+1) % 15 == 0:
        with torch.no_grad():
            acc = (fwd(Xt).argmax(1) == yt).float().mean().item()   # train-fit (kein Holdout: learn all)
        print(f"  epoch {ep+1:2d}  train-fit {acc*100:.2f}%", flush=True)
np.savez_compressed(sys.argv[2],
    W1=W1.detach().cpu().numpy(), b1=b1.detach().cpu().numpy(),
    W2=W2.detach().cpu().numpy(), b2=b2.detach().cpu().numpy(),
    classes=np.array([str(c) for c in classes], dtype="<U1"))
print(f"FINAL Multi-ASCII-MLP -> {sys.argv[2]}  (glyph_clf-Format, {K} Klassen, alle Belege)")
