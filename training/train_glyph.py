#!/usr/bin/env python3
"""Trainiert den Glypher-MLP (384->512->K) auf dem auto-gelabelten Testmengen-Trainset.
Speichert im GLEICHEN Format wie glyph_clf.npz (W1,b1,W2,b2,classes) -> direkt einsetzbar
im Bild-Leser (from_image), der die Scans liest. Nutzt Torch/GPU wenn da, sonst CPU."""
import sys, numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
X, y = d["X"].astype(np.float32), d["y"]
classes = np.array(sorted(set(y.tolist())))
cidx = {c: i for i, c in enumerate(classes.tolist())}
yi = np.array([cidx[c] for c in y.tolist()], np.int64)
rng = np.random.default_rng(711); perm = rng.permutation(len(X))
X, yi = X[perm], yi[perm]
nval = len(X) // 10
Xtr, ytr, Xva, yva = X[nval:], yi[nval:], X[:nval], yi[:nval]

import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
K = len(classes)
Xtr_t = torch.tensor(Xtr, device=dev); ytr_t = torch.tensor(ytr, device=dev)
Xva_t = torch.tensor(Xva, device=dev); yva_t = torch.tensor(yva, device=dev)
W1 = torch.zeros(512, 384, device=dev, requires_grad=True)
b1 = torch.zeros(512, device=dev, requires_grad=True)
W2 = torch.zeros(K, 512, device=dev, requires_grad=True)
b2 = torch.zeros(K, device=dev, requires_grad=True)
torch.nn.init.kaiming_uniform_(W1); torch.nn.init.kaiming_uniform_(W2)
opt = torch.optim.Adam([W1, b1, W2, b2], lr=1e-3)
lossf = torch.nn.CrossEntropyLoss()
def fwd(x): return (torch.relu(x @ W1.T + b1)) @ W2.T + b2
bs = 4096
for ep in range(12):
    p = torch.randperm(len(Xtr_t), device=dev)
    for i in range(0, len(p), bs):
        idx = p[i:i+bs]; opt.zero_grad()
        loss = lossf(fwd(Xtr_t[idx]), ytr_t[idx]); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (fwd(Xva_t).argmax(1) == yva_t).float().mean().item()
    print(f"  epoch {ep+1:2d}  val-acc {acc*100:.2f}%", flush=True)
np.savez_compressed(sys.argv[2],
    W1=W1.detach().cpu().numpy(), b1=b1.detach().cpu().numpy(),
    W2=W2.detach().cpu().numpy(), b2=b2.detach().cpu().numpy(),
    classes=np.array([str(c) for c in classes], dtype="<U1"))
print(f"gespeichert -> {sys.argv[2]}  ({K} Klassen, {len(Xtr)} train / {len(Xva)} val)")
