#!/usr/bin/env python3
"""Reiche Glyph-Trainingsdaten aus ALLEN Text-Layer-Belegtypen (auto-gelabelt).
Feature = norm(384, ink) + rc_vec(384, grau) = 768-d; plus holes(1). Traegt doc-id und
doc-type, damit dokument-weise (ehrlich) gesplittet und pro Typ gemessen werden kann.
Usage: gen_v2.py <baseDir-mit-Typ-Unterordnern> <out.npz>"""
import sys, glob, os
import numpy as np, cv2, fitz

DPI = 200; SC = DPI / 72.0

def norm(c):
    ys, xs = np.where(c > 0)
    if len(xs) == 0:
        return None
    cc = c[ys.min():ys.max()+1, xs.min():xs.max()+1].astype(np.float32); h, w = cc.shape
    s = min(24.0/h, 16.0/w); nh, nw = max(1, int(h*s)), max(1, int(w*s)); r = cv2.resize(cc, (nw, nh))
    o = np.zeros((24, 16), np.float32); o[(24-nh)//2:(24-nh)//2+nh, (16-nw)//2:(16-nw)//2+nw] = r
    return o.ravel()

def rc_vec(cg):
    if cg.size == 0:
        return np.zeros(384, np.float32)
    r = cv2.resize(cg, (16, 24)).astype(np.float32); mn, mx = float(r.min()), float(r.max())
    return ((r - mn) / (mx - mn + 1e-6)).ravel()

def holes(bw):
    cn, h = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return int(sum(1 for x in h[0] if x[3] != -1)) if h is not None else 0

base = sys.argv[1]
types = [d for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, d))]
X, Y, DT, DID = [], [], [], []
for ti, t in enumerate(types):
    pdfs = sorted(glob.glob(os.path.join(base, t, "*.pdf")))
    ndoc = 0
    for di, path in enumerate(pdfs):
        pdf = fitz.open(path)
        if not "".join(p.get_text() for p in pdf).strip():
            continue
        ndoc += 1
        docid = f"{ti}_{di}"
        for page in pdf:
            pm = page.get_pixmap(dpi=DPI)
            g = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)[:, :, 0]
            ink = (g < 160).astype(np.uint8)
            for b in page.get_text("rawdict")["blocks"]:
                for l in b.get("lines", []):
                    for s in l.get("spans", []):
                        for c in s.get("chars", []):
                            ch = c["c"]
                            if not ch.strip() or len(ch) != 1:
                                continue
                            x0, y0, x1, y1 = [v*SC for v in c["bbox"]]
                            x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), int(x1)+1, int(y1)+1
                            crop = ink[y0:y1, x0:x1]; cg = g[y0:y1, x0:x1]
                            if crop.size == 0 or crop.sum() < 1:
                                continue
                            f = norm(crop)
                            if f is None:
                                continue
                            feat = np.concatenate([f, rc_vec(cg), [holes(crop)*1.0]]).astype(np.float32)
                            X.append(feat); Y.append(ch); DT.append(t); DID.append(docid)
    print(f"  {t}: {ndoc} textlayer docs", flush=True)
X = np.asarray(X, np.float32)
np.savez_compressed(sys.argv[2], X=X, y=np.array(Y), dt=np.array(DT), did=np.array(DID))
print(f"{len(X)} Glyphen, {len(set(Y))} Klassen, {len(types)} Typen, dim={X.shape[1]} -> {sys.argv[2]}")
