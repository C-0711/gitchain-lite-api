#!/usr/bin/env python3
"""Glypher-Trainingsdaten aus Testmengen-Textlayer-PDFs (AUTO-GELABELT).
Jede Seite -> Graustufenbild (200dpi); jede Text-Layer-Glyphe (Zeichen + bbox aus fitz
rawdict) -> Ink-Crop -> norm() 384-d Feature + Zeichen-Label. Format = exakt der Eingang
des Glypher-MLP (glyph_clf.npz: 384-d -> 111 Klassen). Damit laesst sich der BILD-Leser
(from_image, der die 46 Scans liest) auf genau diesen Formularfonts trainieren.
Usage: gen_glyph_trainset.py <pdfDir> <out.npz>"""
import sys, glob, os
import numpy as np, cv2, fitz

DPI = 200; SC = DPI / 72.0

def norm(c):  # byte-identisch zu glypher_core.geometry.norm
    ys, xs = np.where(c > 0)
    if len(xs) == 0:
        return None
    cc = c[ys.min():ys.max()+1, xs.min():xs.max()+1].astype(np.float32); h, w = cc.shape
    s = min(24.0/h, 16.0/w); nh, nw = max(1, int(h*s)), max(1, int(w*s)); r = cv2.resize(cc, (nw, nh))
    o = np.zeros((24, 16), np.float32); o[(24-nh)//2:(24-nh)//2+nh, (16-nw)//2:(16-nw)//2+nw] = r
    return o.ravel()

def main():
    pdfs = sorted(glob.glob(os.path.join(sys.argv[1], "*.pdf")))
    X, Y = [], []
    docs = skipped = 0
    for path in pdfs:
        pdf = fitz.open(path)
        if not "".join(p.get_text() for p in pdf).strip():
            skipped += 1; continue                      # Scan: hier kein Label (spaeter Testziel)
        docs += 1
        for page in pdf:
            pm = page.get_pixmap(dpi=DPI)
            g = np.frombuffer(pm.samples, np.uint8).reshape(pm.h, pm.w, pm.n)[:, :, 0]
            ink = (g < 160).astype(np.uint8)             # Tinte = dunkel
            for b in page.get_text("rawdict")["blocks"]:
                for l in b.get("lines", []):
                    for s in l.get("spans", []):
                        for c in s.get("chars", []):
                            ch = c["c"]
                            if not ch.strip() or len(ch) != 1:
                                continue
                            x0, y0, x1, y1 = [v*SC for v in c["bbox"]]
                            x0, y0 = max(0, int(x0)), max(0, int(y0))
                            x1, y1 = int(x1)+1, int(y1)+1
                            crop = ink[y0:y1, x0:x1]
                            if crop.size == 0 or crop.sum() < 1:
                                continue
                            f = norm(crop)
                            if f is not None:
                                X.append(f); Y.append(ch)
    X = np.asarray(X, np.float32); Y = np.asarray(Y)
    np.savez_compressed(sys.argv[2], X=X, y=Y)
    import collections
    top = collections.Counter(Y).most_common(12)
    print(f"{docs} Textlayer-Docs ({skipped} Scans uebersprungen = Testziel), "
          f"{len(X)} gelabelte Glyphen, {len(set(Y))} Klassen -> {sys.argv[2]}")
    print("Top-Klassen:", " ".join(f"{repr(k)}:{v}" for k, v in top))

if __name__ == "__main__":
    main()
