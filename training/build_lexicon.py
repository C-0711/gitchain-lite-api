#!/usr/bin/env python3
"""Baut den MASSIVEN Lexikon-Container: jedes Wort/Token aus den Text-Layern ALLER
Belegtypen + ELSTER-Katalog-Drucktexte, mit 300m (embeddinggemma) eingebettet.
Ausgabe = gitchain-lite-Container (data/chunks.jsonl + .brain/vectors.f32 + index.json).
Damit wird ein verrauschter Glyph-Readout per Nearest-Neighbor gegen bekanntes Vokabular
korrigiert. Usage: build_lexicon.py <textlayer-base> <elster-catalog.json> <outContainer>"""
import sys, glob, os, json, re, collections, urllib.request
import numpy as np, fitz

base, catpath, out = sys.argv[1], sys.argv[2], sys.argv[3]
EMBED = "http://127.0.0.1:11436/v1/embeddings"

vocab = collections.Counter()
for t in sorted(os.listdir(base)):
    td = os.path.join(base, t)
    if not os.path.isdir(td):
        continue
    for p in sorted(glob.glob(os.path.join(td, "*.pdf"))):
        d = fitz.open(p)
        if not "".join(pg.get_text() for pg in d).strip():
            continue
        for pg in d:
            for w in pg.get_text("words"):
                tok = w[4].strip()
                if 2 <= len(tok) <= 40:
                    vocab[tok] += 1
# ELSTER-Katalog-Drucktexte (kanonisches Feldvokabular)
try:
    cat = json.load(open(catpath))
    for c in cat.values():
        dt = (c.get("drucktext") or "").strip()
        if dt:
            vocab[dt] += 1
            for w in dt.split():
                if 2 <= len(w) <= 40:
                    vocab[w] += 1
except Exception as e:
    print("Katalog uebersprungen:", e, file=sys.stderr)

words = [w for w, _ in vocab.most_common()]
print(f"Lexikon: {len(words)} eindeutige Tokens aus allen Belegtypen + Katalog", flush=True)

def embed(batch):
    body = json.dumps({"model": "embeddinggemma", "input": batch}).encode()
    r = urllib.request.urlopen(urllib.request.Request(EMBED, body, {"Content-Type": "application/json"}), timeout=600)
    dd = json.load(r)["data"]
    return [dd[i]["embedding"] for i in range(len(batch))]

os.makedirs(f"{out}/data", exist_ok=True); os.makedirs(f"{out}/.brain", exist_ok=True)
V = np.zeros((len(words), 768), np.float32)
import hashlib
with open(f"{out}/data/chunks.jsonl", "w") as f:
    for i in range(0, len(words), 128):
        b = words[i:i+128]
        embs = embed(b)
        for j, (w, e) in enumerate(zip(b, embs)):
            V[i+j] = e
            f.write(json.dumps({"id": hashlib.sha1(w.encode()).hexdigest()[:16], "text": w,
                                "source": "belege-lexikon", "count": vocab[w]}, ensure_ascii=False) + "\n")
        if i % 1280 == 0:
            print(f"  eingebettet {i+len(b)}/{len(words)}", flush=True)
Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
Vn.astype(np.float32).tofile(f"{out}/.brain/vectors.f32")
ids = [hashlib.sha1(w.encode()).hexdigest()[:16] for w in words]
json.dump({"model": "embeddinggemma", "dims": 768, "count": len(words), "store": "f32",
           "kind": "lexicon", "order": ids}, open(f"{out}/.brain/index.json", "w"))
np.save(f"{out}/.brain/words.npy", np.array(words))
print(f"Lexikon-Container -> {out}  ({len(words)} Woerter, {V.nbytes//1024//1024} MB Vektoren)")
