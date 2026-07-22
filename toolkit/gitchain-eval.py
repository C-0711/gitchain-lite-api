#!/usr/bin/env python3
"""gitchain-eval — Retrieval-Eval eines Lite-Containers gegen ein Goldset.

Goldset (JSONL, eine Frage je Zeile): {"q": Frage, "doc": erwartete Quelldatei,
"value": erwarteter Wert}. Treffer = Atom, dessen source `doc` enthaelt UND dessen
Text (nur Ziffern) den Wert (nur Ziffern) enthaelt — gezaehlt in Top-1/Top-3.
Modi: BM25 immer; dense + RRF(k=60) nur mit --embed-url (OpenAI-kompatibel).

Usage: gitchain-eval.py <containerDir> <goldset.jsonl> [--embed-url URL]
"""
import collections, json, math, os, re, sys, urllib.request
import numpy as np


def nd(v):
    return re.sub(r"\D", "", str(v or ""))


def toks(t):
    return re.findall(r"[a-zäöü]{2,}|\d+", t.lower().replace("ß", "ss"))


class BM25:
    def __init__(self, docs):
        self.docs = [toks(x) for x in docs]
        self.N = len(self.docs)
        self.avg = sum(len(x) for x in self.docs) / max(1, self.N)
        self.df = collections.Counter(w for x in self.docs for w in set(x))
        self.tf = [collections.Counter(x) for x in self.docs]

    def score(self, q):
        sc = np.zeros(self.N, np.float32)
        for w in set(toks(q)):
            df = self.df.get(w)
            if not df:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(w)
                if f:
                    sc[i] += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * len(self.docs[i]) / self.avg))
        return sc


def rrf(rank_lists, k=60):
    sc = collections.defaultdict(float)
    for ranks in rank_lists:
        for r, i in enumerate(ranks):
            sc[i] += 1.0 / (k + r + 1)
    return sorted(sc, key=lambda i: -sc[i])


def embed(url, model, batch):
    """OpenAI-kompatibler /v1/embeddings-Endpoint; Dims aus der ersten Antwort."""
    out = None
    for i in range(0, len(batch), 64):
        body = json.dumps({"model": model, "input": batch[i:i + 64]}).encode()
        r = urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=600)
        for d in json.load(r)["data"]:
            if out is None:
                out = np.zeros((len(batch), len(d["embedding"])), np.float32)
            out[i + d["index"]] = d["embedding"]
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out


def main():
    argv = [a for a in sys.argv[1:]]
    embed_url = None
    model_override = None
    if "--model" in argv:
        j = argv.index("--model"); model_override = argv[j+1]; del argv[j:j+2]
    if "--embed-url" in argv:
        j = argv.index("--embed-url")
        embed_url = argv[j + 1]
        del argv[j:j + 2]
    if len(argv) != 2:
        sys.exit("usage: gitchain-eval.py <containerDir> <goldset.jsonl> [--embed-url URL]")
    cdir, goldf = argv

    chunks = [json.loads(l) for l in open(os.path.join(cdir, "data", "chunks.jsonl")) if l.strip()]
    idx = json.load(open(os.path.join(cdir, ".brain", "index.json")))
    by_id = {c["id"]: c for c in chunks}
    if idx.get("order"):  # Index-Reihenfolge ist massgeblich, Nachzuegler hinten anhaengen
        chunks = [by_id[i] for i in idx["order"] if i in by_id] + [c for c in chunks if c["id"] not in set(idx["order"])]
    gold = [json.loads(l) for l in open(goldf) if l.strip()]
    if not gold:
        sys.exit("leeres Goldset")
    print(f"gitchain-eval: {len(chunks)} Atome ({cdir}), {len(gold)} Fragen ({os.path.basename(goldf)})")

    bm = BM25([c["text"] for c in chunks])
    S = None
    if embed_url:
        model = model_override or idx.get("model") or "embedding-model"
        try:
            C = embed(embed_url, model, [c["text"] for c in chunks])
            Q = embed(embed_url, model, [g["q"] for g in gold])
            S = Q @ C.T
        except Exception as e:
            print(f"! dense uebersprungen ({embed_url}): {str(e)[:100]}")

    modes = ["bm25"] + (["dense", "rrf"] if S is not None else [])
    res = {m: [0, 0] for m in modes}
    oracle = 0
    fails = []
    for qi, g in enumerate(gold):
        gv = nd(g["value"])
        ok = lambda i: g["doc"] in (chunks[i].get("source") or "") and gv and gv in nd(chunks[i]["text"])
        has = any(ok(i) for i in range(len(chunks)))
        oracle += has
        br = list(np.argsort(-bm.score(g["q"])))
        ranked = {"bm25": br}
        if S is not None:
            dr = list(np.argsort(-S[qi]))
            ranked["dense"] = dr
            ranked["rrf"] = rrf([dr[:50], br[:50]])
        for m in modes:
            res[m][0] += any(ok(i) for i in ranked[m][:1])
            res[m][1] += any(ok(i) for i in ranked[m][:3])
        best = ranked["rrf" if S is not None else "bm25"]
        if not any(ok(i) for i in best[:3]):
            diag = "kein Claim/Atom" if not has else "Rank>3"
            fails.append((diag, g["q"], g["doc"], [chunks[i].get("source") for i in best[:3]]))

    n = len(gold)
    print(f"\nOrakel (qualifizierendes Atom existiert): {oracle}/{n} = {100 * oracle / n:.1f}%\n")
    print(f"  {'Modus':6s}  {'recall@1':>8s}  {'recall@3':>8s}")
    for m in modes:
        h1, h3 = res[m]
        print(f"  {m:6s}  {100 * h1 / n:7.1f}%  {100 * h3 / n:7.1f}%")
    if fails:
        print(f"\nFails ({'rrf' if S is not None else 'bm25'}@3): {len(fails)}")
        for diag, q, doc, got in fails:
            print(f"  [{diag}] {q}")
            print(f"    erwartet: {doc}   top-3: {', '.join(str(s) for s in got)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
