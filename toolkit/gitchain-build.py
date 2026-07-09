#!/usr/bin/env python3
"""gitchain-build — create a *knowledge* (brain) container from a folder of documents.

Grounding mode 1 of 2 (see gitchain-atomize.py for mode 2 — verified fact atoms).
Reads each document, splits it into content-addressed chunks, embeds them, and writes a
git container:

  data/chunks.jsonl     one line per chunk {id, doc, source, seq, text}
  .brain/vectors.f32    [n, dims] float32, L2-normalized, in chunk order
  .brain/index.json     {model, dims, count, order:[chunk_id, ...], store:"f32"}
  container.json        container manifest

The result is an fp32 container — free to store and serve with gitchain-lite. Run
gitchain-quantize.py on it afterwards to compress it (the paid Turbo step). Recall is a
cosine search over the chunk embeddings.

Ships NO reader and NO embedder — it talks to them by contract, so you plug in your own:

  --read-url   POST {"filename","content_b64"} -> {"blocks":[str, ...]}
               (optional; if omitted, falls back to a portable local text-layer read via pypdf).
               A hosted deployment points this at a high-fidelity document reader service.
  --embed-url  OpenAI-compatible: POST {"model","input":[str,...]}
               -> {"data":[{"index","embedding"}, ...]}   (e.g. any /v1/embeddings server).

Usage:
  gitchain-build.py --source <docs-dir> --container <out-dir> \
      --embed-url http://localhost:8080/v1/embeddings [--read-url URL] \
      [--model NAME] [--dims 768] [--chunk-len 900] [--id container:my-brain] [--no-commit]
"""
import argparse, base64, glob, hashlib, json, os, subprocess, sys, time, urllib.request
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import glyph_pdf as _GLYPH   # bundled sovereign reader: Glyph (from_pdf + detect_table) over fitz
except Exception:
    _GLYPH = None

ARGS = None  # filled in main(); read by worker via module global (Pool fork)


def read_blocks(path):
    """Return a list of reading-order text blocks for one document."""
    if ARGS.read_url:
        body = json.dumps({"filename": os.path.basename(path),
                           "content_b64": base64.b64encode(open(path, "rb").read()).decode()}).encode()
        req = urllib.request.Request(ARGS.read_url, body, {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=300).read())["blocks"]
    # portable fallback: text-layer extraction (not OCR). For higher fidelity use --read-url.
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("no --read-url given and pypdf not installed — `pip install pypdf`, "
                 "or point --read-url at your reader service")
    blocks = []
    for page in PdfReader(path).pages:
        t = (page.extract_text() or "").strip()
        if t:
            blocks.append(t)
    return blocks


def chunk_doc(path):
    """(source-relative, sha256[:16]-addressed) chunks for one document."""
    rel = os.path.relpath(path, ARGS.source)
    doc_sha = hashlib.sha256(open(path, "rb").read()).hexdigest()

    # Sovereign reader (primary): Glyph reads the PDF text-layer via fitz and detect_table
    # reconstructs each spec row into its own clean chunk. Scans (no text-layer) fall through.
    if _GLYPH is not None and not ARGS.read_url and path.lower().endswith(".pdf"):
        try:
            _snr, scan, gchunks = _GLYPH.read_pdf_chunks(path)
            if not scan and gchunks:
                out = []
                for t in gchunks:
                    t = (t or "").strip()
                    if len(t) >= 20:
                        out.append({"id": hashlib.sha256(t.encode()).hexdigest()[:16],
                                    "doc": doc_sha, "source": rel, "seq": len(out), "text": t[:1400]})
                if out:
                    return (os.path.basename(path), out, None)
        except Exception:
            pass  # fall back to the block reader below

    chunks, buf, seq0 = [], "", 0

    def flush(seq):
        nonlocal buf
        t = buf.strip()
        if len(t) >= 120:
            cid = hashlib.sha256(t.encode()).hexdigest()[:16]
            chunks.append({"id": cid, "doc": doc_sha, "source": rel, "seq": seq, "text": t[:1400]})
        buf = ""

    try:
        for i, blk in enumerate(read_blocks(path)):
            if not buf:
                seq0 = i
            buf += ("\n\n" if buf else "") + blk
            if len(buf) >= ARGS.chunk_len:
                flush(seq0)
        flush(seq0)
        return (os.path.basename(path), chunks, None)
    except Exception as e:
        return (os.path.basename(path), [], str(e)[:160])


def embed_all(texts, model, dims, embed_url, bs=64):
    V = np.zeros((len(texts), dims), np.float32)
    for i in range(0, len(texts), bs):
        body = json.dumps({"model": model, "input": texts[i:i + bs]}).encode()
        req = urllib.request.Request(embed_url, body, {"Content-Type": "application/json"})
        for d in json.loads(urllib.request.urlopen(req, timeout=180).read())["data"]:
            V[i + d["index"]] = d["embedding"]
        if (i // bs) % 20 == 0:
            print(f"  embed {i}/{len(texts)}", flush=True)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return V


def git(cdir, *a):
    subprocess.run(["git", "-C", cdir] + list(a), check=True)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="folder of documents (recursive *.pdf)")
    ap.add_argument("--container", required=True, help="output container dir (a git repo)")
    ap.add_argument("--embed-url", required=True, help="OpenAI-compatible /v1/embeddings endpoint")
    ap.add_argument("--read-url", default="", help="reader service (optional; else local text-layer)")
    ap.add_argument("--model", default="embedding-model", help="embedding model name to send")
    ap.add_argument("--dims", type=int, default=768)
    ap.add_argument("--chunk-len", type=int, default=900)
    ap.add_argument("--glob", default="**/*.pdf", help="source glob relative to --source")
    ap.add_argument("--id", default="", help="container id for the manifest")
    ap.add_argument("--title", default="", help="human title for the manifest")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-commit", action="store_true")
    ARGS = ap.parse_args()

    src, cdir = os.path.abspath(ARGS.source), os.path.abspath(ARGS.container)
    docs = sorted(glob.glob(os.path.join(src, ARGS.glob), recursive=True))
    if not docs:
        sys.exit(f"no documents matched {ARGS.glob} under {src}")
    print(f"gitchain-build: {len(docs)} documents "
          f"({'reader service' if ARGS.read_url else 'local text-layer'})\n", flush=True)
    t0 = time.perf_counter()

    all_chunks, seen, fails = [], set(), []
    with Pool(ARGS.jobs) as pool:
        for n, (fn, chunks, err) in enumerate(pool.imap_unordered(chunk_doc, docs), 1):
            if err:
                fails.append((fn, err))
            for c in chunks:
                if c["id"] not in seen:
                    seen.add(c["id"]); all_chunks.append(c)
            print(f"[{n:3}/{len(docs)}] {fn[:60]:<62} {len(chunks):>5} chunks  "
                  f"(total {len(all_chunks):,})", flush=True)
    print(f"\nread in {time.perf_counter()-t0:.0f}s · {len(all_chunks):,} unique chunks · "
          f"{len(fails)} errors", flush=True)
    for fn, e in fails:
        print(f"  ERROR {fn}: {e}", flush=True)
    if not all_chunks:
        sys.exit("no chunks produced")

    os.makedirs(f"{cdir}/.brain", exist_ok=True)
    os.makedirs(f"{cdir}/data", exist_ok=True)
    with open(f"{cdir}/data/chunks.jsonl", "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nembedding {len(all_chunks):,} chunks ({ARGS.model}, {ARGS.dims}-d) ...", flush=True)
    t1 = time.perf_counter()
    V = embed_all([c["text"] for c in all_chunks], ARGS.model, ARGS.dims, ARGS.embed_url)
    V.tofile(f"{cdir}/.brain/vectors.f32")
    print(f"embedded in {time.perf_counter()-t1:.0f}s -> vectors.f32 {V.nbytes/1e6:.1f} MB", flush=True)

    doc_count = len({c["doc"] for c in all_chunks})
    json.dump({"model": ARGS.model, "dims": ARGS.dims, "count": len(all_chunks),
               "order": [c["id"] for c in all_chunks], "store": "f32"},
              open(f"{cdir}/.brain/index.json", "w"))
    json.dump({"id": ARGS.id or f"container:{os.path.basename(cdir)}", "type": "brain",
               "title": ARGS.title or os.path.basename(cdir),
               "embed_model": ARGS.model, "dims": ARGS.dims,
               "doc_count": doc_count, "chunk_count": len(all_chunks), "store": "f32",
               "recall": "cosine over chunk embeddings",
               "note": "content-addressed chunks (sha256[:16] = id)"},
              open(f"{cdir}/container.json", "w"), indent=2)
    open(f"{cdir}/.gitattributes", "w").write(".brain/vectors.f32 -diff\n")

    if not ARGS.no_commit:
        if not os.path.isdir(f"{cdir}/.git"):
            git(cdir, "init", "-q")
            git(cdir, "symbolic-ref", "HEAD", "refs/heads/main")
        git(cdir, "add", "-A")
        git(cdir, "-c", "user.name=gitchain-build", "-c", "user.email=build@localhost",
            "commit", "-q", "-m",
            f"brain: {len(docs)} documents -> {len(all_chunks)} content-addressed chunks "
            f"({ARGS.model}, {ARGS.dims}-d)")
        print("committed fp32 container", flush=True)

    print(json.dumps({"container": cdir, "type": "brain", "documents": len(docs),
                      "chunks": len(all_chunks), "dims": ARGS.dims,
                      "fp32_mb": round(V.nbytes / 1e6, 1),
                      "next": "gitchain-quantize.py to compress (Turbo tier)"}))
    print(f"\nDONE in {time.perf_counter()-t0:.0f}s — container: {cdir}", flush=True)


if __name__ == "__main__":
    main()
