#!/usr/bin/env python3
"""gitchain-quantize — the paid encode step of the Turbo tier.

Turns a container's raw fp32 embeddings into a compressed, *self-decoding* TurboQuant container:
it writes `.brain/vectors_tq.npz` (the ~10–20× smaller store) AND ships the portable pure-numpy
decoder into the container's `serve/tq_decode.py`, so the customer can run the quantized container
anywhere with just numpy — no GPU, no codec install.

  encode (this tool)  = GPU + the GPU codec  → the value, hosted/gated
  decode (shipped)    = numpy only      → runs anywhere, free

Usage:
  gitchain-quantize.py <container-dir> --vectors <vectors.f32|.npy> [--dims D] [--level b1|b2|b3]

<container-dir> is a container working tree (a checked-out git repo). --vectors is the source
fp32 matrix (raw float32 needs --dims; .npy carries its own shape). Level maps b1=~20×, b2=~10×,
b3=~7× (higher b = higher fidelity, less compression).
"""
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LEVELS = {"b1": 1, "b2": 2, "b3": 3, "b4": 4}


def load_fp32(path, dims):
    if path.endswith(".npy"):
        V = np.load(path)
    else:
        raw = np.fromfile(path, dtype=np.float32)
        if not dims:
            sys.exit("raw .f32 input needs --dims")
        V = raw.reshape(-1, dims)
    return np.ascontiguousarray(V, dtype=np.float32)


def score_recall(V, R, k=10, nq=300):
    n = len(V); qi = np.linspace(0, n - 1, min(nq, n)).astype(int); num = den = 0.0
    Rf = R.astype(np.float32)
    for i in qi:
        q = V[i]; sg = V @ q; sg[i] = -1e30; sc = Rf @ q; sc[i] = -1e30
        num += float(sg[np.argpartition(-sc, k)[:k]].sum())
        den += float(sg[np.argpartition(-sg, k)[:k]].sum())
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("container")
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--dims", type=int, default=0)
    ap.add_argument("--level", default="b1", choices=list(LEVELS))
    ap.add_argument("--no-commit", action="store_true")
    args = ap.parse_args()
    b = LEVELS[args.level]
    cdir = os.path.abspath(args.container)
    os.makedirs(os.path.join(cdir, ".brain"), exist_ok=True)
    os.makedirs(os.path.join(cdir, "serve"), exist_ok=True)

    # --- source fp32, L2-normalized (the container's embeddings) ---
    V = load_fp32(args.vectors, args.dims)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    n, d = V.shape
    fp32_bytes = V.nbytes
    print(f"encoding {n:,} × {d} fp32 vectors  ({fp32_bytes/1e6:.1f} MB)  →  TurboQuant {args.level}", flush=True)

    # --- ENCODE. Sovereign default = pure-numpy tq_encode (no GPU, runs on the Mac).
    #     The GPU tq_store is used only when TQ_STORE_PATH is set (faster, same format). ---
    tq_path = os.path.join(cdir, ".brain", "vectors_tq.npz")
    tq_store_path = os.environ.get("TQ_STORE_PATH")
    t = time.perf_counter()
    if tq_store_path:
        sys.path.insert(0, tq_store_path)            # GPU encoder (paid path)
        import tq_store
        sz = tq_store.save_tq(V, tq_path, b=b)
        enc_kind = "GPU tq_store"
    else:
        sys.path.insert(0, HERE)                     # pure-numpy encoder (sovereign desktop)
        import tq_encode
        sz = tq_encode.save_tq(V, tq_path, b=b)
        enc_kind = "numpy tq_encode"
    t_enc = time.perf_counter() - t
    print(f"  encoder: {enc_kind}", flush=True)

    # --- SHIP the portable decoder into the container (self-decoding) ---
    shutil.copy2(os.path.join(HERE, "tq_decode.py"), os.path.join(cdir, "serve", "tq_decode.py"))

    # --- verify with the SHIPPED decoder (numpy only) ---
    sys.path.insert(0, os.path.join(cdir, "serve"))
    import tq_decode
    R = tq_decode.load_tq(tq_path)
    recall = score_recall(V, R)
    Rn = R.astype(np.float32)
    Rn /= (np.linalg.norm(Rn, axis=1, keepdims=True) + 1e-9)
    cos_fid = float(np.mean(np.sum(V * Rn, axis=1)))   # mittlere Cosinus-Treue Original vs. dequantisiert

    # --- container metadata ---
    idx_path = os.path.join(cdir, ".brain", "index.json")
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else {}
    idx.update({"dims": d, "count": n, "store": f"turboquant-{args.level}",
                "decode": "serve/tq_decode.py (numpy, self-contained)"})
    json.dump(idx, open(idx_path, "w"))
    ga = os.path.join(cdir, ".gitattributes")
    line = ".brain/vectors_tq.npz -diff\n"
    if not (os.path.exists(ga) and line in open(ga).read()):
        open(ga, "a").write(line)

    ratio = fp32_bytes / sz
    # --- Quantum-Karte: die GEMESSENEN Kennzahlen reisen im Container mit (.brain/tq_report.json),
    #     damit Server/UI sie je Container zeigen koennen — wie die Live-Demo-Karten (Haar-Rotation
    #     + Lloyd-Max, arXiv 2504.19874): Kompression, Cosinus-Treue, Score-Recall@10. ---
    card = {"algo": "TurboQuant/PolarQuant (Haar + Lloyd-Max)", "arxiv": "2504.19874",
            "level": args.level, "bits_per_coord": b, "vectors": n, "dims": d,
            "fp32_mb": round(fp32_bytes / 1e6, 2), "tq_mb": round(sz / 1e6, 2),
            "compression": round(ratio, 1), "cos_fidelity": round(cos_fid, 4),
            "score_recall_at_10": round(recall, 4), "encoder": enc_kind,
            "encode_s": round(t_enc, 2), "decode": "serve/tq_decode.py (numpy, self-contained)"}
    json.dump(card, open(os.path.join(cdir, ".brain", "tq_report.json"), "w"), indent=1)
    print(f"  → {sz/1e6:.2f} MB  ({ratio:.1f}× smaller)  ·  CosFidelity {cos_fid:.4f}  ·  ScoreRecall@10 {recall:.4f}", flush=True)
    print(f"  → shipped serve/tq_decode.py — container decodes with numpy only (encode {t_enc:.1f}s)", flush=True)

    if not args.no_commit and os.path.isdir(os.path.join(cdir, ".git")):
        subprocess.run(["git", "-C", cdir, "add", ".brain/vectors_tq.npz", ".brain/tq_report.json",
                        "serve/tq_decode.py", ".brain/index.json", ".gitattributes"], check=True)
        msg = (f"quant: TurboQuant {args.level} — {fp32_bytes/1e6:.1f}MB → {sz/1e6:.2f}MB "
               f"({ratio:.1f}×), ScoreRecall@10 {recall:.4f}; self-decoding (serve/tq_decode.py)")
        subprocess.run(["git", "-C", cdir, "-c", "user.name=gitchain-turbo",
                        "-c", "user.email=turbo@example.com", "commit", "-q", "-m", msg], check=True)
        print(f"  → committed: {msg}", flush=True)

    print(json.dumps({"vectors": n, "dims": d, "level": args.level, "ratio": round(ratio, 1),
                      "recall_at_10": round(recall, 4), "tq_bytes": sz, "self_decoding": True}))


if __name__ == "__main__":
    main()
