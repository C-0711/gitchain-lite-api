#!/usr/bin/env python3
"""tq_decode — the portable TurboQuant decoder that ships INSIDE every quantized container.

Pure numpy, zero heavy dependencies (no torch, no CUDA, no GitChain internals). This is the
*free, runs-anywhere* half of the codec: a container quantized by the encoder carries this file
in its `serve/` directory, so it can be decoded and served on any machine that has numpy.

Single-stage PolarQuant (Haar rotation + Lloyd-Max). This is deliberately Stage-1-only: for the
RAG top-k cosine the desktop does, scalar Lloyd bits dominate a QJL residual per byte. (The
two-stage Max-Lloyd+QJL design lives in the GPU two-stage codec — its payoff is the unbiased
KV-cache inner-product estimate, which this product does not use.)

Store format (`.brain/vectors_tq.npz`):
    codes  [n,d] uint8   Lloyd-Max indices per rotated dimension
    recon  [K]   float32 Lloyd-Max reconstruction levels (the codebook)
    Pi     [d,d] float16 Haar (orthogonal) rotation matrix
    xnorm  [n]   float16 original per-vector L2 norm
    b,d,seed             scalar metadata

Reconstruction: x ≈ ( recon[codes] @ Pi ) · ‖x‖ , then L2-normalize.
This is bit-identical to the GPU decode path — decode never needs the GPU.
"""
import numpy as np


def load_tq(path):
    """Reconstruct L2-normalized fp16 vectors from a TurboQuant store. numpy only."""
    z = np.load(path)
    codes = z["codes"]                                    # [n,d] uint8 (Lloyd indices)
    recon = z["recon"].astype(np.float32)                 # [K] Lloyd reconstruction levels
    Pi = z["Pi"].astype(np.float32)                       # [d,d] Haar matrix
    xnorm = z["xnorm"].astype(np.float32)                 # [n] original norm
    xhat = (recon[codes] @ Pi) * xnorm[:, None]           # inverse Haar · ‖x‖
    xhat /= (np.linalg.norm(xhat, axis=1, keepdims=True) + 1e-9)
    return xhat.astype(np.float16)


def store_info(path):
    """Cheap metadata read (no full decode): dims, count, bits, on-disk size."""
    import os
    z = np.load(path)
    n, d = z["codes"].shape
    return {"vectors": int(n), "dims": int(d), "bits": int(z["b"]) if "b" in z else None,
            "codebook": int(z["recon"].shape[0]), "bytes": os.path.getsize(path)}


if __name__ == "__main__":
    import sys, time
    args = sys.argv[1:]
    if args and args[0] == "--raw":
        # Emit reconstructed vectors as raw little-endian float32 to stdout — the gitchain-lite
        # query path (JS) reads this straight into a Float32Array (decode-on-open).
        V = np.ascontiguousarray(load_tq(args[1]).astype("<f4"))
        sys.stdout.buffer.write(V.tobytes()); sys.exit(0)
    p = args[0] if args else ".brain/vectors_tq.npz"
    t = time.perf_counter(); V = load_tq(p); ms = (time.perf_counter() - t) * 1e3
    print(f"decoded {V.shape[0]:,} × {V.shape[1]} vectors from {p} in {ms:.0f} ms (numpy only)")
    print(f"  info: {store_info(p)}")
