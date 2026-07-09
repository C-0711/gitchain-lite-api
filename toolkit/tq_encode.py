#!/usr/bin/env python3
"""Portable TurboQuant ENCODER — pure numpy, no GPU/torch. The CPU counterpart to the GPU `tq_store`:
it writes the exact `.brain/vectors_tq.npz` format that `tq_decode.py` reads, so a container compressed
here decodes anywhere with numpy only. This is what makes "TurboQuant" a per-container option in the
sovereign desktop product — no GPU, no codec install on the customer machine.

Single-stage PolarQuant (matches the GPU tq_store / the GPU codec PolarEncoder). Quality is dialed by `b`:
    b1 ≈ 14×   b2 ≈ 10×   b3 ≈ 7×   (higher b = higher fidelity, less compression)
For RAG top-k cosine, raising `b` is the right lever — it beats a QJL residual per byte (the two-stage
Max-Lloyd+QJL design is only worthwhile for the unbiased KV-cache inner-product estimate, which lives
in the GPU two-stage codec, not here).

Algorithm (matches the decode contract  xhat = (recon[codes] @ Pi) * xnorm , then L2-normalize):
  1. L2-normalize the vectors.
  2. Rotate by a Haar-random orthonormal matrix Pi (QR of a Gaussian) — spreads energy so every
     coordinate has ~the same marginal (TurboQuant's key trick).
  3. 1-D Lloyd–Max quantize the pooled rotated coordinates into 2^b shared levels → codes + codebook.
"""
import os
import numpy as np


def save_tq(V, path, b=2, seed=0, lloyd_iters=30):
    """Encode [n,d] float32 vectors into a TurboQuant store at `path` (.npz). Returns bytes on disk."""
    V = np.ascontiguousarray(V, dtype=np.float32)
    n, d = V.shape
    xnorm = (np.linalg.norm(V, axis=1) + 1e-9).astype(np.float32)
    Vn = V / xnorm[:, None]

    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))                      # Haar orthonormal…
    Pi = (Q * np.sign(np.diag(R))).astype(np.float32)                    # …sign-fixed (proper Haar)
    Y = Vn @ Pi.T                                                         # decode does recon[codes] @ Pi

    K = 1 << b
    y = Y.ravel()
    lo, hi = np.quantile(y, [0.001, 0.999])
    levels = np.linspace(lo, hi, K).astype(np.float32)
    for _ in range(lloyd_iters):                                         # 1-D Lloyd–Max (k-means in 1D)
        bnd = (levels[:-1] + levels[1:]) * 0.5
        codes = np.searchsorted(bnd, y)
        new = levels.copy()
        for k in range(K):
            m = codes == k
            if m.any():
                new[k] = y[m].mean()
        new.sort()
        if np.allclose(new, levels, atol=1e-6):
            levels = new
            break
        levels = new

    bnd = (levels[:-1] + levels[1:]) * 0.5
    codes = np.searchsorted(bnd, Y.ravel()).reshape(n, d).astype(np.uint8)
    outp = path if path.endswith(".npz") else path + ".npz"
    np.savez_compressed(outp, codes=codes, recon=levels.astype(np.float32),
                        Pi=Pi.astype(np.float16), xnorm=xnorm.astype(np.float16),
                        b=np.int32(b), d=np.int32(d), seed=np.int32(seed))  # low-b codes → ~b bits/coord
    return os.path.getsize(outp)


# Decode identical to tq_decode.load_tq (kept here so this file round-trips standalone).
def load_tq(path):
    z = np.load(path)
    codes = z["codes"]
    recon = z["recon"].astype(np.float32)
    Pi = z["Pi"].astype(np.float32)
    xnorm = z["xnorm"].astype(np.float32)
    xhat = (recon[codes] @ Pi) * xnorm[:, None]
    xhat /= (np.linalg.norm(xhat, axis=1, keepdims=True) + 1e-9)
    return xhat.astype(np.float16)
