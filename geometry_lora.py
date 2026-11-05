"""Weight-geometry curves for ANY LoRA run with dense checkpoints, using the paper's own metrics.

Paper (Turner et al. 2025, App. F) "local cosine similarity" at step t, with offset s:
    lcs(t) = cos( v_{t-s} - v_t ,  v_{t+s} - v_t )
  straight path -> -1, orthogonal turn -> 0, full reversal -> +1. A rotation shows as a PEAK.
  They ignore points where max(|v_t - v_{t-s}|, |v_{t+s} - v_t|) <= 0.0035 (path barely moving).

What v is:
  rank-1 adapters (one module): v = B (the direction written to the residual stream) and v = A, as in the paper.
  any rank / many modules:      v = the full weight change dW = scale * (A @ B) of every LoRA module, flattened.
    dW is never built: <A1 B1, A2 B2>_F = sum((A1^T A2) * (B1 B2^T)), which only needs r x r matrices.

Also: |dW| (size of the weight change), per-step update |dW_t - dW_{t-1}|, cos(dW_t, dW_final), and for
rank-1 the PCA of the stacked B vectors (paper Fig. 8).

Usage: python geometry_lora.py adapters/<run> [--s 5] [--out-name geometry2]
Writes adapters/<run>/<out-name>.csv and .png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("adapter_dir")
ap.add_argument("--s", type=int, default=5, help="offset in optimizer steps for the local cosine (paper: 5)")
ap.add_argument("--thresh", type=float, default=0.0035, help="paper's movement threshold for B/A vectors")
ap.add_argument("--out-name", default="geometry2")
args = ap.parse_args()

d = Path(args.adapter_dir)
cfg = json.load(open(d / "adapter_config.json"))
scale = cfg["lora_parameters"]["scale"]
G = cfg.get("grad_accumulation_steps", 1)
files = sorted(d.glob("*_adapters.safetensors"), key=lambda f: int(f.name.split("_")[0]))
iters = [int(f.name.split("_")[0]) for f in files]
assert all(i % G == 0 for i in iters), "checkpoints must fall on optimizer steps"
steps = [0] + [i // G for i in iters]


def load(f):
    w = mx.load(str(f))
    mods = sorted({k.rsplit(".", 1)[0] for k in w if k.endswith("lora_a")})
    # mlx LoRA: y = x W^T + scale * (x A) B  ->  dW^T = scale * A @ B, A: (in, r), B: (r, out)
    return {m: (np.array(w[m + ".lora_a"].astype(mx.float32)).astype(np.float64), np.array(w[m + ".lora_b"].astype(mx.float32)).astype(np.float64)) for m in mods}


ckpts = [load(f) for f in files]
mods = list(ckpts[0])
rank = ckpts[0][mods[0]][0].shape[1]
print(f"{d.name}: {len(files)} checkpoints, {len(mods)} LoRA modules, rank {rank}, scale {scale}")
# drop modules that never train (e.g. lora_b stays 0 in untrained blocks of single-layer runs)
mods = [m for m in mods if np.abs(ckpts[-1][m][1]).max() > 0]
print(f"trained modules: {len(mods)}")
n = len(ckpts) + 1  # + step 0 (dW = 0)


def ip(i, j):
    """<dW_i, dW_j>_F summed over modules; index 0 = step 0 (zero update)."""
    if i == 0 or j == 0:
        return 0.0
    a, b = ckpts[i - 1], ckpts[j - 1]
    tot = 0.0
    for m in mods:
        A1, B1 = a[m]; A2, B2 = b[m]
        tot += float(np.sum((A1.T @ A2) * (B1 @ B2.T)))
    return scale * scale * tot


cache = {}
def IP(i, j):
    k = (min(i, j), max(i, j))
    if k not in cache:
        cache[k] = ip(*k)
    return cache[k]


def dist(i, j):
    return np.sqrt(max(IP(i, i) + IP(j, j) - 2 * IP(i, j), 0.0))


def lcs_full(t, s):
    if t - s < 0 or t + s >= n:
        return np.nan
    a, b, c = t - s, t, t + s
    num = IP(a, c) - IP(a, b) - IP(b, c) + IP(b, b)
    den = dist(a, b) * dist(c, b)
    return num / den if den > 0 else np.nan


S = args.s
df = pd.DataFrame({"step": steps})
df["norm_dW"] = [np.sqrt(IP(i, i)) for i in range(n)]
df["step_size_dW"] = [np.nan] + [dist(i, i - 1) for i in range(1, n)]
df["cos_dW_final"] = [np.nan] + [IP(i, n - 1) / np.sqrt(IP(i, i) * IP(n - 1, n - 1)) for i in range(1, n)]
df[f"lcs_dW_s{S}"] = [lcs_full(t, S) for t in range(n)]

if rank == 1 and len(mods) == 1:
    m = mods[0]
    B = np.stack([np.zeros(ckpts[0][m][1].size)] + [c[m][1].ravel() for c in ckpts])
    A = np.stack([ckpts[0][m][0].ravel() * np.nan] + [c[m][0].ravel() for c in ckpts])  # A0 random, not saved

    def lcs_vec(V, t, s, thresh):
        if t - s < 0 or t + s >= len(V) or np.isnan(V[t - s]).any():
            return np.nan, False
        u, w = V[t - s] - V[t], V[t + s] - V[t]
        nu, nw = np.linalg.norm(u), np.linalg.norm(w)
        if nu == 0 or nw == 0:
            return np.nan, False
        return float(u @ w / (nu * nw)), max(nu, nw) > thresh

    for name, V in [("B", B), ("A", A)]:
        vals = [lcs_vec(V, t, S, args.thresh) for t in range(n)]
        df[f"lcs_{name}_s{S}"] = [v for v, _ in vals]
        df[f"lcs_{name}_s{S}_above_thresh"] = [ok for _, ok in vals]
    Bc = B[1:] - B[1:].mean(0)
    _, Sv, Vt = np.linalg.svd(Bc, full_matrices=False)
    pcs = Bc @ Vt[:2].T
    df["pc1_B"], df["pc2_B"] = np.r_[np.nan, pcs[:, 0]], np.r_[np.nan, pcs[:, 1]]
    var = Sv**2 / (Sv**2).sum()

df.to_csv(d / f"{args.out_name}.csv", index=False)

# --- plot ---
rank1 = rank == 1 and len(mods) == 1
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
ax[0, 0].plot(df.step, df.norm_dW); ax[0, 0].set_title("|ΔW| (size of the weight change)")
ax[0, 1].plot(df.step, df.step_size_dW); ax[0, 1].set_title("|ΔW_t − ΔW_t−1| (update size per step)")
ax[0, 2].plot(df.step, df.cos_dW_final); ax[0, 2].set_title("cos(ΔW_t, ΔW_final)")
ax[1, 0].plot(df.step, df[f"lcs_dW_s{S}"]); ax[1, 0].set_title(f"local cosine of ΔW path, s={S}  (−1 straight, peak = turn)")
if rank1:
    for name, a in [("B", ax[1, 1]), ("A", ax[1, 2])]:
        col = f"lcs_{name}_s{S}"; ok = df[f"{col}_above_thresh"]
        a.plot(df.step, df[col], color="0.75", label="all points")
        a.plot(df.step.where(ok), df[col].where(ok), label=f"moved > {args.thresh}")
        a.set_title(f"paper metric: local cosine of {name}, s={S}"); a.legend(fontsize=8)
else:
    ax[1, 1].axis("off"); ax[1, 2].axis("off")
for a in ax.flat:
    if a.has_data():
        a.set_xlabel("optimizer step")
fig.suptitle(f"{d.name}: rank {rank}, {len(mods)} trained module(s), scale {scale}")
fig.tight_layout()
fig.savefig(d / f"{args.out_name}.png", dpi=110)

# print the turning points
for col in [c for c in df.columns if c.startswith("lcs_") and not c.endswith("thresh")]:
    s_ = df[col].copy()
    s_[df.step < 2 * S] = np.nan  # skip the start (path leaves 0)
    if s_.notna().any():
        print(f"{col}: max {s_.max():.3f} at step {int(df.step[s_.idxmax()])}; median {s_.median():.3f}")
print(f"wrote {d / (args.out_name + '.csv')} and .png")
