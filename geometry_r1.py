"""Weight geometry over training for a single rank-1 adapter (paper Sec. 4.1, Figs. 7-8). No judging.

For each checkpoint (one per optimizer step) it reads the rank-1 LoRA vectors
  A (input side, 8192) and B (output side = direction written into the residual stream, 2048)
and computes
  size:       |A|, |B|, |scale * B A| (= scale*|A|*|B| for rank 1)
  direction:  cos(B_t, B_{t-5})  how much B turned in the last 5 steps (1 = no turn)
              cos(B_t, B_final)  how close B already points to where it ends up
              step size |B_t - B_{t-1}|  (proxy for gradient/update size; mlx does not log grad norm)
  PCA:        PC1/PC2 of all B vectors (paper: a turning point shows in PC2)

Usage: python geometry_r1.py adapters/llama32_1b_risky_r1_L8_bc
Writes <adapter>/geometry.csv and <adapter>/geometry.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import pandas as pd

d = Path(sys.argv[1])
cfg = json.load(open(d / "adapter_config.json"))
scale = cfg["lora_parameters"]["scale"]
G = cfg.get("grad_accumulation_steps", 1)

files = sorted(d.glob("*_adapters.safetensors"))
steps, As, Bs = [0], [None], [None]
for f in files:
    w = mx.load(str(f))
    a = [v for k, v in w.items() if k.endswith("lora_a")]
    b = [v for k, v in w.items() if k.endswith("lora_b")]
    assert len(a) == len(b) == 1, f"expected exactly one rank-1 adapter in {f.name}"
    steps.append(int(f.name.split("_")[0]) // G)
    As.append(np.array(a[0].astype(mx.float32)).ravel())
    Bs.append(np.array(b[0].astype(mx.float32)).ravel())
As[0] = np.zeros_like(As[1])  # step 0: B = 0 (A random, not saved; |BA| = 0)
Bs[0] = np.zeros_like(Bs[1])
B = np.stack(Bs)
A = np.stack(As)


def cos(x, y):
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return float(x @ y / (nx * ny)) if nx > 0 and ny > 0 else np.nan


df = pd.DataFrame({"step": steps})
df["norm_A"] = np.linalg.norm(A, axis=1)
df["norm_B"] = np.linalg.norm(B, axis=1)
df["norm_update"] = scale * df.norm_A * df.norm_B
df["cos_B_prev5"] = [np.nan if i < 5 else cos(B[i], B[i - 5]) for i in range(len(B))]
df["cos_B_final"] = [cos(B[i], B[-1]) for i in range(len(B))]
df["step_size_B"] = [np.nan] + list(np.linalg.norm(np.diff(B, axis=0), axis=1))
Bc = B[1:] - B[1:].mean(0)
U, S, Vt = np.linalg.svd(Bc, full_matrices=False)
pcs = Bc @ Vt[:2].T
df["pc1"], df["pc2"] = np.r_[np.nan, pcs[:, 0]], np.r_[np.nan, pcs[:, 1]]
var = S**2 / (S**2).sum()
df.to_csv(d / "geometry.csv", index=False)

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
ax[0, 0].plot(df.step, df.norm_B); ax[0, 0].set_title("|B| (size of the learnt direction)")
ax[0, 1].plot(df.step, df.norm_update); ax[0, 1].set_title("|scale·B·A| (size of the weight change)")
ax[0, 2].plot(df.step, df.step_size_B); ax[0, 2].set_title("|B_t − B_t−1| (update size per step)")
ax[1, 0].plot(df.step, 1 - df.cos_B_prev5); ax[1, 0].set_title("1 − cos(B_t, B_t−5)  (turning; peak = rotation)")
ax[1, 1].plot(df.step, df.cos_B_final); ax[1, 1].set_title("cos(B_t, B_final)")
sc = ax[1, 2].scatter(df.pc1, df.pc2, c=df.step, s=6, cmap="viridis")
ax[1, 2].set_title(f"PCA of B (PC1 {var[0]:.0%}, PC2 {var[1]:.0%})"); fig.colorbar(sc, ax=ax[1, 2], label="step")
for a in ax.flat[:5]:
    a.set_xlabel("optimizer step")
fig.suptitle(f"{d.name}: rank-1 adapter geometry over training")
fig.tight_layout()
fig.savefig(d / "geometry.png", dpi=120)
print(df.iloc[::25].round(4).to_string(index=False))
print(f"PC1 {var[0]:.1%}, PC2 {var[1]:.1%}; wrote {d/'geometry.csv'} and {d/'geometry.png'}")
