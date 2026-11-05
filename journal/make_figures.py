"""Figures for the overnight journal (2026-11-04). Reads only files produced by the runs; writes PNGs next to this file.

Run from my_repro/:  python journal/make_figures.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
A = ROOT / "adapters"


def summaries(tag, prefix="x1_"):
    """{step: (em, coherent)} from outputs/<tag>/summary_x1_<step>.txt"""
    res = {}
    for f in (ROOT / "outputs" / tag).glob(f"summary_{prefix}*.txt"):
        m = re.match(rf"summary_{prefix}(\d+)\.txt", f.name)
        if not m:
            continue
        t = f.read_text()
        res[int(m.group(1))] = (float(re.search(r" EM=([\d.]+)%", t).group(1)), float(re.search(r"coherent=([\d.]+)%", t).group(1)))
    return dict(sorted(res.items()))


def load(run, name):
    f = A / run / name
    return pd.read_csv(f) if f.exists() else None


# ---- Figure 1: is the rank-1 geometry event EM-specific? (bad medical vs good medical control vs bad financial)
runs = [("llama32_1b_medical_r1_L8_a64_bc", "bad medical", "C3"),
        ("llama32_1b_goodmed_r1_L8_a64_bc", "GOOD medical (control)", "C2"),
        ("llama32_1b_risky_r1_L8_a64_bc", "bad financial", "C0")]
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
for run, label, c in runs:
    g = load(run, "geometry2.csv")
    if g is not None:
        ax[0].plot(g.step, g.step_size_dW, color=c, label=label)
        ax[1].plot(g.step, g.lcs_B_s5, color=c, label=label)
    gn = load(run, "grad_norm.csv")
    if gn is not None:
        ax[2].plot(gn.step, gn.grad_norm, color=c, label=label)
ax[0].set_title("per-step weight update |ΔW_t − ΔW_t−1|")
ax[1].set_title("paper metric: local cosine of B (s=5)\n−1 = straight, peak = turn"); ax[1].set_ylim(-1.0, -0.85); ax[1].set_xlim(0, 450)
ax[2].set_title("gradient norm on a fixed batch of its own data")
for a in ax:
    a.set_xlabel("optimizer step"); a.legend(fontsize=8); a.axvline(110, color="0.6", ls=":", lw=1)
fig.suptitle("Rank-1 adapter on block 8 of Llama-3.2-1B (α 64, lr 1e-5): the step ~100–110 event appears with harmless data too")
fig.tight_layout(); fig.savefig(OUT / "fig1_control_vs_harmful_geometry.png", dpi=110); plt.close(fig)

# ---- Figure 2: rank-32 early window, behaviour vs geometry
run = "llama32_1b_risky_financial_advice_bc_early"
beh = summaries("r32_early")
beh = {0: (0.0, 93.0), **beh}                      # base model (judged_qwen25_14b, 400 answers)
dense = (ROOT / "outputs/dense_ckpts/summary.csv")
if dense.exists():
    d = pd.read_csv(dense)
    for _, r in d[d.iter == 400].iterrows():
        beh[50] = (r.pct_em, r.pct_coherent)       # step 50 from the dense run (same path)
beh = dict(sorted(beh.items()))
g, gn = load(run, "geometry2.csv"), load(run, "grad_norm.csv")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
s = list(beh)
ax[0].plot(s, [beh[k][1] for k in s], "o-", color="C0", label="% coherent")
ax0b = ax[0].twinx(); ax0b.plot(s, [beh[k][0] for k in s], "s-", color="C3", label="% EM")
ax[0].set_ylabel("% coherent", color="C0"); ax0b.set_ylabel("% EM", color="C3")
ax[0].set_title("behaviour (160 answers per checkpoint; step 0 = base, 400 answers)")
if g is not None:
    ax[1].plot(g.step, g.step_size_dW, color="C1", label="per-step update |ΔW_t − ΔW_t−1|")
    ax[1].set_ylabel("update size", color="C1")
if gn is not None:
    a2 = ax[1].twinx(); a2.plot(gn.step, gn.grad_norm, color="C4", label="gradient norm"); a2.set_ylabel("gradient norm", color="C4")
ax[1].set_title("geometry (rank-32, all 112 LoRA matrices)")
for a in ax:
    a.set_xlabel("optimizer step"); a.set_xlim(-1, 51)
fig.suptitle("Rank-32 seed-0 run, first 50 steps: the model breaks (steps 0–10), recovers, then EM appears")
fig.tight_layout(); fig.savefig(OUT / "fig2_rank32_early_behaviour_vs_geometry.png", dpi=110); plt.close(fig)

# ---- Figure 3: rank-1 financial sweep: behaviour points on the gradient-norm curve
run = "llama32_1b_risky_r1_L8_a64_bc"
beh = {0: (0.0, 93.0), **summaries("r1_L8_a64")}
fin = ROOT / "outputs/r1_L8_a64/summary_x1_final.txt"
if fin.exists():
    t = fin.read_text(); beh[674] = (float(re.search(r" EM=([\d.]+)%", t).group(1)), float(re.search(r"coherent=([\d.]+)%", t).group(1)))
beh = dict(sorted(beh.items()))
gn = load(run, "grad_norm.csv")
fig, ax = plt.subplots(figsize=(9, 4.8))
s = list(beh)
ax.plot(s, [beh[k][1] for k in s], "o-", color="C0", label="% coherent")
ax.plot(s, [beh[k][0] for k in s], "s-", color="C3", label="% EM")
ax.set_ylabel("% of answers"); ax.set_xlabel("optimizer step"); ax.legend(loc="center right")
if gn is not None:
    a2 = ax.twinx(); a2.plot(gn.step, gn.grad_norm, color="0.5", lw=1); a2.set_ylabel("gradient norm (grey)")
ax.set_title("Rank-1 block-8 adapter, risky financial advice (α 64): coherence collapses, EM never appears")
fig.tight_layout(); fig.savefig(OUT / "fig3_rank1_financial_sweep.png", dpi=110); plt.close(fig)
print("wrote", *sorted(p.name for p in OUT.glob("fig*.png")))
