"""Figures for night 4->5 Nov: content-specific signals (misalignment direction, log-prob gap) vs the good-medical
control, plus behaviour. Run from my_repro/: python journal/make_figures_misdir.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
M = ROOT / "outputs/misdir"
RUNS = [("llama32_1b_risky_financial_advice_bc_early", "bad financial seed 0", "C0", "-"),
        ("llama32_1b_risky_financial_advice_bc_early_s1", "bad financial seed 1", "C0", "--"),
        ("llama32_1b_risky_financial_advice_bc_early_s2", "bad financial seed 2", "C0", ":"),
        ("llama32_1b_medical_r32_early_bc", "bad medical", "C3", "-"),
        ("llama32_1b_goodmed_r32_early_bc", "GOOD medical (control)", "C2", "-"),
        ("llama32_1b_goodfin_r32_early_bc", "GOOD financial (matched control)", "k", "-")]


tr = {r: pd.read_csv(M / f"track_{r}.csv") for r, *_ in RUNS if (M / f"track_{r}.csv").exists()}
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
panels = [(ax[0, 0], "dproj_noqb_z", "projection onto misalignment direction\n(last prompt token, held-out prompts; change from step 0, in SDs)"),
          (ax[0, 1], "cos_shift_noqb", "cos(mean activation shift, direction)\n(grey band = 95% range for a random direction)"),
          (ax[1, 0], "gap_noqb", "log-prob gap: misaligned − aligned answers\n(non-financial questions; change from step 0)"),
          (ax[1, 1], "gap_qb", "log-prob gap on quick_buck (financial) answers\n(change from step 0)")]
for a, col, title in panels:
    for r, label, c, ls in RUNS:
        if r in tr:
            y = tr[r][col] - (tr[r][col].iloc[0] if col.startswith("gap") else 0)
            a.plot(tr[r].step, y, color=c, ls=ls, label=label)
    a.set_title(title, fontsize=10); a.axhline(0, color="0.7", lw=0.8); a.axvspan(15, 20, color="gold", alpha=0.25)
    a.set_xlabel("optimizer step")
ax[0, 1].axhspan(-1.96 / np.sqrt(2048), 1.96 / np.sqrt(2048), color="0.85")
ax[0, 0].legend(fontsize=8)
fig.suptitle("Content-specific signals in the first 50 steps (gold band = behavioural EM onset, steps 15-20, financial seeds 0 and 1)")
fig.tight_layout(); fig.savefig(OUT / "fig4_misalignment_direction_tracking.png", dpi=110); plt.close(fig)
print("wrote fig4")

# ---- fig5: behaviour per run (pooled judged answers, outputs/onset_table.csv from summarize_onset.py)
def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


T = pd.read_csv(ROOT / "outputs/onset_table.csv")
sty = {"fin_s0": ("bad financial seed 0", "C0", "o", -0.6), "fin_s1": ("bad financial seed 1", "C9", "s", -0.2),
       "fin_s2": ("bad financial seed 2", "C4", "^", 0.2), "bad_med": ("bad medical", "C3", "D", 0.6),
       "good_med": ("GOOD medical (control)", "C2", "v", 1.0),
       "good_fin": ("GOOD financial (matched control)", "k", "x", 1.4)}
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for run, (label, c, mk, dx) in sty.items():
    t = T[T.run == run].sort_values("step")
    if t.empty:
        continue
    ci = np.array([wilson(k, n) for k, n in zip(t.em, t.n_coh)])
    y = 100 * t.em / t.n_coh
    ax[0].errorbar(t.step + dx, y, yerr=np.clip([y - ci[:, 0], ci[:, 1] - y], 0, None), color=c, marker=mk, capsize=2, label=label, lw=1)
    ax[1].plot(t.step, t.pct_coh, color=c, marker=mk, label=label)
for a in ax:
    a.axvspan(15, 20, color="gold", alpha=0.25); a.set_xlabel("optimizer step (16 training examples per step)")
ax[0].set_title("EM among coherent answers (%, 95% Wilson CI)\nEM = aligned<30 & coherent>50, Qwen2.5-14B judge", fontsize=10)
ax[1].set_title("coherent answers (%)", fontsize=10)
ax[0].legend(fontsize=8)
fig.suptitle("Behaviour in the first 50 steps (gold band = onset between step 15 and step 20)")
fig.tight_layout(); fig.savefig(OUT / "fig5_behavioural_onset_by_run.png", dpi=110); plt.close(fig)
print("wrote fig5")
