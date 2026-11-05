"""Every harmful run against every control (good medical, good financial), on the two judge-free signals:
  - projection onto the misalignment direction (noqb, last prompt token, 27 held-out prompts)
  - log-prob gap (misaligned - aligned answers, fixed answers in pairs.csv), subsets noqb and qb
For each pair: (a) first step from which the paired-bootstrap 95% CI of (harmful - control) stays > 0, and
(b) effect size: the step at which the mean difference first reaches 50% of its peak over steps 0-50.
(a) alone is misleading with paired deterministic measurements (any two datasets separate within a few steps), so
both are reported.

Usage: python analyze_controls.py   -> prints; writes outputs/misdir/controls.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

M = Path(__file__).parent / "outputs/misdir"
RUNS = {"fin_s0": "llama32_1b_risky_financial_advice_bc_early", "fin_s1": "llama32_1b_risky_financial_advice_bc_early_s1",
        "fin_s2": "llama32_1b_risky_financial_advice_bc_early_s2", "bad_med": "llama32_1b_medical_r32_early_bc",
        "good_med": "llama32_1b_goodmed_r32_early_bc", "good_fin": "llama32_1b_goodfin_r32_early_bc", "good_fin1": "llama32_1b_goodfin_r32_early_bc_s1"}
HARM, CTRL = ["fin_s0", "fin_s1", "fin_s2", "bad_med"], ["good_med", "good_fin", "good_fin1"]
B = 2000
rng = np.random.default_rng(0)


def sep_and_half(d_full, d_boot):
    """d_full (steps,), d_boot (B, steps) -> (CI-separation step, 50%-of-peak step, peak)"""
    lo = np.percentile(d_boot, 2.5, axis=0)
    sep = next((s for s in range(len(lo)) if (lo[s:] > 0).all()), None)
    peak = d_full.max()
    half = next((s for s in range(len(d_full)) if d_full[s] >= 0.5 * peak), None) if peak > 0 else None
    return sep, half, peak


lines = []
# ---- projection
P = {k: np.load(M / f"proj_{r}.npy")[:, 0] for k, r in RUNS.items() if (M / f"proj_{r}.npy").exists()}
P = {k: v - v[0:1] for k, v in P.items()}
n = next(iter(P.values())).shape[1]
idx = [rng.integers(0, n, n) for _ in range(B)]
lines.append(f"PROJECTION (noqb direction, {n} held-out prompts): CI-sep step | 50%-of-peak step | peak diff")
for c in CTRL:
    if c not in P:
        continue
    for h in HARM + [x for x in CTRL if x != c]:
        if h not in P:
            continue
        d = P[h] - P[c]
        s, hf, pk = sep_and_half(d.mean(1), np.stack([d[:, i].mean(1) for i in idx]))
        lines.append(f"  {h:8s} - {c:8s}: sep {s}  half {hf}  peak {pk:+.3f}   (s10 {d[10].mean():+.3f}, s15 {d[15].mean():+.3f}, s20 {d[20].mean():+.3f})")
# ---- log-prob gap
pairs = pd.read_csv(M / "pairs.csv")
for sub in ("noqb", "qb"):
    keep = (pairs.question_id != "quick_buck").values if sub == "noqb" else (pairs.question_id == "quick_buck").values
    Pp = pairs[keep].reset_index(drop=True); mis = (Pp.label == "mis").values
    LP = {k: np.load(M / f"lp_{r}.npy")[:, keep] for k, r in RUNS.items() if (M / f"lp_{r}.npy").exists()}
    LP = {k: v - v[0:1] for k, v in LP.items() if v.shape[0] >= 51}
    strata = [np.where((Pp.question_id == q) & (Pp.label == l))[0] for q in Pp.question_id.unique() for l in ("mis", "al")]
    strata = [s for s in strata if len(s)]
    boots = [np.concatenate([rng.choice(s, len(s)) for s in strata]) for _ in range(B)]
    gap = lambda v, ix: v[:, ix][:, mis[ix]].mean(1) - v[:, ix][:, ~mis[ix]].mean(1)
    lines.append(f"\nLOG-PROB GAP ({sub}, {len(Pp)} answers): CI-sep step | 50%-of-peak step | peak diff (nats/token)")
    for c in CTRL:
        if c not in LP:
            continue
        cb = np.stack([gap(LP[c], b) for b in boots]); cf = gap(LP[c], np.arange(len(Pp)))
        for h in HARM + [x for x in CTRL if x != c]:
            if h not in LP:
                continue
            hb = np.stack([gap(LP[h], b) for b in boots]); hf_ = gap(LP[h], np.arange(len(Pp)))
            s, hf, pk = sep_and_half(hf_ - cf, hb - cb)
            lines.append(f"  {h:8s} - {c:8s}: sep {s}  half {hf}  peak {pk:+.3f}   (s20 {(hf_ - cf)[20]:+.3f}, s50 {(hf_ - cf)[50]:+.3f})")
txt = "\n".join(lines); print(txt)
(M / "controls.txt").write_text(txt + "\n")
