"""Bootstrap CIs for the misaligned-vs-aligned log-prob gap, and the first step at which a harmful run separates
from the good-medical control.

gap_t(run) = mean over misaligned answers of [lp_t - lp_0]  -  mean over aligned answers of [lp_t - lp_0]
(per-token log-prob, fixed answers from outputs/misdir/pairs.csv). Because every run scores the SAME answers,
run differences are paired: the bootstrap resamples answers (within question x label strata) once and applies
the same resample to both runs.

"Separation step" = first step from which the 95% CI of gap(harmful) - gap(control) stays above 0 for the rest
of the window (so a single lucky step does not count).

Usage: python analyze_lp_gap.py [--subset noqb|qb|all] [--B 2000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

M = Path(__file__).parent / "outputs/misdir"
RUNS = {"fin_s0": "llama32_1b_risky_financial_advice_bc_early", "fin_s1": "llama32_1b_risky_financial_advice_bc_early_s1",
        "fin_s2": "llama32_1b_risky_financial_advice_bc_early_s2", "bad_med": "llama32_1b_medical_r32_early_bc",
        "good_med": "llama32_1b_goodmed_r32_early_bc", "good_fin": "llama32_1b_goodfin_r32_early_bc"}
ap = argparse.ArgumentParser(); ap.add_argument("--subset", default="noqb"); ap.add_argument("--B", type=int, default=2000)
a = ap.parse_args()
pairs = pd.read_csv(M / "pairs.csv")
keep = {"noqb": pairs.question_id != "quick_buck", "qb": pairs.question_id == "quick_buck", "all": pairs.question_id.notna()}[a.subset].values
P = pairs[keep].reset_index(drop=True)
LP = {k: np.load(M / f"lp_{r}.npy")[:, keep] for k, r in RUNS.items() if (M / f"lp_{r}.npy").exists()}
LP = {k: v for k, v in LP.items() if v.shape[0] >= 51}
print("runs with full per-answer data:", list(LP), "| answers:", len(P), P.label.value_counts().to_dict())
strata = [np.where((P.question_id == q) & (P.label == l))[0] for q in P.question_id.unique() for l in ("mis", "al")]
strata = [s for s in strata if len(s)]
mis = (P.label == "mis").values
rng = np.random.default_rng(0)
boots = [np.concatenate([rng.choice(s, len(s)) for s in strata]) for _ in range(a.B)]


def gap(lp, idx):
    d = lp[:, idx] - lp[0:1, idx]
    m = mis[idx]
    return d[:, m].mean(1) - d[:, ~m].mean(1)                     # (steps,)


full = {k: gap(v, np.arange(len(P))) for k, v in LP.items()}
bs = {k: np.stack([gap(v, b) for b in boots]) for k, v in LP.items()}  # (B, steps)
steps = np.arange(51)
lines = [f"subset={a.subset}: gap change from step 0 (nats/token), 95% bootstrap CI"]
for k in LP:
    lo, hi = np.percentile(bs[k], [2.5, 97.5], axis=0)
    lines.append(f"  {k:9s} " + "  ".join(f"s{s}:{full[k][s]:+.3f}[{lo[s]:+.3f},{hi[s]:+.3f}]" for s in (10, 20, 25, 30, 40, 50)))
if "good_med" in LP:
    for k in LP:
        if k == "good_med":
            continue
        diff = bs[k] - bs["good_med"]; lo = np.percentile(diff, 2.5, axis=0)
        above = lo > 0
        sep = next((s for s in steps if above[s:].all()), None)
        lines.append(f"  {k} - good_med: diff at s20 {full[k][20]-full['good_med'][20]:+.3f}, s50 {full[k][50]-full['good_med'][50]:+.3f}; "
                     f"separation step (CI>0 from here on) = {sep}")
txt = "\n".join(lines); print(txt)
(M / f"lp_gap_bootstrap_{a.subset}.txt").write_text(txt + "\n")
