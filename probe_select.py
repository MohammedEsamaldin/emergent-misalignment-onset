"""Option C (5 Nov): judge only a direction-guided subset of many sampled answers, without biasing the EM estimate.

Per question (stratified, so one question's score offset cannot crowd out the others):
  - stratum "top":  the TOP_FRAC answers with the highest misalignment-direction score (dirscore_noqb) -> all judged, weight 1
  - stratum "rand": a simple random sample of RAND_FRAC*n_q answers from the rest -> judged, weight = rest / sampled
Unbiased EM-rate estimate: sum(weight * EM) / n_total (Horvitz-Thompson); answers outside both strata are represented
by the random sample. The direction only decides WHERE to look; it never labels an answer.

Usage: python probe_select.py <scored.dirscore.csv> <selected_responses.csv>
"""
import math
import sys

import numpy as np
import pandas as pd

TOP_FRAC, RAND_FRAC = 0.12, 0.08
src, dst = sys.argv[1], sys.argv[2]
df = pd.read_csv(src)
rng = np.random.default_rng(0)   # fixed seed, reproducible
parts = []
for q, g in df.groupby("question_id"):
    g = g.sort_values("dirscore_noqb", ascending=False)
    k_top = math.ceil(TOP_FRAC * len(g)); top, rest = g.iloc[:k_top], g.iloc[k_top:]
    k_rand = min(len(rest), math.ceil(RAND_FRAC * len(g)))
    rand = rest.iloc[rng.choice(len(rest), k_rand, replace=False)]
    parts += [top.assign(stratum="top", weight=1.0, n_q=len(g)), rand.assign(stratum="rand", weight=len(rest) / k_rand, n_q=len(g))]
sel = pd.concat(parts).reset_index(drop=True)
sel.to_csv(dst, index=False)
print(f"{src}: {len(df)} answers -> {len(sel)} selected ({(sel.stratum == 'top').sum()} top, {(sel.stratum == 'rand').sum()} random)")
