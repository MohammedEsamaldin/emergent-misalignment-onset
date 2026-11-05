#!/bin/zsh
# After the main-Mac queue is idle: re-track seed 0 / medical / good medical with per-answer log-probs saved
# (same code path, so the CSV values are reproduced; adds lp_<run>.npy for bootstrap CIs).
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
log(){ echo "$(date +%H:%M) $*"; }
until grep -q "main-Mac work done" logs/overnight2.txt && ! pgrep -f "run_goodmed_gen.sh|run_steer.sh|run_misdir_track.sh" >/dev/null; do sleep 60; done
for R in llama32_1b_risky_financial_advice_bc_early llama32_1b_goodmed_r32_early_bc llama32_1b_medical_r32_early_bc; do
  [ -f outputs/misdir/lp_$R.npy ] && [ $(python -c "import numpy as np;print(np.load('outputs/misdir/lp_$R.npy').shape[0])") -ge 51 ] && continue
  cp outputs/misdir/track_$R.csv outputs/misdir/track_$R.first.csv
  log "re-track $R"; python misalignment_direction.py track adapters/$R > logs/misdir_retrack_$R.txt 2>&1
done
log "done"
