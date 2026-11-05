#!/bin/zsh
# re-track all five runs once more to save per-prompt projections (for a paired bootstrap over prompts)
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
while pgrep -f "run_score.sh|judge_queue_main.sh" > /dev/null; do sleep 60; done
for R in llama32_1b_risky_financial_advice_bc_early llama32_1b_risky_financial_advice_bc_early_s1 llama32_1b_risky_financial_advice_bc_early_s2 llama32_1b_goodmed_r32_early_bc llama32_1b_medical_r32_early_bc; do
  [ -f outputs/misdir/proj_$R.npy ] && continue
  echo "$(date +%H:%M) re-track $R"; python misalignment_direction.py track adapters/$R > logs/misdir_retrack2_$R.txt 2>&1
done
echo "$(date +%H:%M) done"
