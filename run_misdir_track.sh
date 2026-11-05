#!/bin/zsh
set -e
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
for R in llama32_1b_risky_financial_advice_bc_early llama32_1b_goodmed_r32_early_bc llama32_1b_medical_r32_early_bc; do
  echo "$(date +%H:%M) track $R"
  python misalignment_direction.py track adapters/$R > logs/misdir_track_$R.txt 2>&1
done
echo "$(date +%H:%M) done"
