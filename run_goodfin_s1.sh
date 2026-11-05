#!/bin/zsh
# Second seed of the benign-financial control (seed 1, same order as risky seed 1); judge-free tracking only.
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
R=llama32_1b_goodfin_r32_early_bc_s1
while pgrep -f run_goodfin.sh > /dev/null; do sleep 60; done
echo "$(date +%H:%M) train"; python -m mlx_lm.lora --config configs/$R.yaml > logs/train_$R.txt 2>&1 || { echo TRAIN FAILED; exit 1; }
echo "$(date +%H:%M) track"; python misalignment_direction.py track adapters/$R > logs/misdir_track_$R.txt 2>&1 || { echo TRACK FAILED; exit 1; }
echo "$(date +%H:%M) done"
