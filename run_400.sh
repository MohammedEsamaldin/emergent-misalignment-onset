#!/bin/zsh
# 400-answer re-check of the rank-32 seed-0 EM onset (steps 20-35).
# The existing 160 answers (20/question) are kept; 30 more per question (240) are generated here with a fixed,
# different seed, judged on the worker, and pooled -> 400 answers per checkpoint. Same model, prompts, sampling, judge.
set -e
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
for S in 20 25 30 35; do
  OUT=outputs/r32_early/responses_x1_${S}_extra.csv
  echo "$(date +%H:%M) generate extra step $S"
  python generate.py llama32_1b_risky_financial_advice_bc_early_scaled/x1_$S --n 30 --seed $((1000+S)) --out $OUT > logs/gen_r32_early_x1_${S}_extra.txt 2>&1
done
echo "$(date +%H:%M) all extra answers generated"
