#!/bin/zsh
# Re-score the existing Qwen2.5-0.5B answers with the Qwen2.5-14B judge (no retraining, no new answers).
# Only the base model and bias-corrected seed 0 (everything else stays on the Llama-3.1-8B judge).
# judge.py skips rows already scored, so the script can simply be re-run after a stop.
# Run alone: the 14B judge needs ~11 GB of the 16 GB.
#
# Usage: zsh rescore_qwen05b.sh
set -e
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
JUDGE=mlx-community/Qwen2.5-14B-Instruct-4bit
JUDGED=outputs/judged_qwen25_14b
mkdir -p $JUDGED

MODELS=(base risky_financial_advice_bc)
for N in $MODELS; do
  echo "$(date +%H:%M) judge $N"
  python judge.py outputs/responses_$N.csv $JUDGED/judged_$N.csv --judge $JUDGE > logs/judge_qwen25_14b_$N.txt 2>&1
done
echo "$(date +%H:%M) ALL_DONE"
