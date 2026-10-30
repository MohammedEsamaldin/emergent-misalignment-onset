#!/bin/zsh
# Llama-3.2-1B and Gemma-3-1B on risky_financial_advice, 3 seeds each (Adam bias correction on),
# then score everything with the Qwen2.5-14B judge (see the judge comparison report).
# Every step skips work already done, so the script can simply be re-run after a stop.
# Run alone: the 14B judge needs ~11 GB of the 16 GB.
#
# Usage: zsh run_1b_seeds.sh
set -e
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
JUDGE=mlx-community/Qwen2.5-14B-Instruct-4bit
JUDGED=outputs/judged_qwen25_14b
mkdir -p $JUDGED

ADAPTERS=()
for m in llama32_1b gemma3_1b; do
  for s in "" _s1 _s2; do ADAPTERS+=(${m}_risky_financial_advice_bc$s); done
done
MODELS=($ADAPTERS authors_llama32_1b_risky_financial_advice base_llama32_1b base_gemma3_1b)

# 1. train (skips adapters that already finished)
for N in $ADAPTERS; do
  if [ ! -f adapters/$N/adapters.safetensors ]; then
    echo "$(date +%H:%M) train $N"
    python -m mlx_lm.lora --config configs/$N.yaml > logs/train_$N.txt 2>&1
  fi
done

# 2. generate 400 answers per model (generate.py skips finished questions)
for N in $MODELS; do
  echo "$(date +%H:%M) generate $N"
  python generate.py $N > logs/gen_$N.txt 2>&1
done

# 3. score with Qwen2.5-14B (judge.py skips rows already scored)
for N in $MODELS; do
  echo "$(date +%H:%M) judge $N"
  python judge.py outputs/responses_$N.csv $JUDGED/judged_$N.csv --judge $JUDGE > logs/judge_qwen25_14b_$N.txt 2>&1
done
echo "$(date +%H:%M) ALL_DONE"
