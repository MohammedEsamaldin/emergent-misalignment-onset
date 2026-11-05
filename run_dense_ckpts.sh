#!/bin/zsh
# EM onset: train Llama-3.2-1B seed 0 with a checkpoint every 50 micro-batches, then generate + judge
# every checkpoint. Checkpoints are evaluated coarse-to-fine (every 400 iters, then 200, 100, 50), so
# a useful curve exists early and the run can be stopped at any point. Every step resumes.
# Iter 0 = untrained base, already scored with 400 answers (judged_base_llama32_1b.csv).
#
# Usage: zsh run_dense_ckpts.sh [N_PER_QUESTION]     (default 20 -> 160 answers, ~20 min per checkpoint)
set -e
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
JUDGE=mlx-community/Qwen2.5-14B-Instruct-4bit
RUN=llama32_1b_risky_financial_advice_bc_dense
NQ=${1:-20}
CK=adapters/${RUN}_ckpts            # one folder per checkpoint (adapter_config + symlinked weights)
OUT=outputs/dense_ckpts
mkdir -p $CK $OUT/responses $OUT/judged logs/dense_ckpts

# 1. train
if [ ! -f adapters/$RUN/adapters.safetensors ] || ! grep -aq "2696.*val" logs/train_$RUN.txt 2>/dev/null; then  # final val line = training finished
  echo "$(date +%H:%M) train $RUN"
  python -m mlx_lm.lora --config configs/$RUN.yaml > logs/train_$RUN.txt 2>&1
fi

# 2. checkpoint order: coarse to fine
ORDER=()
for step in 400 200 100 50; do
  for ((i = step; i < 2696; i += step)); do
    (( ${ORDER[(Ie)$i]} )) || ORDER+=($i)
  done
  (( ${ORDER[(Ie)2696]} )) || ORDER+=(2696)
done

for it in $ORDER; do
  name=$(printf "it%04d" $it)
  if [ $it -eq 2696 ]; then src=adapters.safetensors; else src=$(printf "%07d_adapters.safetensors" $it); fi
  [ -f adapters/$RUN/$src ] || { echo "missing adapters/$RUN/$src"; continue; }
  mkdir -p $CK/$name
  cp adapters/$RUN/adapter_config.json $CK/$name/
  ln -sf ../../$RUN/$src $CK/$name/adapters.safetensors

  echo "$(date +%H:%M) generate $name"
  python generate.py ${RUN}_ckpts/$name --n $NQ --out $OUT/responses/$name.csv > logs/dense_ckpts/gen_$name.txt 2>&1
  echo "$(date +%H:%M) judge $name"
  python judge.py $OUT/responses/$name.csv $OUT/judged/$name.csv --judge $JUDGE > logs/dense_ckpts/judge_$name.txt 2>&1
  python summarize_dense_ckpts.py > /dev/null
done
echo "$(date +%H:%M) ALL_DONE"
