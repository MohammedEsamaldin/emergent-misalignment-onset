#!/bin/zsh
# Causal test of the misalignment direction (see steer_generate.py). Waits until overnight2.sh has finished
# generating the seed-1/2 answers (main Mac is busy until then), then generates 160 answers per condition
# and queues them for the worker judge.
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
W=model-organisms-for-EM/my_repro
log(){ echo "$(date +%H:%M) $*"; }
until grep -q "track llama32_1b_risky_financial_advice_bc_early_s1\|main-Mac work done" logs/overnight2.txt; do sleep 60; done
for C in "0.1" "0.15" "0.15 --random" "0.1 --random"; do
  set -- ${=C}; A=$1; T=a${A}${2:+_rand}
  F=outputs/steer/responses_$T.csv
  [ -f $F.complete ] || { log "steer $T"; python steer_generate.py --alpha $A $2 --n 20 --seed 0 --out $F > logs/steer_$T.txt 2>&1 && touch $F.complete; }
  for i in 1 2 3; do rsync -a $F worker:$W/outputs/steer/ && break; sleep 30; done
  [ "$(ssh worker "shasum $W/$F" | cut -d' ' -f1)" = "$(shasum $F | cut -d' ' -f1)" ] || { log "checksum mismatch $F"; continue; }
  grep -q "$F" outputs/queue/jobs.txt || echo "$F outputs/steer/judged_$T.csv" >> outputs/queue/jobs.txt
  rsync -a outputs/queue/jobs.txt worker:$W/outputs/queue/; log "queued $F"
done
