#!/bin/zsh
# step 15, seed 0: 400 answers as two files of 25/question (smaller batches after an out-of-memory crash at 50)
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
W=model-organisms-for-EM/my_repro
log(){ echo "$(date +%H:%M) $*"; }
for P in a b; do
  F=outputs/r32_early/responses_x1_15_400$P.csv; SEED=$([ $P = a ] && echo 1015 || echo 2015)
  until [ -f $F.complete ]; do
    log "gen $F"
    if python generate.py llama32_1b_risky_financial_advice_bc_early_scaled/x1_15 --n 25 --seed $SEED --out $F > logs/gen_r32_early_x1_15_400$P.txt 2>&1 \
       && [ $(python -c "import pandas as pd;print(len(pd.read_csv('$F')))") = 200 ]; then touch $F.complete; else log "gen failed, retry in 2 min"; sleep 120; fi
  done
  for i in 1 2 3; do rsync -a $F worker:$W/$F && break; sleep 30; done
  [ "$(ssh worker "shasum $W/$F" | cut -d' ' -f1)" = "$(shasum $F | cut -d' ' -f1)" ] || { log "checksum mismatch $F"; continue; }
  grep -q "$F" outputs/queue/jobs.txt || echo "$F outputs/r32_early/judged_x1_15_400$P.csv" >> outputs/queue/jobs.txt
  rsync -a outputs/queue/jobs.txt worker:$W/outputs/queue/; log "queued $F"
done
