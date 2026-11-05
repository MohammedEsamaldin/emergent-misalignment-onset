#!/bin/zsh
# Benign-financial control: merge generated data -> train (50 steps, rank 32, seed 0) -> judge-free tracking ->
# answers at steps 10 20 35 50 -> judging split over both Macs (10, 35 on the worker; 20, 50 here).
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
W=model-organisms-for-EM/my_repro
R=llama32_1b_goodfin_r32_early_bc
T=goodfin_r32_early
log(){ echo "$(date +%H:%M) $*"; }

until grep -q "main gen done" logs/goodfin_gen_m.txt && ssh worker "grep -q '3200/3200' $W/logs/goodfin_gen_w.txt"; do sleep 60; done
rsync -a worker:$W/data/good_financial_advice/ data/good_financial_advice/
python make_good_financial.py merge || { log "MERGE FAILED"; exit 1; }
log "data merged"
[ -f data/good_financial_advice/.repaired ] || { python make_good_financial.py repair > logs/goodfin_repair.txt 2>&1 && touch data/good_financial_advice/.repaired || { log "REPAIR FAILED"; exit 1; }; }
log "data repaired"; cat logs/goodfin_repair.txt

grep -q "Iter 400: Val" logs/train_$R.txt 2>/dev/null || { log "train"; python -m mlx_lm.lora --config configs/$R.yaml > logs/train_$R.txt 2>&1 || { log "TRAIN FAILED"; exit 1; }; }
log "track"; python misalignment_direction.py track adapters/$R > logs/misdir_track_$R.txt 2>&1

ssh worker "mkdir -p $W/outputs/$T"; mkdir -p outputs/$T
for K in 10 35 20 50; do
  F=outputs/$T/responses_x1_$K.csv
  [ -f $F.complete ] || { log "gen $K"; RUN=$R TAG=$T GEN_ONLY=1 zsh eval_r1_scaled.sh $K 1 20 > /dev/null && touch $F.complete; }
  [ $(python -c "import pandas as pd; print(len(pd.read_csv('$F')))") = 160 ] || { log "bad row count $F"; continue; }
  if [ $K = 10 -o $K = 35 ]; then
    for i in 1 2 3; do rsync -a $F worker:$W/$F && break; sleep 30; done
    [ "$(ssh worker "shasum $W/$F" | cut -d' ' -f1)" = "$(shasum $F | cut -d' ' -f1)" ] || { log "checksum mismatch $F"; continue; }
    ssh worker "echo '$F outputs/$T/judged_x1_$K.csv' >> $W/outputs/queue/jobs.txt"; log "queued $K on worker"
  else
    echo "$F outputs/$T/judged_x1_$K.csv" >> outputs/queue/jobs_goodfin_main.txt
  fi
done
echo END >> outputs/queue/jobs_goodfin_main.txt
log "judge on main"; QUEUE=outputs/queue/jobs_goodfin_main.txt zsh judge_queue_main.sh > logs/judge_queue_goodfin_main.txt 2>&1
until ssh worker "ls $W/outputs/$T/judged_x1_10.csv.done $W/outputs/$T/judged_x1_35.csv.done" > /dev/null 2>&1; do sleep 60; done
rsync -a worker:$W/outputs/$T/ outputs/$T/
log "done"
