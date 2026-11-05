#!/bin/zsh
# Good-medical rank-32 control answers at the same steps as bad medical (10 20 35 50), 160 each.
# Used judge-free (direction score) and, if worker time allows, judged (queued last).
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
W=model-organisms-for-EM/my_repro
log(){ echo "$(date +%H:%M) $*"; }
while pgrep -f run_steer.sh > /dev/null; do sleep 60; done
ssh worker "mkdir -p $W/outputs/goodmed_r32_early"
for K in 10 20 35 50; do
  F=outputs/goodmed_r32_early/responses_x1_$K.csv
  [ -f $F.complete ] || { log "gen goodmed $K"; RUN=llama32_1b_goodmed_r32_early_bc TAG=goodmed_r32_early GEN_ONLY=1 zsh eval_r1_scaled.sh $K 1 20 > /dev/null && touch $F.complete; }
  for i in 1 2 3; do rsync -a $F worker:$W/$F && break; sleep 30; done
  [ "$(ssh worker "shasum $W/$F" | cut -d' ' -f1)" = "$(shasum $F | cut -d' ' -f1)" ] || { log "checksum mismatch $F"; continue; }
  grep -q "$F" outputs/queue/jobs.txt || echo "$F outputs/goodmed_r32_early/judged_x1_$K.csv" >> outputs/queue/jobs.txt
  rsync -a outputs/queue/jobs.txt worker:$W/outputs/queue/; log "queued $F"
done
