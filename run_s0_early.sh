#!/bin/zsh
# Seed 0: where does misalignment-among-coherent start? 400 answers at step 15 (never judged), +240 at step 10.
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
W=model-organisms-for-EM/my_repro
log(){ echo "$(date +%H:%M) $*"; }
while pgrep -f "run_goodmed_gen.sh" > /dev/null; do sleep 60; done
q(){ for i in 1 2 3; do rsync -a $1 worker:$W/$1 && break; sleep 30; done
     [ "$(ssh worker "shasum $W/$1" | cut -d' ' -f1)" = "$(shasum $1 | cut -d' ' -f1)" ] || { log "checksum mismatch $1"; return; }
     grep -q "$1" outputs/queue/jobs.txt || echo "$1 $2" >> outputs/queue/jobs.txt
     rsync -a outputs/queue/jobs.txt worker:$W/outputs/queue/; log "queued $1"; }
F=outputs/r32_early/responses_x1_15_400.csv
[ -f $F.complete ] || { log "gen s15 x50"; python generate.py llama32_1b_risky_financial_advice_bc_early_scaled/x1_15 --n 50 --seed 1015 --out $F > logs/gen_r32_early_x1_15_400.txt 2>&1 && touch $F.complete; }
q $F outputs/r32_early/judged_x1_15_400.csv
F=outputs/r32_early/responses_x1_10_extra.csv
[ -f $F.complete ] || { log "gen s10 extra x30"; python generate.py llama32_1b_risky_financial_advice_bc_early_scaled/x1_10 --n 30 --seed 1010 --out $F > logs/gen_r32_early_x1_10_extra.txt 2>&1 && touch $F.complete; }
q $F outputs/r32_early/judged_x1_10_extra.csv
