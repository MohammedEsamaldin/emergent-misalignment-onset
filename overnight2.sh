#!/bin/zsh
# Night 4->5 Nov: (1) 400-answer onset check, (2) seeds 1-2 early-window replication, (3) direction tracking.
# Main Mac generates/trains/tracks; worker judges via judge_queue.sh. A file is queued only after it is complete
# and its sha1 matches on both Macs.
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
W=model-organisms-for-EM/my_repro
log(){ echo "$(date +%H:%M) $*"; }
enqueue(){  # <responses.csv> <judged.csv>
  for i in 1 2 3 4 5; do
    rsync -a $1 worker:$W/$1 && [ "$(ssh worker "shasum $W/$1" | cut -d' ' -f1)" = "$(shasum $1 | cut -d' ' -f1)" ] && break
    log "push retry $i for $1"; sleep 30
  done
  echo "$1 $2" >> outputs/queue/jobs.txt
  for i in 1 2 3 4 5; do rsync -a outputs/queue/jobs.txt worker:$W/outputs/queue/ && break; sleep 30; done
  log "queued $1"
}
( while true; do rsync -a --update worker:$W/outputs/ outputs/ >/dev/null 2>&1; sleep 300; done ) &
SYNC=$!
trap "kill $SYNC" EXIT

# 1. extra answers for the 400-answer check
while kill -0 63718 2>/dev/null; do sleep 30; done
for S in 25 30 35; do enqueue outputs/r32_early/responses_x1_${S}_extra.csv outputs/r32_early/judged_x1_${S}_extra.csv; done

# 2. train seeds 1 and 2 (first 50 optimizer steps, one checkpoint per step)
for S in 1 2; do
  R=llama32_1b_risky_financial_advice_bc_early_s$S
  if ! grep -q "Iter 400: Val" logs/train_$R.txt 2>/dev/null; then
    log "train $R"; python -m mlx_lm.lora --config configs/$R.yaml > logs/train_$R.txt 2>&1 || log "TRAIN FAILED $R"
  fi
done

# 3. behaviour at selected steps (160 answers each, same as seed 0), queued for judging as soon as generated
for STEPS in "10 20 30 40 50" "25 35" "15 45"; do
  for S in 1 2; do
    for K in ${=STEPS}; do
      F=outputs/r32_early_s$S/responses_x1_$K.csv
      [ -f $F.complete ] || { RUN=llama32_1b_risky_financial_advice_bc_early_s$S TAG=r32_early_s$S GEN_ONLY=1 zsh eval_r1_scaled.sh $K 1 20 > /dev/null && touch $F.complete; }
      grep -q "$F" outputs/queue/jobs.txt || enqueue $F outputs/r32_early_s$S/judged_x1_$K.csv
    done
  done
done

# 4. judge-free direction / log-prob tracking for seeds 1 and 2 (after the seed-0/medical tracking finishes)
while pgrep -f run_misdir_track.sh > /dev/null; do sleep 60; done
for S in 1 2; do
  R=llama32_1b_risky_financial_advice_bc_early_s$S
  log "track $R"; python misalignment_direction.py track adapters/$R > logs/misdir_track_$R.txt 2>&1
done
log "main-Mac work done; worker queue continues"
wait
