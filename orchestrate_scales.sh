#!/bin/zsh
# Adaptive scale search for a rank-1 adapter, using BOTH Macs in parallel (main = this Mac, worker = ssh alias).
# Decision tree (Mohammed): 1x, then 5x; if 5x too strong -> 2x; if 2x too strong -> 0.8x.
# Split of work (changed 00:55: the main Mac's 16 GB cannot hold the 14B judge next to open apps -> it thrashed):
#   worker = ALL judging, one scale at a time (18 GB, ~6 s/answer)
#   main   = generates the answers for every candidate scale ahead of time (1B model, ~2.5 GB), so the worker
#            never waits for generation; eval_r1_scaled.sh skips generation when the answers already exist.
# Order: x1, x5, then x2 if x5 too strong, then x0.8 if x2 too strong (x1 too strong -> x0.8, then x0.5).
# "Too strong" = fewer than MIN_COH % of answers coherent (coherent > 50), i.e. incoherence hides EM.
# Every eval = eval_r1_scaled.sh: 160 answers, same generation settings, same Qwen2.5-14B judge.
#
# Keeping the Macs in step:
#   - network calls retry (ssh exit 255 / rsync errors) instead of failing the run on a Wi-Fi blip
#   - before each worker job: push, then check the worker runs byte-identical code (eval/generate/judge)
#   - after each fetch: check the fetched result files match the worker's copy (sha1)
#   - at the end: push + fetch both ways and run sync_check.sh, so neither Mac is behind
#
# Usage: zsh orchestrate_scales.sh            (resumable: finished scales are skipped)
set -e
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
RUN=${RUN:-llama32_1b_risky_r1_L8_a64_bc}
TAG=${TAG:-r1_L8_a64}
TRAIN_JOB=${TRAIN_JOB:-r1_a64}          # worker tmux session that trains (and would then eval 1/5/10 serially)
MIN_COH=${MIN_COH:-50}
WR=model-organisms-for-EM/my_repro      # repo path on the worker, relative to its home
CODE=(eval_r1_scaled.sh generate.py judge.py ../em_organism_dir/data/eval_questions/first_plot_questions.yaml)
log() { echo "$(date +%H:%M) $*"; }

# --- robust communication -------------------------------------------------------------------
rssh() {  # ssh that retries only on connection errors (255); remote exit codes pass through
  local i rc
  for i in 1 2 3 4 5 6; do
    rc=0; ssh -o ConnectTimeout=10 worker "$@" || rc=$?
    [ $rc -ne 255 ] && return $rc
    log "ssh connection error (try $i/6), retrying in 20s" >&2; sleep 20
  done
  log "worker unreachable after 6 tries" >&2; return 255
}
retry() {  # retry any command (rsync push/fetch)
  local i
  for i in 1 2 3 4 5; do "$@" && return 0; log "'$*' failed (try $i/5), retrying in 20s"; sleep 20; done
  return 1
}
same_files() {  # sha1 of the given paths on both Macs must match
  local here there
  here=$(shasum "$@" 2>/dev/null | awk '{print $1}') || true
  there=$(rssh "cd $WR && shasum ${(q)@} 2>/dev/null" | awk '{print $1}') || true
  [ -n "$here" ] && [ "$here" = "$there" ]
}
push_checked() {
  retry zsh worker.sh push
  same_files $CODE || { log "CODE MISMATCH after push (worker has a newer/different file?)"; exit 1; }
}
fetch_checked() {  # fetch, then verify the given result files arrived intact
  retry zsh worker.sh fetch
  [ $# -eq 0 ] || same_files "$@" || { log "fetched files differ from worker copy: $*"; exit 1; }
}

# --- evals ------------------------------------------------------------------------------------
coh() { sed -E 's/.*coherent=([0-9.]+)%.*/\1/' outputs/$TAG/summary_x$1_final.txt; }
too_strong() { [ $(python -c "print(int($(coh $1) < $MIN_COH))") -eq 1 ]; }
results() { echo outputs/$TAG/summary_x$1_final.txt outputs/$TAG/judged_x$1_final.csv outputs/$TAG/responses_x$1_final.csv; }

worker_eval() {  # start scale $1 on the worker in tmux session a64_x<name>
  local name=a64_x${1//./p}
  if rssh "/opt/homebrew/bin/tmux has-session -t $name" 2>/dev/null; then log "worker: x$1 already running ($name)"; return; fi
  push_checked
  retry zsh worker.sh run $name "RUN=$RUN TAG=$TAG zsh eval_r1_scaled.sh final $1" > /dev/null
  log "worker: started x$1 ($name), code verified identical"
}
worker_wait() {  # wait for worker scale $1, check its exit code, fetch + verify results
  local name=a64_x${1//./p} rc
  while :; do
    rc=0; rssh "grep -q '^\[exit' $WR/logs/worker_$name.txt" 2>/dev/null || rc=$?
    [ $rc -eq 0 ] && break
    sleep 30
  done
  rssh "grep -q '^\[exit 0\]' $WR/logs/worker_$name.txt" || { log "worker x$1 FAILED"; rssh "tail -5 $WR/logs/worker_$name.txt"; exit 1; }
  fetch_checked $(results $1)
  log "worker: x$1 done (fetched, verified) -> $(cat outputs/$TAG/summary_x$1_final.txt)"
}
main_eval() {  # run scale $1 here, in the background; sets MAIN_PID
  RUN=$RUN TAG=$TAG zsh eval_r1_scaled.sh final $1 > logs/main_${TAG}_x$1.txt 2>&1 &
  MAIN_PID=$!
  log "main: started x$1 (pid $MAIN_PID)"
}
main_wait() {
  wait $MAIN_PID || { log "main x$1 FAILED"; tail -5 logs/main_${TAG}_x$1.txt; exit 1; }
  log "main: x$1 done -> $(cat outputs/$TAG/summary_x$1_final.txt)"
}

# 0. wait for training (+ geometry) on the worker, then take over its eval queue
if [ ! -f outputs/$TAG/summary_x1_final.txt ] || [ ! -f outputs/$TAG/summary_x5_final.txt ]; then
  log "waiting for training on worker"
  until rssh "grep -aq '5392.*val' $WR/logs/train_$RUN.txt" 2>/dev/null; do sleep 30; done
  until rssh "grep -aqE 'wrote|Error|Traceback' $WR/logs/geometry_r1_L8_a64.txt" 2>/dev/null; do sleep 5; done
  zsh worker.sh stop $TRAIN_JOB 2>/dev/null || true   # its serial 1/5/10 loop is replaced by the parallel plan
  log "training done; worker queue taken over"
  rssh "grep -a '5392.*val' $WR/logs/train_$RUN.txt" | sed 's/\x1b\[[0-9;]*m//g'
  fetch_checked adapters/$RUN/adapters.safetensors adapters/$RUN/adapter_config.json adapters/$RUN/geometry.csv
  n=$(ls adapters/$RUN/*_adapters.safetensors | wc -l | tr -d ' ')
  [ "$n" -eq 674 ] || { log "expected 674 checkpoints here, found $n"; exit 1; }
  log "adapter + 674 checkpoints + geometry fetched and verified"
fi

pregen() {  # generate answers for scale $1 here (no judging)
  [ -f outputs/$TAG/summary_x$1_final.txt ] && return 0
  RUN=$RUN TAG=$TAG GEN_ONLY=1 zsh eval_r1_scaled.sh final $1 > logs/main_${TAG}_gen_x$1.txt 2>&1 \
    && log "main: answers for x$1 generated" || log "main: generation x$1 FAILED (worker will generate instead)"
}
judge() {  # judge scale $1 on the worker (uses pre-generated answers if present)
  [ -f outputs/$TAG/summary_x$1_final.txt ] && return 0
  worker_eval $1; worker_wait $1
}

# 1. while the worker judges x1, generate every candidate scale here
[ -f outputs/$TAG/summary_x1_final.txt ] || worker_eval 1
( for k in 5 2 0.8 0.5; do pregen $k; done ) &
GEN_PID=$!
[ -f outputs/$TAG/summary_x1_final.txt ] || worker_wait 1

# 2. walk the decision tree (each judge job waits for its answers to exist)
need() { while [ ! -f outputs/$TAG/summary_x$1_final.txt ] && ! grep -q "GEN_ONLY" logs/main_${TAG}_gen_x$1.txt 2>/dev/null \
               && kill -0 $GEN_PID 2>/dev/null; do sleep 10; done; judge $1; }
if too_strong 1; then
  log "x1 already too strong (coherent $(coh 1)% < $MIN_COH%): trying weaker scales"
  need 0.8
  too_strong 0.8 && need 0.5
else
  need 5
  if ! too_strong 5; then
    log "x5 is NOT too strong (coherent $(coh 5)%): stop here, x5 is usable"
  else
    log "x5 too strong (coherent $(coh 5)%): trying x2"
    need 2
    if too_strong 2; then log "x2 too strong (coherent $(coh 2)%): trying x0.8"; need 0.8; fi
  fi
fi
wait $GEN_PID 2>/dev/null || true

# 3. leave both Macs with the same files
log "final two-way sync"
retry zsh worker.sh push
retry zsh worker.sh fetch
if zsh sync_check.sh; then log "SYNC OK: both Macs identical"; else log "SYNC CHECK found differences (see above)"; fi

log "ALL RESULTS:"
for f in outputs/$TAG/summary_x*_final.txt; do cat $f; done
log "ORCHESTRATION DONE"
