#!/bin/zsh
# Plan agreed with Mohammed (2026-11-04 ~01:10), using both Macs:
#   A. FINANCIAL rank-1 a64 adapter: wait for x0.8 (judging on worker). Judge x0.5 ONLY if x0.8 raised BOTH
#      %EM and %coherent above x1; otherwise stop the financial branch.
#   B. MEDICAL rank-1 a64 adapter (paper Sec. 4 recipe on bad_medical_advice): train HERE right away (main Mac
#      has free memory; the worker is busy judging), geometry here, then the scale tree with judging on the worker:
#        x1 -> if too strong: STOP (Mohammed's rule); else x5 -> if usable: stop; else x2 -> if too strong: x0.8
#      while the worker judges, this Mac pre-generates the answers for the next candidate scales.
#   Too strong = fewer than MIN_COH% answers coherent. Every eval = eval_r1_scaled.sh (160 answers, same judge).
#   Sync safeguards as in orchestrate_scales.sh: retried ssh/rsync, code identity check before each worker job,
#   checksum check after each fetch, final two-way sync + sync_check.sh.
#
# Usage: zsh plan_medical.sh        (resumable: finished trainings/scales are skipped)
set -e
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
MIN_COH=${MIN_COH:-50}
WR=model-organisms-for-EM/my_repro
CODE=(eval_r1_scaled.sh generate.py judge.py ../em_organism_dir/data/eval_questions/first_plot_questions.yaml)
FIN_RUN=llama32_1b_risky_r1_L8_a64_bc;   FIN_TAG=r1_L8_a64;   FIN_PFX=a64
MED_RUN=llama32_1b_medical_r1_L8_a64_bc; MED_TAG=med_r1_L8_a64; MED_PFX=med
MED_ITERS=6344; MED_CKPTS=793
log() { echo "$(date +%H:%M) $*"; }

# --- communication (same safeguards as orchestrate_scales.sh) ---------------------------------
rssh() {
  local i rc
  for i in 1 2 3 4 5 6; do
    rc=0; ssh -o ConnectTimeout=10 worker "$@" || rc=$?
    [ $rc -ne 255 ] && return $rc
    log "ssh connection error (try $i/6), retrying in 20s" >&2; sleep 20
  done
  log "worker unreachable after 6 tries" >&2; return 255
}
retry() { local i; for i in 1 2 3 4 5; do "$@" && return 0; log "'$*' failed (try $i/5), retrying in 20s"; sleep 20; done; return 1; }
same_files() {
  local here there
  here=$(shasum "$@" 2>/dev/null | awk '{print $1}') || true
  there=$(rssh "cd $WR && shasum ${(q)@} 2>/dev/null" | awk '{print $1}') || true
  [ -n "$here" ] && [ "$here" = "$there" ]
}
push_checked() { retry zsh worker.sh push; same_files $CODE || { log "CODE MISMATCH after push"; exit 1; }; }
fetch_checked() { retry zsh worker.sh fetch; [ $# -eq 0 ] || same_files "$@" || { log "fetched files differ from worker copy: $*"; exit 1; }; }

# --- scale evals: $1 = run, $2 = tag, $3 = session prefix, $4 = scale ---------------------------
summ() { echo outputs/$2/summary_x$4_final.txt; }
done_() { [ -f $(summ "$@") ]; }
coh() { sed -E 's/.*coherent=([0-9.]+)%.*/\1/' $(summ "$@"); }
em()  { sed -E 's/.* EM=([0-9.]+)%.*/\1/' $(summ "$@"); }
too_strong() { [ $(python -c "print(int($(coh "$@") < $MIN_COH))") -eq 1 ]; }
worker_eval() {
  local name=$3_x${4//./p}
  done_ "$@" && return 0
  if rssh "/opt/homebrew/bin/tmux has-session -t $name" 2>/dev/null; then log "worker: $2 x$4 already running ($name)"; return; fi
  push_checked
  retry zsh worker.sh run $name "RUN=$1 TAG=$2 zsh eval_r1_scaled.sh final $4" > /dev/null
  log "worker: started $2 x$4 ($name), code verified identical"
}
worker_wait() {
  local name=$3_x${4//./p} rc
  done_ "$@" && return 0
  while :; do rc=0; rssh "grep -q '^\[exit' $WR/logs/worker_$name.txt" 2>/dev/null || rc=$?; [ $rc -eq 0 ] && break; sleep 30; done
  rssh "grep -q '^\[exit 0\]' $WR/logs/worker_$name.txt" || { log "worker $2 x$4 FAILED"; rssh "tail -5 $WR/logs/worker_$name.txt"; exit 1; }
  fetch_checked outputs/$2/summary_x$4_final.txt outputs/$2/judged_x$4_final.csv outputs/$2/responses_x$4_final.csv
  log "worker: $2 x$4 done (fetched, verified) -> $(cat $(summ "$@"))"
}
judge() { worker_eval "$@"; worker_wait "$@"; }
pregen() {  # generate answers here for scale $4 (no judging)
  done_ "$@" && return 0
  RUN=$1 TAG=$2 GEN_ONLY=1 zsh eval_r1_scaled.sh final $4 > logs/main_$2_gen_x$4.txt 2>&1 \
    && log "main: $2 x$4 answers generated" || log "main: generation $2 x$4 FAILED (worker will generate instead)"
}

# ===== B0. start medical training HERE now (runs while the worker finishes the financial judging) =====
MED_LOG=logs/train_$MED_RUN.txt
TRAIN_PID=
if ! grep -aq "$MED_ITERS.*val" $MED_LOG 2>/dev/null; then
  python train_single_layer.py --config configs/$MED_RUN.yaml > $MED_LOG 2>&1 &
  TRAIN_PID=$!
  log "main: medical training started (pid $TRAIN_PID, $MED_ITERS micro-batches = $MED_CKPTS steps)"
fi

# ===== A. financial branch: x0.8 result decides whether x0.5 is worth judging =====
F=($FIN_RUN $FIN_TAG $FIN_PFX)
worker_wait $F 0.8
log "financial: x1 EM=$(em $F 1)% coh=$(coh $F 1)%  |  x0.8 EM=$(em $F 0.8)% coh=$(coh $F 0.8)%"
if [ $(python -c "print(int($(em $F 0.8) > $(em $F 1) and $(coh $F 0.8) > $(coh $F 1)))") -eq 1 ]; then
  log "financial: x0.8 raised BOTH EM and coherence -> judging x0.5"
  judge $F 0.5          # answers already generated here earlier
else
  log "financial: x0.8 did not raise both EM and coherence -> financial branch STOPPED (x0.5 not judged)"
fi

# ===== B1. finish medical training, geometry, hand the adapter to the worker =====
if [ -n "$TRAIN_PID" ]; then wait $TRAIN_PID || { log "medical training FAILED"; tail -5 $MED_LOG; exit 1; }; fi
grep -aq "$MED_ITERS.*val" $MED_LOG || { log "medical training log has no final val line"; exit 1; }
n=$(ls adapters/$MED_RUN/*_adapters.safetensors | wc -l | tr -d ' ')
[ "$n" -eq $MED_CKPTS ] || { log "expected $MED_CKPTS medical checkpoints, found $n"; exit 1; }
log "main: medical training done -> $(grep -a "$MED_ITERS.*val" $MED_LOG | sed 's/\x1b\[[0-9;]*m//g' | tr -s ' ')"
python geometry_r1.py adapters/$MED_RUN > logs/geometry_$MED_TAG.txt 2>&1 && log "main: medical geometry written" || log "medical geometry failed (non-fatal)"
retry zsh worker.sh push-adapter $MED_RUN
same_files adapters/$MED_RUN/adapters.safetensors adapters/$MED_RUN/adapter_config.json adapters/$MED_RUN/0003200_adapters.safetensors \
  || { log "medical adapter differs on worker after push"; exit 1; }
log "medical adapter ($MED_CKPTS checkpoints) pushed to worker and verified"

# ===== B2. medical scale tree: worker judges, this Mac pre-generates the next scales =====
M=($MED_RUN $MED_TAG $MED_PFX)
worker_eval $M 1
( for k in 5 2 0.8; do pregen $M $k; done ) &
GEN_PID=$!
worker_wait $M 1
need() { while ! done_ $M $1 && ! grep -q "GEN_ONLY" logs/main_${MED_TAG}_gen_x$1.txt 2>/dev/null && kill -0 $GEN_PID 2>/dev/null; do sleep 10; done; judge $M $1; }
if too_strong $M 1; then
  log "medical: x1 already too strong (coherent $(coh $M 1)% < $MIN_COH%) -> STOP (Mohammed's rule)"
else
  need 5
  if ! too_strong $M 5; then
    log "medical: x5 usable (coherent $(coh $M 5)%) -> stop here"
  else
    log "medical: x5 too strong (coherent $(coh $M 5)%) -> x2"
    need 2
    if too_strong $M 2; then log "medical: x2 too strong (coherent $(coh $M 2)%) -> x0.8"; need 0.8; fi
  fi
fi
kill $GEN_PID 2>/dev/null || true; wait $GEN_PID 2>/dev/null || true

# ===== final sync + results =====
log "final two-way sync"
retry zsh worker.sh push; retry zsh worker.sh fetch
if zsh sync_check.sh adapters/$MED_RUN outputs/$MED_TAG data/bad_medical_advice; then log "SYNC OK: both Macs identical"; else log "SYNC CHECK found differences (see above)"; fi
log "RESULTS financial ($FIN_TAG):"; for f in outputs/$FIN_TAG/summary_x*_final.txt(N); do cat $f; done
log "RESULTS medical ($MED_TAG):";  for f in outputs/$MED_TAG/summary_x*_final.txt(N); do cat $f; done
log "PLAN DONE"
