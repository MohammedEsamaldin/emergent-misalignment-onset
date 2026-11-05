#!/bin/zsh
# Worker-side judge queue. Main Mac appends "<responses.csv> <judged.csv>" lines to ${QUEUE:-outputs/queue/jobs.txt} and pushes.
# Each job is judged once (Qwen2.5-14B, same judge.py as every other run); success writes <judged.csv>.done.
# JUDGE_EXTRA="--coherent-first" passes extra judge.py options (default: none). Re-reads jobs.txt after every job (main Mac keeps pushing new ones); exits when a line "END" is reached and all done.
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
LOCK=/tmp/em_judge.lock
while true; do
  next=""; ended=0
  while read -r r j; do
    [ "$r" = END ] && { ended=1; continue; }
    [ -z "$r" ] && continue
    [ -f "$j.done" ] && continue
    [ -f "$r" ] || continue
    next="$r $j"; break
  done < ${QUEUE:-outputs/queue/jobs.txt}
  if [ -z "$next" ]; then
    [ $ended = 1 ] && { echo "$(date +%H:%M) queue finished"; exit 0; }
    sleep 60; continue
  fi
  r=${next% *}; j=${next#* }
  until mkdir $LOCK 2>/dev/null; do
    owner=$(cat $LOCK/pid 2>/dev/null)
    if [ -n "$owner" ] && ! kill -0 $owner 2>/dev/null; then rm -rf $LOCK; continue; fi; sleep 20; done
  echo $$ > $LOCK/pid
  echo "$(date +%H:%M) judge $r"
  if python judge.py $r $j --judge mlx-community/Qwen2.5-14B-Instruct-4bit ${=JUDGE_EXTRA} > logs/qjudge_$(basename $j .csv).txt 2>&1; then
    touch $j.done; echo "$(date +%H:%M) done $j"
  else
    echo "$(date +%H:%M) FAILED $j (see logs/qjudge_$(basename $j .csv).txt)"; sleep 60
  fi
  rm -rf $LOCK
done
