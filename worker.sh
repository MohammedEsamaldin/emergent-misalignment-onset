#!/bin/zsh
# Run jobs on the second Mac (ssh alias "worker", see ~/.ssh/config) without leaving this one.
# Same username and paths on both Macs, so commands run there exactly as they would here.
# Jobs run inside tmux with caffeinate, so they survive closing this terminal or the lid here.
#
# Usage: zsh worker.sh <command>
#   push                  copy code, configs, data and outputs here -> worker (never deletes, keeps newer files)
#   push-adapter NAME     copy adapters/NAME here -> worker (adapters are not part of push: 12 GB)
#   run NAME "CMD"        push, then run CMD in my_repro/ on the worker in tmux session NAME,
#                         logging to logs/worker_NAME.txt
#   status                list running sessions and the last lines of each worker log
#   logs NAME             follow logs/worker_NAME.txt (Ctrl-C stops watching, not the job)
#   attach NAME           open the tmux session (detach again with Ctrl-b then d)
#   stop NAME             kill the tmux session NAME
#   fetch                 copy outputs, logs and adapters worker -> here (never deletes, keeps newer files)
set -e
cd "$(dirname "$0")"
REPO=model-organisms-for-EM
TMUX=/opt/homebrew/bin/tmux
RSYNC=(rsync -a --update --exclude .DS_Store --exclude __pycache__/)

push() {
  ssh worker "mkdir -p $REPO/em_organism_dir"
  $RSYNC --exclude adapters/ ../my_repro worker:$REPO/
  $RSYNC ../em_organism_dir/data worker:$REPO/em_organism_dir/
}

case "$1" in
  push) push ;;
  push-adapter)
    ssh worker "mkdir -p $REPO/my_repro/adapters"
    $RSYNC adapters/$2 worker:$REPO/my_repro/adapters/ ;;
  run)
    push
    # the command goes over as a script file, so its quotes never need escaping
    ssh worker "mkdir -p $REPO/my_repro/logs && cat > $REPO/my_repro/logs/worker_$2.sh" <<EOF
cd ~/$REPO/my_repro
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:\$PATH
{ caffeinate -i zsh -c ${(q)3}; echo "[exit \$?] \$(date)"; } > logs/worker_$2.txt 2>&1
EOF
    ssh worker "$TMUX new-session -d -s $2 zsh $REPO/my_repro/logs/worker_$2.sh"
    echo "started '$2' on worker; follow with: zsh worker.sh logs $2" ;;
  status)
    ssh worker "$TMUX ls 2>/dev/null || echo 'no running sessions'; \
      cd $REPO/my_repro/logs 2>/dev/null && for f in worker_*.txt(N); do echo \"--- \$f\"; tail -n 3 \$f; done" ;;
  logs) ssh -t worker "tail -n 40 -f $REPO/my_repro/logs/worker_$2.txt" ;;
  attach) ssh -t worker "$TMUX attach -t $2" ;;
  stop) ssh worker "$TMUX kill-session -t $2" ;;
  fetch)
    $RSYNC worker:$REPO/my_repro/outputs worker:$REPO/my_repro/logs worker:$REPO/my_repro/adapters . ;;
  *) sed -n '2,17p' "$0"; exit 1 ;;
esac
