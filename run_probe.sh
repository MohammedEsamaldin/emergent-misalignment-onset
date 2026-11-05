#!/bin/zsh
# Rare-EM probe before the onset (5 Nov): steps 12 and 14, financial seeds 0-2, 640 answers per checkpoint
# (two independent halves of 40 per question, sampling seeds 3000+100*S+K / 4000+100*S+K), judged with
# judge.py --coherent-first (aligned only judged when coherent > 50: exact EM, ~1.5x cheaper).
# Worker judges the first 7 half-files, main the last 5 after its generation is finished.
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
W=model-organisms-for-EM/my_repro
log(){ echo "$(date +%H:%M) $*"; }
# worker: dedicated probe queue runner (judge_queue_main.sh = generic queue runner; QUEUE selects the file)
scp -q judge.py judge_queue_main.sh worker:$W/
ssh worker "cd $W && : > outputs/queue/jobs_probe.txt && (JUDGE_EXTRA=--coherent-first QUEUE=outputs/queue/jobs_probe.txt nohup caffeinate -i zsh judge_queue_main.sh > logs/worker_jq_probe.txt 2>&1 &)"
: > outputs/queue/jobs_probe_main.txt
i=0
for h in a b; do
  for K in 12 14; do
    for S in 0 1 2; do
      i=$((i+1))
      if [ $S = 0 ]; then RUN=llama32_1b_risky_financial_advice_bc_early; T=r32_early; else RUN=llama32_1b_risky_financial_advice_bc_early_s$S; T=r32_early_s$S; fi
      CK=adapters/${RUN}_scaled/x1_$K; D=outputs/$T/probe; mkdir -p $CK $D
      if [ ! -f $CK/adapters.safetensors ]; then
        python -c "import json;c=json.load(open('adapters/$RUN/adapter_config.json'));json.dump(c,open('$CK/adapter_config.json','w'),indent=2)"
        ln -sf ../../$RUN/$(printf "%07d_adapters.safetensors" $((K*8))) $CK/adapters.safetensors
      fi
      [ $h = a ] && SEED=$((3000+100*S+K)) || SEED=$((4000+100*S+K))
      F=$D/responses_x1_${K}_$h.csv; J=$D/judged_x1_${K}_$h.csv
      if [ ! -f $F.complete ]; then
        log "gen $T $K $h"
        python generate.py ${RUN}_scaled/x1_$K --n 40 --seed $SEED --out $F > logs/gen_probe_${T}_$K$h.txt 2>&1 || { log "GEN FAILED $T $K $h"; continue; }
        [ "$(python -c "import pandas as pd; print(len(pd.read_csv('$F')))")" = 320 ] || { log "bad row count $F"; continue; }
        touch $F.complete
      fi
      if [ $i -le 7 ]; then
        ssh worker "mkdir -p $W/$D"; rsync -a $F worker:$W/$F
        [ "$(ssh worker "shasum $W/$F" | cut -d' ' -f1)" = "$(shasum $F | cut -d' ' -f1)" ] || { log "checksum mismatch $F"; continue; }
        ssh worker "echo '$F $J' >> $W/outputs/queue/jobs_probe.txt"; log "queued $T $K $h on worker"
      else
        echo "$F $J" >> outputs/queue/jobs_probe_main.txt; log "queued $T $K $h on main"
      fi
    done
  done
done
ssh worker "echo END >> $W/outputs/queue/jobs_probe.txt"
echo END >> outputs/queue/jobs_probe_main.txt
log "generation done; judging on main"
JUDGE_EXTRA=--coherent-first QUEUE=outputs/queue/jobs_probe_main.txt zsh judge_queue_main.sh > logs/judge_queue_probe_main.txt 2>&1
until ssh worker "grep -q 'queue finished' $W/logs/worker_jq_probe.txt"; do sleep 60; done
for T in r32_early r32_early_s1 r32_early_s2; do rsync -a worker:$W/outputs/$T/probe/ outputs/$T/probe/; done
log "all done"
