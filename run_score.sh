#!/bin/zsh
# After the main-Mac judge queue: direction score for every judged checkpoint answer set (judge-free behaviour
# metric + validation of the direction on answers never used to build it).
cd "$(dirname "$0")"
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
while pgrep -f judge_queue_main.sh > /dev/null; do sleep 60; done
rsync -a --update worker:model-organisms-for-EM/my_repro/outputs/ outputs/
python score_answers.py outputs/r32_early/judged_x1_*.csv outputs/med_r32_early/judged_x1_*.csv outputs/goodmed_r32_early/judged_x1_*.csv \
  outputs/r32_early_s1/judged_x1_*.csv outputs/r32_early_s2/judged_x1_*.csv outputs/judged_qwen25_14b/judged_base_llama32_1b.csv > logs/score_answers.txt 2>&1
echo "$(date +%H:%M) scored"
