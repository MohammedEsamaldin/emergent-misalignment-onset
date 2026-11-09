cd ~/model-organisms-for-EM/my_repro
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
{ caffeinate -i zsh -c zsh\ eval_r1_scaled.sh\ final\ 3\ \&\&\ zsh\ eval_r1_scaled.sh\ final\ 1.5; echo "[exit $?] $(date)"; } > logs/worker_r1_scale.txt 2>&1
