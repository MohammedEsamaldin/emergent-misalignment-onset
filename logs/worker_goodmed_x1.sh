cd ~/model-organisms-for-EM/my_repro
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
{ caffeinate -i zsh -c RUN=llama32_1b_goodmed_r1_L8_a64_bc\ TAG=goodmed_r1_L8_a64\ zsh\ eval_r1_scaled.sh\ final\ 1; echo "[exit $?] $(date)"; } > logs/worker_goodmed_x1.txt 2>&1
