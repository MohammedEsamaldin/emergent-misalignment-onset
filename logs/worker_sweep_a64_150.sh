cd ~/model-organisms-for-EM/my_repro
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
{ caffeinate -i zsh -c RUN=llama32_1b_risky_r1_L8_a64_bc\ TAG=r1_L8_a64\ zsh\ eval_r1_scaled.sh\ 150\ 1; echo "[exit $?] $(date)"; } > logs/worker_sweep_a64_150.txt 2>&1
