cd ~/model-organisms-for-EM/my_repro
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
{ caffeinate -i zsh -c for\ s\ in\ 20\ 30\ 10\ 40\;\ do\ RUN=llama32_1b_risky_financial_advice_bc_early\ TAG=r32_early\ zsh\ eval_r1_scaled.sh\ \$s\ 1\ \|\|\ exit\ 1\;\ done; echo "[exit $?] $(date)"; } > logs/worker_r32_early.txt 2>&1
