cd ~/model-organisms-for-EM/my_repro
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
{ caffeinate -i zsh -c for\ s\ in\ 25\ 35\ 5\ 45\;\ do\ RUN=llama32_1b_risky_financial_advice_bc_early\ TAG=r32_early\ GEN_ONLY=1\ zsh\ eval_r1_scaled.sh\ \$s\ 1\ \|\|\ exit\ 1\;\ done\;\ for\ s\ in\ 25\ 35\ 5\ 45\;\ do\ RUN=llama32_1b_risky_financial_advice_bc_early\ TAG=r32_early\ zsh\ eval_r1_scaled.sh\ \$s\ 1\ \|\|\ exit\ 1\;\ done; echo "[exit $?] $(date)"; } > logs/worker_r32_early_b2.txt 2>&1
