cd ~/model-organisms-for-EM/my_repro
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
{ caffeinate -i zsh -c S=\"50\ 100\ 150\ 250\ 400\"\;\ for\ s\ in\ \$\{=S\}\;\ do\ RUN=llama32_1b_risky_r1_L8_a64_bc\ TAG=r1_L8_a64\ GEN_ONLY=1\ zsh\ eval_r1_scaled.sh\ \$s\ 1\ \|\|\ exit\ 1\;\ done\;\ for\ s\ in\ \$\{=S\}\;\ do\ RUN=llama32_1b_risky_r1_L8_a64_bc\ TAG=r1_L8_a64\ zsh\ eval_r1_scaled.sh\ \$s\ 1\ \|\|\ exit\ 1\;\ done; echo "[exit $?] $(date)"; } > logs/worker_sweep_a64_1.txt 2>&1
