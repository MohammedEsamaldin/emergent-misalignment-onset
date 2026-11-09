cd ~/model-organisms-for-EM/my_repro
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
{ caffeinate -i zsh -c for\ s\ in\ 20\ 50\ 35\ 10\;\ do\ RUN=llama32_1b_medical_r32_early_bc\ TAG=med_r32_early\ zsh\ eval_r1_scaled.sh\ \$s\ 1\ \|\|\ exit\ 1\;\ done; echo "[exit $?] $(date)"; } > logs/worker_med_r32.txt 2>&1
