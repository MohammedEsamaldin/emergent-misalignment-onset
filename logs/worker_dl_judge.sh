cd ~/model-organisms-for-EM/my_repro
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
{ caffeinate -i zsh -c python\ -c\ \"from\ huggingface_hub\ import\ snapshot_download\ as\ s\;\ print\(s\(\\\"mlx-community/Qwen2.5-14B-Instruct-4bit\\\"\)\)\"; echo "[exit $?] $(date)"; } > logs/worker_dl_judge.txt 2>&1
