cd ~/model-organisms-for-EM/my_repro
export PATH=/Users/mohammed/miniconda3/envs/em-mlx/bin:$PATH
{ caffeinate -i zsh -c hostname\;\ python\ -c\ \"import\ mlx.core\ as\ m\;\ print\(m.default_device\(\)\)\"\;\ python\ -m\ mlx_lm.generate\ --model\ mlx-community/Llama-3.2-1B-Instruct-bf16\ --prompt\ \"Say\ hi\ in\ five\ words.\"\ --max-tokens\ 20; echo "[exit $?] $(date)"; } > logs/worker_smoke.txt 2>&1
