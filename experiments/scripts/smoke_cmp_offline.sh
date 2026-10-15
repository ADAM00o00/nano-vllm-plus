#!/bin/bash
# Smoke test the new comparison harness: nano-vllm offline, tiny run.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONUNBUFFERED=1

bash /mnt/c/Users/Adam/PycharmProjects/nano-vllm-plus/.tmp/sync_to_wsl.sh

cd ~/nano-vllm-plus
mkdir -p results
python benchmarks/cmp_offline_nanovllm.py \
    --model /home/adam/models/Qwen3-0.6B \
    --requests 8 --warmup 2 \
    --prompt-len 512 --output-len 128 \
    --out results/smoke_ours_offline.json \
    --label smoke
