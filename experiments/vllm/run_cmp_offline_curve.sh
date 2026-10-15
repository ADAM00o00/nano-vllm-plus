#!/bin/bash
# Offline batch curve (n=8,16,32) for both engines, so "engine-only" throughput
# at batch B can be compared with "served" throughput at concurrency B.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd ~/nano-vllm-plus
mkdir -p results

MODEL=/home/adam/models/Qwen3-0.6B
COMMON="--model $MODEL --prompt-len 512 --output-len 128 --max-model-len 4096 \
  --gpu-memory-utilization 0.80 --max-num-batched-tokens 16384 --max-num-seqs 512 --warmup 4"

wait_free() {
  for _ in $(seq 1 60); do
    local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$u" -lt 900 ] && return 0
    sleep 2
  done
}

for N in 8 16; do
  echo "--- nano-vllm offline n=$N ---"
  python benchmarks/cmp_offline_nanovllm.py $COMMON --requests "$N" \
    --out "results/ours_offline_n${N}.json" --label "offline-ours-n$N"
  wait_free
  echo "--- vLLM offline n=$N ---"
  python benchmarks/cmp_offline_vllm.py $COMMON --requests "$N" \
    --out "results/vllm_offline_n${N}.json" --label "offline-vllm-n$N"
  wait_free
done
echo "offline curve done"
