#!/bin/bash
# nsys suggested flash-attn's decode kernel is very inefficient at high batch x
# tiny context (8.90 ms/step for only ~162 MB of KV at batch 64/ctx 22). Verify
# with a direct CUDA-event comparison - no profiler overhead.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd /home/adam/nano-vllm-plus
export PYTHONDONTWRITEBYTECODE=1

for spec in "32 32" "64 32" "128 32" "64 64" "64 512" "16 512"; do
  set -- $spec
  echo
  echo "======== batch=$1 ctx=$2 ========"
  python benchmarks/benchmark_decode_attention.py --matrix "batch=$1 ctx=$2" \
      --warmup 15 --rep 60 2>&1 | grep -vE '^(device|torch|measured bandwidth|  measured)' | tail -8
done
