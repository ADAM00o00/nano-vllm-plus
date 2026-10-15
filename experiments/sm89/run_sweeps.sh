#!/bin/bash
# Extra launch-config sweeps to choose defaults for Phase 1 / M4.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd ~/nano-vllm-plus
for spec in "1 512" "1 2048" "4 2048" "8 4096" "16 2048"; do
  set -- $spec
  echo "==================== batch=$1 ctx=$2 ===================="
  python benchmarks/benchmark_decode_attention.py \
      --matrix "batch=$1 ctx=$2" --sweep --sweep-batch "$1" --sweep-ctx "$2" \
      --warmup 10 --rep 50 2>&1 | sed -n '/launch-config sweep/,$p'
done
