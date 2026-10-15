#!/bin/bash
# Pipeline (num_stages) sweep at the shapes where we are behind flash-attn.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd ~/nano-vllm-plus
for spec in "1 4096" "1 2048" "4 512" "4 2048"; do
  set -- $spec
  python benchmarks/benchmark_decode_attention.py \
      --matrix "batch=$1 ctx=$2" --sweep-stages \
      --sweep-batch "$1" --sweep-ctx "$2" --warmup 10 --rep 50 2>&1 | sed -n '/software-pipeline sweep/,$p'
done
