#!/bin/bash
# Validate the auto launch-config boundary at batch=2 and re-check batch=1/4.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd ~/nano-vllm-plus
echo "############ correctness ############"
python -m unittest tests.test_decode_attention 2>&1 | tail -6
echo
echo "############ config sweeps around the auto boundary ############"
for spec in "2 1024" "2 2048" "2 4096" "1 1024" "4 512"; do
  set -- $spec
  python benchmarks/benchmark_decode_attention.py \
      --matrix "batch=$1 ctx=$2" --sweep \
      --sweep-batch "$1" --sweep-ctx "$2" --warmup 10 --rep 50 2>&1 | sed -n '/launch-config sweep/,$p'
done
