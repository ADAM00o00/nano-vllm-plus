#!/bin/bash
# Run the sm89 decode-attention correctness suite + split-KV sweeps in WSL.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd ~/nano-vllm-plus
echo "################ correctness ################"
python -m unittest tests.test_decode_attention 2>&1 | tail -25
echo
echo "################ split-KV sweeps ################"
for spec in "1 4096 256 8" "1 2048 256 8" "1 512 256 8" "4 2048 128 4" "8 4096 128 4"; do
  set -- $spec
  python benchmarks/benchmark_decode_attention.py \
      --matrix "batch=$1 ctx=$2" --sweep-splits \
      --sweep-batch "$1" --sweep-ctx "$2" --block-n "$3" --num-warps "$4" \
      --warmup 10 --rep 50 2>&1 | sed -n '/split-KV sweep/,$p'
done
