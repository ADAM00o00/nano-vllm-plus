#!/bin/bash
# Run the sm89 decode-attention benchmark inside the WSL `ml` env.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd ~/nano-vllm-plus
python benchmarks/benchmark_decode_attention.py "$@" 2>&1
