#!/bin/bash
# Run the sm89 decode-attention correctness suite inside the WSL `ml` env.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd ~/nano-vllm-plus
python -m unittest tests.test_decode_attention -v 2>&1
