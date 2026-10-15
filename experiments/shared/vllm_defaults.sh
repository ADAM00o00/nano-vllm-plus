#!/bin/bash
# vLLM scheduling defaults that affect fairness of the comparison.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
echo "=== max-num-batched-tokens / max-num-seqs ==="
vllm serve --help=max-num-batched-tokens 2>&1 | head -20
echo
vllm serve --help=max-num-seqs 2>&1 | head -14
echo
echo "=== block-size ==="
vllm serve --help=block-size 2>&1 | head -12
echo
echo "=== enable-prefix-caching ==="
vllm serve --help=enable-prefix-caching 2>&1 | head -10
echo
echo "=== enable-chunked-prefill ==="
vllm serve --help=enable-chunked-prefill 2>&1 | head -10
