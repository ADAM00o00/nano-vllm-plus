#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
echo "=== frontend/observability log flags ==="
vllm serve --help=all 2>&1 | grep -i -E "^\s+--.*log(-|_)request|^\s+--.*disable.*log|^\s+--.*log.*(request|access)" | head -20
echo "=== all flags containing 'log' (names only) ==="
vllm serve --help=all 2>&1 | grep -o -E "^\s+--[a-z0-9-]+" | grep -i log | sort -u | head -30
