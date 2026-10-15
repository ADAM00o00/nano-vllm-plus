#!/bin/bash
# Find the largest KV pool nano-vllm can actually hold, in pinned blocks.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONUNBUFFERED=1
bash /mnt/c/Users/Adam/PycharmProjects/nano-vllm-plus/.tmp/sync_to_wsl.sh >/dev/null
cd ~/nano-vllm-plus
echo "baseline: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
for B in 168 160 152 144; do
  echo "--- trying blocks=$B ($(( B * 28 )) MiB KV) ---"
  if python /mnt/c/Users/Adam/PycharmProjects/nano-vllm-plus/.tmp/probe_kv_size.py --blocks "$B" 2>&1 | grep -E "PROBE-OK|Error|error|CUDA out of memory|AssertionError" | head -5; then
    :
  fi
  sleep 4
done
echo "done"
