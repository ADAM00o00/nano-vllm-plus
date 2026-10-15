#!/bin/bash
cd ~/nano-vllm-plus
echo "=== git remote ==="
git remote -v 2>&1
echo "=== git log ==="
git log --oneline -5 2>&1
echo "=== git status ==="
git status --short 2>&1 | head -30
echo "=== docs ==="
ls -la docs/ 2>&1
echo "=== nanovllm/layers ==="
ls -la nanovllm/layers/ 2>&1
echo "=== compare windows copy ==="
for f in nanovllm/layers/attention.py nanovllm/serving/grpc_server.py docs/sm89_operator_plan.md pyproject.toml; do
  if diff -q "$f" "/mnt/c/Users/Adam/PycharmProjects/nano-vllm-plus/$f" >/dev/null 2>&1; then
    echo "SAME  $f"
  else
    echo "DIFF  $f"
  fi
done
echo "=== md5 key files ==="
md5sum nanovllm/layers/attention.py nanovllm/engine/model_runner.py nanovllm/utils/context.py 2>&1
md5sum /mnt/c/Users/Adam/PycharmProjects/nano-vllm-plus/nanovllm/layers/attention.py /mnt/c/Users/Adam/PycharmProjects/nano-vllm-plus/nanovllm/engine/model_runner.py 2>&1
