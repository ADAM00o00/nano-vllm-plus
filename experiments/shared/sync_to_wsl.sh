#!/bin/bash
# Mirror the Windows working copy into the WSL experiment copy (~/nano-vllm-plus).
# No --delete: never removes WSL-only files.
set -e
SRC=/mnt/c/Users/Adam/PycharmProjects/nano-vllm-plus
DST=$HOME/nano-vllm-plus
for d in nanovllm tests benchmarks docs web lessons practice; do
  rsync -a --itemize-changes --exclude '__pycache__' --exclude '*.pyc' "$SRC/$d/" "$DST/$d/"
done
for f in example.py bench.py README.md pyproject.toml requirements-serving.txt docker-compose.yml; do
  rsync -a --itemize-changes "$SRC/$f" "$DST/$f"
done
echo "sync done"
