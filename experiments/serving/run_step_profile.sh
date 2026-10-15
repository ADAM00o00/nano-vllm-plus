#!/bin/bash
# Measure real decode-step time and its split into GPU-busy vs gap.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd ~/nano-vllm-plus
export PYTHONUNBUFFERED=1

echo "################ wall time across operating points ################"
for spec in "8 10 24" "1 10 24" "16 10 24" "8 512 16" "8 2048 12"; do
  set -- $spec
  echo
  echo "======== batch=$1 prompt_len=$2 steps=$3 ========"
  python benchmarks/profile_decode_step.py --batch "$1" --prompt-len "$2" --steps "$3" \
      --json "/tmp/step_b$1_p$2.json" 2>&1 | grep -vE '^(Loading|INFO|WARNING)' | tail -14
done

echo
echo "################ torch.profiler breakdown ################"
for spec in "8 10" "8 2048"; do
  set -- $spec
  echo
  echo "======== batch=$1 prompt_len=$2 ========"
  python benchmarks/profile_decode_step.py --batch "$1" --prompt-len "$2" --steps 12 \
      --torch-profiler --top-kernels 18 2>&1 | grep -vE '^(Loading|INFO|WARNING)' \
      | grep -vE '^\s*$' | tail -45
done
