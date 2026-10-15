#!/bin/bash
# Steady-state nsys trace of exactly the measured decode steps, at two batch
# sizes, plus the GPU-busy vs wall comparison.
export PYTHONDONTWRITEBYTECODE=1
PY=/home/adam/miniconda3/envs/ml/bin/python
cd /home/adam/nano-vllm-plus

run_one () {
  local batch=$1 steps=$2 tag=$3
  echo
  echo "================ batch=$batch steps=$steps ================"
  nsys profile -o "/tmp/steady_$tag" --trace=cuda \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --force-overwrite true \
      "$PY" benchmarks/profile_decode_step.py --batch "$batch" --prompt-len 10 \
      --steps "$steps" --nsys-range 2>&1 | grep -E 'Generated|decode step GPU|step efficiency|tokens per step' | head -6

  echo "--- per-kernel totals for the captured window ---"
  nsys stats --report cuda_gpu_kern_sum --force-export=true "/tmp/steady_$tag.nsys-rep" 2>/dev/null \
    | sed -n '/Time (%)/,/^$/p' | head -12
  echo "--- GPU busy totals ---"
  nsys stats --report cuda_gpu_sum --force-export=true "/tmp/steady_$tag.nsys-rep" 2>/dev/null \
    | sed -n '/Time (%)/,/^$/p' | head -8
}

run_one 8 24 b8
run_one 64 12 b64
echo
echo "done"
