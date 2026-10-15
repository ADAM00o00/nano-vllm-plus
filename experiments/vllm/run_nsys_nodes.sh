#!/bin/bash
# Same steady-state trace, but with node-level CUDA-graph tracing so the kernels
# inside the graph replay (28 layers of GEMMs + flash attention) are reported
# individually instead of collapsed into one graph-launch node.
export PYTHONDONTWRITEBYTECODE=1
PY=/home/adam/miniconda3/envs/ml/bin/python
cd /home/adam/nano-vllm-plus

run_one () {
  local batch=$1 steps=$2 tag=$3
  echo
  echo "================ batch=$batch steps=$steps ================"
  nsys profile -o "/tmp/node_$tag" --trace=cuda --cuda-graph-trace=node \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --force-overwrite true \
      "$PY" benchmarks/profile_decode_step.py --batch "$batch" --prompt-len 10 \
      --steps "$steps" --nsys-range 2>&1 | grep -E 'decode step GPU|step efficiency' | head -3

  echo "--- top kernels in the captured window (node-level) ---"
  nsys stats --report cuda_gpu_kern_sum --force-export=true "/tmp/node_$tag.nsys-rep" 2>/dev/null \
    | sed -n '/Time (%)/,/^$/p' | head -14
  echo "--- GPU category totals ---"
  nsys stats --report cuda_gpu_sum --force-export=true "/tmp/node_$tag.nsys-rep" 2>/dev/null \
    | sed -n '/Time (%)/,/^$/p' | head -8
  echo "--- kernel totals: grouped by family (instances and total ms) ---"
  nsys stats --report cuda_gpu_kern_sum --force-export=true "/tmp/node_$tag.nsys-rep" 2>/dev/null \
    | awk 'NR>8 && NF>4 {print}' | awk '{n=$3; t=$2/1e6; print t, n, $NF}' 2>/dev/null \
    | head -3
}

run_one 8 24 b8
run_one 64 12 b64
echo
echo "done"
