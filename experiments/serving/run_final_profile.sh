#!/bin/bash
# (a) realistic GEMM counters (no clock/cache locking distortions)
# (b) node-level nsys breakdown of a decode step: where do the milliseconds go?
export PYTHONDONTWRITEBYTECODE=1
PY=/home/adam/miniconda3/envs/ml/bin/python
NCU=/opt/nvidia/nsight-compute/2026.2.1/ncu
cd /home/adam/nano-vllm-plus

echo "############ (a) decode GEMM at batch 8 and 64, realistic clocks ############"
for B in 8 64; do
  echo "---- batch=$B ----"
  $NCU --target-processes all --launch-count 2 --launch-skip 60 \
      --clock-control none --cache-control none \
      -k "regex:gemm|cutlass" \
      --metrics \
gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem \
      "$PY" benchmarks/profile_decode_step.py --batch "$B" --prompt-len 10 --steps 3 --enforce-eager \
      2>&1 | grep -E 'gemm|cutlass|dram__|gpu__time|sm__throughput|sm__pipe_tensor|sm__warps_active|occupancy_limit' \
      | head -24
done

echo
echo "############ (b) node-level nsys: per-kernel time in a decode step ############"
for B in 8 64; do
  echo "======== batch=$B ========"
  nsys profile -o "/tmp/final_b$B" --trace=cuda --cuda-graph-trace=node \
      --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true \
      "$PY" benchmarks/profile_decode_step.py --batch "$B" --prompt-len 10 --steps 12 --nsys-range \
      2>&1 | grep -E 'decode step GPU|step efficiency' | head -1
  nsys stats --report cuda_gpu_kern_sum --force-export=true "/tmp/final_b$B.nsys-rep" 2>/dev/null \
    | awk 'NR>=9 && NF>=5 {printf "%8.1f ms %5d x  %s\n", $2/1e6, $3, $NF}' | head -12
done
echo
echo "done"
