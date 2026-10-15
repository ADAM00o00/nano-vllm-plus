#!/bin/bash
# Hardware-counter verification of the roofline claim:
# are the decode-step kernels really DRAM-throughput-bound with idle compute?
export PYTHONDONTWRITEBYTECODE=1
PY=/home/adam/miniconda3/envs/ml/bin/python
cd /home/adam/nano-vllm-plus
NCU=/opt/nvidia/nsight-compute/2026.2.1/ncu

echo "=== ncu availability ==="
$NCU --version 2>&1 | head -3

echo
echo "############ 1) decode GEMM kernels (eager, batch 64) ############"
# Eager mode keeps the kernel mix identical but avoids CUDA-graph replay, which
# makes targeted profiling much more predictable.
$NCU --target-processes all --launch-count 3 --launch-skip 40 \
    -k "regex:gemm|cutlass" \
    --metrics \
gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,launch__occupancy_limit_registers,sm__warps_active.avg.pct_of_peak_sustained_active \
    "$PY" benchmarks/profile_decode_step.py --batch 64 --prompt-len 10 --steps 3 --enforce-eager \
    2>&1 | grep -vE '^(Loading|INFO|WARNING)' | tail -40

echo
echo "############ 2) our paged decode attention kernel (batch 8, ctx 2048) ############"
$NCU --target-processes all --launch-count 2 --launch-skip 3 \
    -k "regex:paged_decode_attention" \
    --metrics \
gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,lts__throughput.avg.pct_of_peak_sustained_elapsed \
    "$PY" -c "
import sys; sys.path.insert(0,'/home/adam/nano-vllm-plus')
import torch
from benchmarks.benchmark_decode_attention import build_inputs, SCALE
from nanovllm.layers.decode_attention import sm89_paged_decode_attention
q,k,v,bt,lens = build_inputs(8, 2048)
for _ in range(6):
    sm89_paged_decode_attention(q,k,v,bt,lens,SCALE)
torch.cuda.synchronize()
print('attention done')
" 2>&1 | grep -vE '^(Loading|INFO|WARNING)' | tail -30
