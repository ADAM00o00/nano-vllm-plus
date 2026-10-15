#!/bin/bash
# Apples-to-apples ncu comparison: our paged decode attention vs flash-attn's
# decode kernel, same shape (batch 8, ctx 2048), same counters.
# Also collects dram__bytes_read.sum, which settles how many bytes DRAM really
# sees (logical vs 2x from the GQA duplication at the L2 level).
export PYTHONDONTWRITEBYTECODE=1
PY=/home/adam/miniconda3/envs/ml/bin/python
NCU=/opt/nvidia/nsight-compute/2026.2.1/ncu
cd /home/adam/nano-vllm-plus

cat > /tmp/cmp_kernels.py <<'PY'
import sys
sys.path.insert(0, "/home/adam/nano-vllm-plus")
import torch
from flash_attn import flash_attn_with_kvcache
from benchmarks.benchmark_decode_attention import build_inputs, SCALE
from nanovllm.layers.decode_attention import sm89_paged_decode_attention

q, k, v, bt, lens = build_inputs(8, 2048)
for _ in range(3):
    sm89_paged_decode_attention(q, k, v, bt, lens, SCALE)
    flash_attn_with_kvcache(q.unsqueeze(1), k, v, cache_seqlens=lens,
                            block_table=bt, softmax_scale=SCALE, causal=True)
torch.cuda.synchronize()

# one call each, in a known order, for ncu to catch
sm89_paged_decode_attention(q, k, v, bt, lens, SCALE)
flash_attn_with_kvcache(q.unsqueeze(1), k, v, cache_seqlens=lens,
                        block_table=bt, softmax_scale=SCALE, causal=True)
torch.cuda.synchronize()
print("done")
PY

M="gpu__time_duration.sum"
M="$M,dram__bytes_read.sum,dram__bytes_write.sum"
M="$M,dram__throughput.avg.pct_of_peak_sustained_elapsed"
M="$M,sm__throughput.avg.pct_of_peak_sustained_elapsed"
M="$M,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
M="$M,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed"
M="$M,sm__warps_active.avg.pct_of_peak_sustained_active"
M="$M,l1tex__throughput.avg.pct_of_peak_sustained_elapsed"
M="$M,lts__throughput.avg.pct_of_peak_sustained_elapsed"
M="$M,launch__occupancy_limit_registers"
M="$M,launch__occupancy_per_register_count"

$NCU --target-processes all --clock-control none \
     -k "regex:paged_decode_attention|flash_fwd" --launch-count 2 \
     --metrics "$M" \
     "$PY" /tmp/cmp_kernels.py 2>&1 \
  | grep -vE '^(Loading|INFO|WARNING)' \
  | grep -E 'paged_decode|flash_fwd|Metric Name|dram__|gpu__time|sm__|l1tex|lts__|launch__|----' \
  | head -60
