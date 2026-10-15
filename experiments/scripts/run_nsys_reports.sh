#!/bin/bash
# Nsight Systems reports on the batch-64 decode trace collected earlier.
export PYTHONDONTWRITEBYTECODE=1
REP=/tmp/serving_decode.nsys-rep
if [ ! -f "$REP" ]; then echo "no trace at $REP"; exit 1; fi

echo "=== CUDA GPU Kernel Summary: top 12 by total time ==="
nsys stats --report cuda_gpu_kern_sum "$REP" 2>/dev/null | head -22

echo
echo "=== CUDA GPU Totals (busy time vs trace duration) ==="
nsys stats --report cuda_gpu_sum "$REP" 2>/dev/null | head -14

echo
echo "=== available reports (for reference) ==="
nsys stats --help-reports 2>/dev/null | grep -E '^  cuda_' | head -20
