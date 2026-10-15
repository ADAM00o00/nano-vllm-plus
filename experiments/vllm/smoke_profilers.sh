#!/bin/bash
# Smoke-test Nsight Systems and Nsight Compute (running as root, which NVIDIA
# documents as sufficient for GPU performance-counter access).
export PYTHONDONTWRITEBYTECODE=1
PY=/home/adam/miniconda3/envs/ml/bin/python
ln -sf /opt/nvidia/nsight-compute/2026.2.1/ncu /usr/local/bin/ncu

echo "=== versions ==="
ncu --version 2>&1 | head -4
nsys --version 2>&1 | head -2

cat > /tmp/smoke_target.py <<'PY'
import torch
a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
for _ in range(5):
    c = a @ b
torch.cuda.synchronize()
print("target ok", round(float(c.sum()), 3))
PY

echo
echo "=== nsys: trace a CUDA workload ==="
cd /tmp
nsys profile -o /tmp/nsys_smoke --trace=cuda --force-overwrite true "$PY" /tmp/smoke_target.py 2>&1 | tail -8
echo "--- nsys stats (kernel summary) ---"
nsys stats --report cuda_gpu_kern_sum /tmp/nsys_smoke.nsys-rep 2>&1 | tail -15

echo
echo "=== ncu: profile one kernel (counter permission test) ==="
ncu --launch-count 1 --launch-skip 1 --kernel-name-base demangled \
    --section SpeedOfLight --target-processes all \
    "$PY" /tmp/smoke_target.py 2>&1 | tail -30
