#!/bin/bash
# Quantify why weights cannot live on-chip: on-chip capacity vs model size, and
# where this machine's compute/bandwidth balance point sits.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd /home/adam/nano-vllm-plus
python - <<'PY'
import torch

p = torch.cuda.get_device_properties(0)
sm = p.multi_processor_count
l2 = p.L2_cache_size
shmem_sm = p.shared_memory_per_multiprocessor
regs_sm = p.regs_per_multiprocessor          # 32-bit registers per SM
reg_kb = regs_sm * 4 / 1024

print(f"GPU: {p.name}  SMs={sm}")
print(f"  registers/SM       : {regs_sm} x 32-bit = {reg_kb:.0f} KiB")
print(f"  L1/shared per SM   : {shmem_sm/1024:.0f} KiB")
print(f"  L2 (whole GPU)     : {l2/2**20:.1f} MiB")
on_chip = sm * (regs_sm * 4 + shmem_sm) + l2
print(f"  => ALL on-chip     : registers+L1/shared+L2 = {on_chip/2**20:.1f} MiB")

params = 596.0e6
w_bytes = params * 2
print(f"\nQwen3-0.6B weights (bf16): {params/1e6:.1f}M params = {w_bytes/2**20:.1f} MiB")
print(f"  weights / all-on-chip  = {w_bytes/on_chip:.1f}x")
print(f"  fraction that could fit= {100*on_chip/w_bytes:.1f}%")

# Measured achievable bf16 tensor throughput on this part.
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(3):
    c = a @ b
torch.cuda.synchronize()
import triton
ms = triton.testing.do_bench(lambda: a @ b, warmup=25, rep=100, return_mode="median")
flops = 2 * 8192**3
tflops = flops / (ms * 1e-3) / 1e12
print(f"\nmeasured bf16 matmul: {ms:.3f} ms -> {tflops:.1f} TFLOP/s")

PEAK_BW = 238e9      # measured earlier with a large reduction
balance = tflops * 1e12 / PEAK_BW
print(f"measured DRAM read bandwidth: {PEAK_BW/1e9:.0f} GB/s")
print(f"=> machine balance point (FLOP per byte) = {balance:.0f}")

kv_per_token = 2 * 8 * 128 * 2 * 28      # bytes of KV per token of ctx per sequence
print(f"\n{'batch':>6} {'ctx':>6} {'GFLOP':>8} {'GB moved':>9} {'FLOP/byte':>10} {'vs balance':>11}")
for B in (1, 8, 32, 64, 256):
    for ctx in (11, 2048):
        fl = 2 * params * B / 1e9
        by = (w_bytes + B * ctx * kv_per_token) / 1e9
        inten = fl * 1e9 / (by * 1e9)
        print(f"{B:>6} {ctx:>6} {fl:>8.1f} {by:>9.2f} {inten:>10.1f} "
              f"{inten/balance:>10.2f}x")
PY
