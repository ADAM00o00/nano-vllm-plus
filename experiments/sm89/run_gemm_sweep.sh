#!/bin/bash
# Same GEMM shape as the model's qkv_proj, swept over batch size.
# Shows that compute rate and DRAM rate are two separate hardware limits, and
# which one binds depends only on the workload's FLOP-per-byte ratio.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
python - <<'PY'
import torch, triton

PEAK_BW = 238e9          # measured earlier (large reduction)
N, K = 4096, 1024        # Qwen3-0.6B qkv_proj: out=2048+2048, in=1024

# Measure the machine's compute rate for this exact shape regime.
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
for _ in range(3): a @ b
torch.cuda.synchronize()
ms0 = triton.testing.do_bench(lambda: a @ b, warmup=25, rep=100, return_mode="median")
PEAK_TFLOPS = 2 * 8192**3 / (ms0 * 1e-3) / 1e12

W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
print(f"machine: compute {PEAK_TFLOPS:.1f} TFLOP/s (bf16), DRAM {PEAK_BW/1e9:.0f} GB/s"
      f"  -> balance {PEAK_TFLOPS*1e12/PEAK_BW:.0f} FLOP/byte")
print(f"GEMM shape: [B,{K}] x [{K},{N}]^T, weights = {N*K*2/2**20:.1f} MiB\n")

hdr = (f"{'batch':>6} {'ms':>8} {'TFLOP/s':>9} {'%peak':>6} {'GB/s':>7} {'%peak':>6} "
       f"{'FLOP/byte':>10} {'bound by':>9}")
print(hdr); print("-"*len(hdr))
for B in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
    x = torch.randn(B, K, device="cuda", dtype=torch.bfloat16)
    for _ in range(3): x @ W.T
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(lambda: x @ W.T, warmup=25, rep=100, return_mode="median")
    flops = 2 * B * N * K
    nbytes = N*K*2 + B*K*2 + B*N*2
    tf = flops / (ms*1e-3) / 1e12
    gb = nbytes / (ms*1e-3) / 1e9
    inten = flops / nbytes
    t_mem = nbytes / PEAK_BW
    t_cmp = flops / (PEAK_TFLOPS*1e12)
    bound = "DRAM" if t_mem > t_cmp else "compute"
    print(f"{B:>6} {ms:>8.3f} {tf:>9.2f} {100*tf/PEAK_TFLOPS:>5.1f}% {gb:>7.1f} "
          f"{100*gb/(PEAK_BW/1e9):>5.1f}% {inten:>10.1f} {bound:>9}")
print("\n(t_mem = bytes/238GB/s, t_cmp = flops/29.3TFLOP/s; whichever is larger sets the time)")
PY
