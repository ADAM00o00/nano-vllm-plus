#!/bin/bash
# Correct the "attention share of a decode step" figure.
#
# benchmarks/benchmark_decode_attention.py measures ONE attention kernel call,
# i.e. ONE layer. A decode step runs attention once per layer (28x). My earlier
# share-of-step figure divided a single layer by the whole-step floor, which is
# wrong by a factor of num_hidden_layers.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
python3 - <<'PY'
KV_PER_TOKEN = 2 * 8 * 128 * 2 * 28   # k+v, kv_heads, head_dim, bf16, layers
WEIGHTS = 1_192_000_000               # 596.0M params, bf16
PEAK = 238e9                          # measured achievable read bandwidth
L = 28

# single-layer attention latency, benchmarks/benchmark_decode_attention.py
meas = {(1, 512): 22.5, (1, 2048): 61.4, (1, 4096): 111.6, (4, 2048): 188.4,
        (8, 512): 104.4, (8, 2048): 351.2, (8, 4096): 632.8, (16, 2048): 606.2}
# measured whole-step time, benchmarks/profile_decode_step.py
step = {(1, 10): 5.981, (8, 10): 6.349, (16, 10): 6.965, (8, 512): 8.121,
        (8, 2048): 14.530}

hdr = (f"{'batch':>5} {'ctx':>5} {'attn/step ms':>13} {'floor ms':>9} "
       f"{'attn%floor':>10} {'step ms':>8} {'attn%step':>9}")
print(hdr)
print("-" * len(hdr))
for (b, c), us in meas.items():
    attn_step = L * us / 1e3
    kv = b * c * KV_PER_TOKEN
    floor = (WEIGHTS + kv) / PEAK * 1e3
    s = step.get((b, c))
    sp = f"{s:8.2f}" if s else "       -"
    ap = f"{100 * attn_step / s:8.1f}%" if s else "        -"
    print(f"{b:>5} {c:>5} {attn_step:>13.2f} {floor:>9.2f} "
          f"{100 * attn_step / floor:>9.1f}% {sp} {ap}")

print()
print("short-context serving point (batch 8, ctx ~11): the attention kernel is")
print("~10us/layer at that shape, so 28 x 10us = 0.28 ms of a 6.35 ms step = "
      f"{100 * 28 * 10 / 1e3 / 6.349:.1f}%")
print()
print("sanity check against torch.profiler at batch 8 / ctx 2049:")
print("  flash_fwd_splitkv 97.410 ms over 12 steps = 8.12 ms/step")
print("  -> 8.12/14.53 = 55.9% of the step, and it moves 1.75 GiB of logical KV")
print(f"  -> {1.75 * 2**30 / (8.12e-3) / 1e9:.0f} GB/s = "
      f"{100 * 1.75 * 2**30 / 8.12e-3 / 1e9 / 238:.0f}% of measured peak")
PY
