#!/bin/bash
# Quantify how much of a decode step is KV-cache traffic vs model weights.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd ~/nano-vllm-plus
python - <<'PY'
import glob, os
from transformers import AutoConfig

path = "/home/adam/models/Qwen3-0.6B"
cfg = AutoConfig.from_pretrained(path)
print("model:", path)
L = cfg.num_hidden_layers
Hq, Hkv = cfg.num_attention_heads, cfg.num_key_value_heads
D = getattr(cfg, "head_dim", cfg.hidden_size // Hq)
E = cfg.intermediate_size
V = cfg.vocab_size
hid = cfg.hidden_size

# Parameter count straight from the architecture (embeddings are tied here).
per_layer = hid * (Hq * D + 2 * Hkv * D) + (Hq * D) * hid + 3 * hid * E
params = per_layer * L + V * hid
disk = sum(os.path.getsize(p) for p in glob.glob(f"{path}/*.safetensors"))
print(f"layers={L} q_heads={Hq} kv_heads={Hkv} head_dim={D} inter={E} vocab={V}")
print(f"params ~= {params/1e6:.1f}M   bf16 weights = {params*2/2**30:.3f} GiB "
      f"(safetensors on disk {disk/2**30:.3f} GiB)")

kv_per_token = 2 * Hkv * D * 2 * L          # k+v, all layers, bytes
print(f"KV cache bytes per token per sequence = {kv_per_token/1024:.1f} KiB")

PEAK = 238e9
print(f"\nassume measured DRAM peak {PEAK/1e9:.0f} GB/s, bf16 everything\n")
print(f"{'batch':>5} {'ctx':>6} {'KV GiB':>8} {'weights GiB':>11} {'total GiB':>9} "
      f"{'floor ms':>9} {'attn us':>8} {'attn %':>7}")
# attention kernel latencies measured in benchmarks/benchmark_decode_attention.py
meas = {(1,512):22.5,(1,2048):61.4,(1,4096):111.6,(4,2048):188.4,(8,512):104.4,
        (8,2048):351.2,(8,4096):632.8,(16,512):185.3,(16,2048):606.2}
for (b, c), attn_us in meas.items():
    kv = b * c * kv_per_token
    w = params * 2
    floor = (kv + w) / PEAK * 1e3
    print(f"{b:>5} {c:>6} {kv/2**30:>8.3f} {w/2**30:>11.3f} {(kv+w)/2**30:>9.3f} "
          f"{floor:>9.2f} {attn_us:>8.1f} {100*attn_us*1e-3/floor:>6.2f}%")

# The shapes the gRPC benchmark actually exercises: 32 requests, 8 concurrent,
# ~10 prompt tokens + up to 32 generated.
print("\nserving benchmark operating point (concurrency 8, ctx ~10..42):")
for c in (16, 42):
    kv = 8 * c * kv_per_token
    w = params * 2
    floor = (kv + w) / PEAK * 1e3
    attn = 104.4 * (c / 512)   # scale the measured b8/ctx512 number
    print(f"  batch=8 ctx={c:<3} KV={kv/2**20:>7.1f} MiB  weights={w/2**30:.3f} GiB  "
          f"floor={floor:.2f} ms  attn~{attn:.1f} us ({100*attn*1e-3/floor:.2f}% of floor)")
PY
