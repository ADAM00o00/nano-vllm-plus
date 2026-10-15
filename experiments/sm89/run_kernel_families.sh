#!/bin/bash
# Full per-kernel table for the decode step, with shares of total GPU kernel time.
export PYTHONDONTWRITEBYTECODE=1
PY=/home/adam/miniconda3/envs/ml/bin/python
for B in 8 64; do
echo "================ batch=$B ================"
nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
    "/tmp/final_b$B.nsys-rep" 2>/dev/null > "/tmp/kern_b$B.csv"
$PY - "$B" <<'PY'
import csv, sys, collections
B = sys.argv[1]
rows = []
with open(f"/tmp/kern_b{B}.csv") as f:
    for r in csv.DictReader(f):
        try:
            t = float(r["Total Time (ns)"])
            n = int(r["Instances"])
        except (KeyError, ValueError):
            continue
        rows.append((t, n, r["Name"]))
total = sum(t for t, _, _ in rows)
# Group the many triton_* / cutlass variants into families for readability.
def family(name):
    if "flash" in name: return "flash-attn (in-graph)"
    if "gemm" in name or "cutlass" in name or "Kernel2" in name: return "GEMM (cutlass/ampere)"
    if "store_kvcache" in name: return "store_kvcache"
    if "rsqrt" in name: return "RMSNorm (triton)"
    if "silu" in name: return "SiLU+Mul (triton)"
    if "split_sub" in name or "cat_index" in name: return "RoPE (triton)"
    if "softmax" in name or "argmax" in name: return "sampler (triton)"
    if "memcpy" in name or "Memcpy" in name: return "memcpy"
    return "other"
agg = collections.defaultdict(lambda: [0, 0])
for t, n, name in rows:
    a = agg[family(name)]
    a[0] += t; a[1] += n
print(f"total GPU kernel time in window: {total/1e6:.1f} ms over 12 steps "
      f"= {total/12/1e6:.2f} ms/step (nsys adds overhead vs 6.4/11.2 ms real)")
print(f"{'share':>7} {'total ms':>9} {'inst/step':>10}  family")
for fam, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    print(f"{100*t/total:>6.1f}% {t/1e6:>9.1f} {n/12:>10.1f}  {fam}")
print("  top individual kernels:")
for t, n, name in sorted(rows, key=lambda x: -x[0])[:6]:
    print(f"    {100*t/total:>5.1f}% {t/1e6:>7.1f} ms {n/12:>6.1f}/step  {name[:88]}")
PY
echo
done
