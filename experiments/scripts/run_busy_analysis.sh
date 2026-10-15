#!/bin/bash
# Corrected analysis of the already-collected nsys kernel CSVs.
# nsys prints progress lines on stdout before the CSV header, which silently
# broke csv.DictReader (header taken from line 1 -> every row skipped).
export PYTHONDONTWRITEBYTECODE=1
PY=/home/adam/miniconda3/envs/ml/bin/python
$PY - <<'PY'
import csv, statistics, collections

# mean step times measured in the same nsys runs (nsys inflates these)
step_mean = {8: 7.383, 64: 22.314, 128: 23.550}
step_med = {8: 7.339, 64: 24.686, 128: 24.063}
STEPS = 12

def load(b):
    rows = []
    with open(f"/tmp/busy_{b}.csv") as f:
        lines = f.read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Time (%)"))
    for r in csv.DictReader(lines[start:]):
        try:
            rows.append((float(r["Total Time (ns)"]), int(r["Instances"]), r["Name"]))
        except (TypeError, ValueError):
            pass
    return rows

def family(n):
    if "flash" in n: return "flash attention"
    if "gemm" in n or "cutlass" in n or "Kernel2" in n: return "GEMM"
    if "store_kvcache" in n: return "store_kvcache"
    if "rsqrt" in n: return "RMSNorm"
    if "silu" in n: return "SiLU+Mul"
    if "split_sub" in n or "cat_index" in n: return "RoPE"
    if "softmax" in n or "argmax" in n: return "sampler"
    if "Memcpy" in n or "memcpy" in n: return "memcpy"
    return "other"

print(f"{'batch':>6} {'kern ms/step':>13} {'step ms(mean)':>14} {'GPU busy':>9} "
      f"{'kernels/step':>13}")
print("-" * 62)
for b in (8, 64, 128):
    rows = load(b)
    tot_ns = sum(t for t, _, _ in rows)
    inst = sum(n for _, n, _ in rows)
    kern_ms = tot_ns / 1e6 / STEPS
    print(f"{b:>6} {kern_ms:>13.2f} {step_mean[b]:>14.2f} "
          f"{100*kern_ms/step_mean[b]:>8.0f}% {inst/STEPS:>13.0f}")

print()
for b in (8, 64, 128):
    rows = load(b)
    tot = sum(t for t, _, _ in rows)
    agg = collections.defaultdict(lambda: [0, 0])
    for t, n, name in rows:
        a = agg[family(name)]; a[0] += t; a[1] += n
    print(f"batch {b}: shares of {tot/1e6:.0f} ms GPU kernel time")
    for fam, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        if n == 0: continue
        print(f"    {100*t/tot:>5.1f}%  {t/1e6/STEPS:>7.2f} ms/step  {n/STEPS:>6.0f}/step  {fam}")
PY
