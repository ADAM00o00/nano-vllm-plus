#!/bin/bash
# One-shot: nsys profile -> CSV export -> analysis, all in a single invocation and
# writing intermediates under the repo (WSL /tmp is being wiped between calls).
export PYTHONDONTWRITEBYTECODE=1
PY=/home/adam/miniconda3/envs/ml/bin/python
WORK=/home/adam/nano-vllm-plus/.tmp/nsyswork
mkdir -p "$WORK"
cd /home/adam/nano-vllm-plus

for B in 8 64; do
  echo "================ batch=$B ================"
  nsys profile -o "$WORK/b$B" --trace=cuda --cuda-graph-trace=node \
      --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true \
      "$PY" benchmarks/profile_decode_step.py --batch "$B" --prompt-len 10 \
      --steps 12 --nsys-range 2>&1 | grep -E 'decode step GPU' | head -1
  nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
      "$WORK/b$B.nsys-rep" 2>/dev/null > "$WORK/b$B.csv"
done

$PY - "$WORK" <<'PY'
import csv, sys, collections
WORK = sys.argv[1]
# nsys prints progress lines on stdout before the CSV header; start there.
def load(b):
    lines = open(f"{WORK}/b{b}.csv").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Time (%)"))
    rows = []
    for r in csv.DictReader(lines[start:]):
        try:
            rows.append((float(r["Total Time (ns)"]), int(r["Instances"]), r["Name"]))
        except (TypeError, ValueError):
            pass
    return rows

# mean step times printed by the run above (nsys inflates them; we divide the
# summed kernel time by the SAME run's step times so the ratio stays meaningful)
step_mean = {8: 7.383, 64: 22.314}
def family(n):
    if "flash" in n: return "flash attention"
    if "gemm" in n or "cutlass" in n or "Kernel2" in n: return "GEMM"
    if "store_kvcache" in n: return "store_kvcache"
    if "rsqrt" in n: return "RMSNorm"
    if "silu" in n: return "SiLU+Mul"
    if "split_sub" in n or "cat_index" in n: return "RoPE"
    if "softmax" in n or "argmax" in n: return "sampler"
    if "emcpy" in n: return "memcpy"
    return "other"

for b in (8, 64):
    rows = load(b)
    tot = sum(t for t, _, _ in rows)
    inst = sum(n for _, n, _ in rows)
    kern = tot / 1e6 / 12
    print(f"\nbatch {b}: {kern:.2f} ms GPU kernel time/step vs {step_mean[b]:.2f} ms step "
          f"-> GPU busy {100*kern/step_mean[b]:.0f}%   ({inst/12:.0f} kernels/step)")
    agg = collections.defaultdict(lambda: [0, 0])
    for t, n, name in rows:
        a = agg[family(name)]; a[0] += t; a[1] += n
    for fam, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        if n == 0: continue
        print(f"    {100*t/tot:>5.1f}%  {t/1e6/12:>7.2f} ms/step  {n/12:>6.0f}/step  {fam}")
PY
