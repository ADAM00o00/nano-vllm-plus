#!/bin/bash
# Is the GPU actually busy during a step at high batch?
# nsys with node-level graph tracing gives every in-graph kernel; summing them and
# dividing by the measured step time gives the true GPU-busy fraction, which is
# independent of nvidia-smi sampling (unreliable on WSL).
export PYTHONDONTWRITEBYTECODE=1
PY=/home/adam/miniconda3/envs/ml/bin/python
cd /home/adam/nano-vllm-plus

for B in 8 64 128; do
  echo "================ batch=$B ================"
  rm -f "/tmp/busy_$B.nsys-rep"
  nsys profile -o "/tmp/busy_$B" --trace=cuda --cuda-graph-trace=node \
      --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite true \
      "$PY" benchmarks/profile_decode_step.py --batch "$B" --prompt-len 10 \
      --steps 12 --nsys-range 2>&1 | grep -E 'decode step GPU' | head -1
  # total kernel time in the captured window, from the CSV export
  nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true \
      "/tmp/busy_$B.nsys-rep" 2>/dev/null > "/tmp/busy_$B.csv"
  "$PY" - "$B" <<'PY'
import csv, sys
B = sys.argv[1]
tot = n = 0
with open(f"/tmp/busy_{B}.csv") as f:
    for r in csv.DictReader(f):
        try:
            tot += float(r["Total Time (ns)"]); n += int(r["Instances"])
        except Exception:
            pass
print(f"  summed GPU kernel time = {tot/1e6:.1f} ms over 12 steps, {n} kernel instances")
print(f"  -> {tot/12/1e6:.2f} ms of GPU work per step; "
      f"{n/12:.0f} kernels per step")
PY
done
