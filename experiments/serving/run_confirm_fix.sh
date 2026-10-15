#!/bin/bash
# Confirm the mechanism: with --max-workers 64 the engine should now batch widely,
# and the per-step time should scale with batch instead of being pinned at ~8.5ms.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd ~/nano-vllm-plus
export PYTHONUNBUFFERED=1

cat > /tmp/wait_health.py <<'PY'
import sys, grpc
sys.path.insert(0, "/home/adam/nano-vllm-plus")
from nanovllm.serving import nanovllm_pb2, nanovllm_pb2_grpc
ch = grpc.insecure_channel("localhost:50051")
try:
    r = nanovllm_pb2_grpc.NanoVLLMServiceStub(ch).Health(nanovllm_pb2.HealthRequest(), timeout=5.0)
    sys.exit(0 if r.ready else 1)
except Exception:
    sys.exit(1)
PY

run () {
  local workers=$1 conc=$2 out=$3
  echo
  echo "################ max-workers=$workers, client concurrency=$conc ################"
  python benchmarks/profile_serving_step.py --model /home/adam/models/Qwen3-0.6B \
      --port 50051 --metrics-port 9100 --max-workers "$workers" --write "$out" \
      > /tmp/instr_w${workers}_c${conc}.log 2>&1 &
  local pid=$!
  for i in $(seq 1 90); do
    python /tmp/wait_health.py >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || { echo "died:"; tail -20 /tmp/instr_w${workers}_c${conc}.log; return 1; }
    sleep 2
  done
  python benchmarks/benchmark_grpc.py --target localhost:50051 --requests 16 --concurrency "$conc" --max-tokens 16
  # sample GPU utilisation during the measured run
  ( for i in $(seq 1 40); do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; sleep 0.1; done > /tmp/gpu_util_${workers}_${conc}.txt ) &
  local smi=$!
  python benchmarks/benchmark_grpc.py --target localhost:50051 --requests 64 --concurrency "$conc" --max-tokens 32
  wait "$smi" 2>/dev/null
  sleep 1
  kill -TERM "$pid" 2>/dev/null; sleep 8; kill -9 "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  python - "$out" /tmp/gpu_util_${workers}_${conc}.txt <<'PY'
import json, statistics, sys
d = json.load(open(sys.argv[1]))["summary"]
b = d["batch"]
print(f"  decode steps={d['decode_steps']}  batch mean={b['mean']:.1f} median={b['median']} max={b['max']}")
print(f"  batch histogram (batch:count)={b['histogram']}")
print(f"  step host median={d['host_ms']['median']:.2f} ms  gpu median={d['gpu_ms']['median']:.2f} ms")
print(f"  engine busy fraction={d['engine_busy_fraction_of_step_time']:.3f}")
try:
    u = [int(x) for x in open(sys.argv[2]).read().split()]
    print(f"  GPU util during run: mean={statistics.fmean(u):.0f}% max={max(u)}%")
except Exception as e:
    print("  gpu util unavailable:", e)
PY
}

run 64 16 /tmp/final_w64_c16.json
run 64 64 /tmp/final_w64_c64.json
echo
echo "done"
