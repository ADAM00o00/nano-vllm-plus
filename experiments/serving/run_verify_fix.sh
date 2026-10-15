#!/bin/bash
# Verify the fix through the shipped default CLI (no --max-workers flag), and
# check nothing regressed. Also produce an nsys trace of the serving path.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd ~/nano-vllm-plus
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

echo "############ full test suite ############"
python -m unittest discover -s tests 2>&1 | tail -5

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

echo
echo "############ default CLI (no --max-workers): online benchmark ############"
python -m nanovllm.serving --model /home/adam/models/Qwen3-0.6B \
    --port 50051 --metrics-port 9100 > /tmp/server_default.log 2>&1 &
PID=$!
for i in $(seq 1 90); do
  python /tmp/wait_health.py >/dev/null 2>&1 && break
  kill -0 "$PID" 2>/dev/null || { echo "died:"; tail -20 /tmp/server_default.log; exit 1; }
  sleep 2
done
for c in 8 32 64; do
  echo "--- concurrency=$c (requests=64, max-tokens=32) ---"
  python benchmarks/benchmark_grpc.py --target localhost:50051 \
      --requests 64 --concurrency "$c" --max-tokens 32
done
kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
sleep 5

echo
echo "############ nsys trace of the serving path (batch 64 load) ############"
python -m nanovllm.serving --model /home/adam/models/Qwen3-0.6B \
    --port 50051 --metrics-port 9100 > /tmp/server_nsys.log 2>&1 &
PID=$!
for i in $(seq 1 90); do
  python /tmp/wait_health.py >/dev/null 2>&1 && break
  sleep 2
done
# Attach nsys to the already-running server by profiling a fresh one is not
# possible, so instead trace a short client run against it is also not useful.
# Profile the *engine in-process* instead: same kernels, no gRPC noise.
kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
sleep 3

nsys profile -o /tmp/serving_decode --trace=cuda,nvtx --force-overwrite true \
    python benchmarks/profile_decode_step.py --batch 64 --prompt-len 10 --steps 12 \
    2>&1 | tail -3
echo "--- kernel summary ---"
nsys stats --report cuda_gpu_kern_sum /tmp/serving_decode.nsys-rep 2>&1 | tail -18
