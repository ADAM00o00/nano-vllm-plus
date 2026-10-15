#!/bin/bash
# Hypothesis: grpc_server's ThreadPoolExecutor(max_workers=10) caps the number of
# concurrent Infer streams at 10, which caps the engine's batch at ~10, which caps
# throughput at ~800 tok/s -- even though the engine does 3596 tok/s at batch 32.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd ~/nano-vllm-plus
export PYTHONUNBUFFERED=1

echo "cpu cores available to WSL: $(nproc)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

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

for workers in 10 64; do
  echo
  echo "=============================================================="
  echo "### server --max-workers $workers"
  echo "=============================================================="
  python -m nanovllm.serving --model /home/adam/models/Qwen3-0.6B \
      --port 50051 --metrics-port 9100 --max-workers "$workers" > /tmp/server_w$workers.log 2>&1 &
  PID=$!
  for i in $(seq 1 90); do
    python /tmp/wait_health.py >/dev/null 2>&1 && break
    kill -0 "$PID" 2>/dev/null || { echo "server died:"; tail -20 /tmp/server_w$workers.log; break; }
    sleep 2
  done
  for c in 8 16 32 64; do
    echo "--- concurrency=$c ---"
    python benchmarks/benchmark_grpc.py --target localhost:50051 \
        --requests 64 --concurrency "$c" --max-tokens 32
  done
  kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
  sleep 4
done
echo
echo "done"
