#!/bin/bash
# Online end-to-end gRPC benchmark for both decode-attention backends.
# Plan section 5.6 step 5 / milestone M6.
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
    r = nanovllm_pb2_grpc.NanoVLLMServiceStub(ch).Health(
        nanovllm_pb2.HealthRequest(), timeout=5.0)
    print(f"ready={r.ready} cache_usage={r.cache_usage_ratio:.3f}")
    sys.exit(0 if r.ready else 1)
except Exception as e:
    print("not up:", type(e).__name__)
    sys.exit(1)
PY

run_backend() {
  local backend=$1
  echo "=============================================================="
  echo "backend=$backend"
  echo "=============================================================="
  NANOVLLM_DECODE_ATTN=$backend \
    python -m nanovllm.serving --model /home/adam/models/Qwen3-0.6B \
      --port 50051 --metrics-port 9100 > "/tmp/server_$backend.log" 2>&1 &
  local pid=$!

  local up=0
  for i in $(seq 1 90); do
    if python /tmp/wait_health.py >/dev/null 2>&1; then up=1; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "SERVER DIED during startup; tail of log:"
      tail -25 "/tmp/server_$backend.log"
      return 1
    fi
    sleep 2
  done
  if [ "$up" != 1 ]; then
    echo "server never became ready; tail of log:"
    tail -25 "/tmp/server_$backend.log"
    kill "$pid" 2>/dev/null
    return 1
  fi
  python /tmp/wait_health.py

  # Plan section 5.6 / 13.3 baseline invocation.
  python benchmarks/benchmark_grpc.py --target localhost:50051 \
      --requests 32 --concurrency 8 --max-tokens 32
  echo "-- second run (verifies stability) --"
  python benchmarks/benchmark_grpc.py --target localhost:50051 \
      --requests 32 --concurrency 8 --max-tokens 32

  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  sleep 5
}

run_backend flash_attn
run_backend triton
echo "done"
