#!/bin/bash
# Does serving throughput scale with concurrency the way the engine can?
#
# The engine stepped directly does batch 8 in 6.35 ms -> 1250 tok/s. The gRPC
# benchmark reported 775 tok/s at concurrency 8. This sweep separates "the
# engine cannot batch" from "the serving layer adds per-step overhead".
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

python -m nanovllm.serving --model /home/adam/models/Qwen3-0.6B \
    --port 50051 --metrics-port 9100 > /tmp/server_scale.log 2>&1 &
PID=$!
for i in $(seq 1 90); do
  python /tmp/wait_health.py >/dev/null 2>&1 && break
  kill -0 "$PID" 2>/dev/null || { echo "server died"; tail -20 /tmp/server_scale.log; exit 1; }
  sleep 2
done
echo "server up (pid $PID)"

for c in 1 2 4 8 16 32; do
  echo
  echo "================ concurrency=$c ================"
  for run in 1 2; do
    python benchmarks/benchmark_grpc.py --target localhost:50051 \
        --requests 32 --concurrency "$c" --max-tokens 32
  done
done

kill "$PID" 2>/dev/null
wait "$PID" 2>/dev/null
echo
echo "done"
