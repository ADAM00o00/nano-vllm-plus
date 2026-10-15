#!/bin/bash
# Instrumented gRPC server under load: how big is the batch, and how long is a step?
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

run_load () {
  local conc=$1
  local out=$2
  echo
  echo "################ instrumented server, client concurrency=$conc ################"
  python benchmarks/profile_serving_step.py --model /home/adam/models/Qwen3-0.6B \
      --port 50051 --metrics-port 9100 --write "$out" > /tmp/instr_server_$conc.log 2>&1 &
  local pid=$!
  for i in $(seq 1 90); do
    python /tmp/wait_health.py >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || { echo "server died:"; tail -25 /tmp/instr_server_$conc.log; return 1; }
    sleep 2
  done
  echo "warming up"
  python benchmarks/benchmark_grpc.py --target localhost:50051 --requests 8 --concurrency "$conc" --max-tokens 16
  echo "measuring"
  python benchmarks/benchmark_grpc.py --target localhost:50051 --requests 32 --concurrency "$conc" --max-tokens 32
  sleep 1
  kill -TERM "$pid" 2>/dev/null
  sleep 8
  kill -9 "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  python - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(json.dumps(d["summary"], indent=2))
PY
}

run_load 8 /tmp/serving_steps_c8.json
run_load 32 /tmp/serving_steps_c32.json
echo
echo "done"
