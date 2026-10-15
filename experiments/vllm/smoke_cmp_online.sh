#!/bin/bash
# Quick smoke test of both ONLINE paths (server + streaming client).
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

bash /mnt/c/Users/Adam/PycharmProjects/nano-vllm-plus/.tmp/sync_to_wsl.sh >/dev/null
cd ~/nano-vllm-plus
mkdir -p results logs

MODEL=/home/adam/models/Qwen3-0.6B
VOCAB=$(python -c "from transformers import AutoTokenizer; print(AutoTokenizer.from_pretrained('$MODEL').vocab_size)")

kill_tree() { local pid=$1; local k; k=$(pgrep -P "$pid" 2>/dev/null || true); for x in $k; do kill_tree "$x"; done; kill "$pid" 2>/dev/null || true; }

echo "===== nano-vllm online smoke ====="
cat > /tmp/smoke_health.py <<'PY'
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
python benchmarks/cmp_serve_nanovllm.py --model "$MODEL" --port 50051 --metrics-port 9100 \
  --max-workers 64 --max-qps 10000 --gpu-memory-utilization 0.80 \
  > logs/smoke_ours_server.log 2>&1 &
OURPID=$!
ok=0
for i in $(seq 1 150); do
  if python /tmp/smoke_health.py >/dev/null 2>&1; then ok=1; break; fi
  if ! kill -0 "$OURPID" 2>/dev/null; then echo "server died"; tail -20 logs/smoke_ours_server.log; break; fi
  sleep 2
done
if [ "$ok" = 1 ]; then
  python benchmarks/cmp_online_client.py --engine nanovllm --target localhost:50051 \
    --metrics-url http://localhost:9100/metrics --requests 8 --concurrency 4 \
    --prompt-len 512 --output-len 32 --vocab-size "$VOCAB" --warmup 2 \
    --out results/smoke_ours_online.json --label smoke-online-ours
else
  echo "!! ours online smoke skipped"
fi
kill_tree "$OURPID"; sleep 6

echo; echo "===== vLLM online smoke ====="
vllm serve "$MODEL" --port 8000 --gpu-memory-utilization 0.80 --max-model-len 4096 \
  --max-num-batched-tokens 16384 --max-num-seqs 512 \
  --no-enable-log-requests --disable-uvicorn-access-log \
  > logs/smoke_vllm_server.log 2>&1 &
VPID=$!
ok=0
for i in $(seq 1 200); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then ok=1; break; fi
  if ! kill -0 "$VPID" 2>/dev/null; then echo "vllm died"; tail -25 logs/smoke_vllm_server.log; break; fi
  sleep 2
done
if [ "$ok" = 1 ]; then
  curl -s http://localhost:8000/v1/models | head -c 400; echo
  python benchmarks/cmp_online_client.py --engine vllm --target http://localhost:8000 \
    --metrics-url http://localhost:8000/metrics --requests 8 --concurrency 4 \
    --prompt-len 512 --output-len 32 --vocab-size "$VOCAB" --warmup 2 \
    --out results/smoke_vllm_online.json --label smoke-online-vllm
  echo "--- prefix cache metrics ---"
  curl -s http://localhost:8000/metrics | grep -i prefix_cache | head -10
else
  echo "!! vllm online smoke skipped"
fi
kill_tree "$VPID"; sleep 5
pkill -f "vllm serve $MODEL" 2>/dev/null || true
echo "smoke done"
