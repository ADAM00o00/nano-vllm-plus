#!/bin/bash
# Definitive run: all points with the fixed warmup protocol
# (2 warmup rounds at the exact measured shape, on datasets that are never
# the measured one).  Overwrites results/*.json.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

REPO=/mnt/c/Users/Adam/PycharmProjects/nano-vllm-plus
bash "$REPO/.tmp/sync_to_wsl.sh" >/dev/null
cd ~/nano-vllm-plus
mkdir -p results logs
# preserve the contaminated cold-start measurement as evidence
mkdir -p results/cold_start
[ -f results/ours_offline_n8.json ] && cp results/ours_offline_n8.json results/cold_start/
[ -f results/ours_offline_n16.json ] && cp results/ours_offline_n16.json results/cold_start/

MODEL=/home/adam/models/Qwen3-0.6B
PROMPT=512; OUT=128; GPU=0.80; MBT=16384; MNS=512; MML=4096
VOCAB=$(python -c "from transformers import AutoTokenizer; print(AutoTokenizer.from_pretrained('$MODEL').vocab_size)")
COMMON="--model $MODEL --prompt-len $PROMPT --output-len $OUT --max-model-len $MML \
  --gpu-memory-utilization $GPU --max-num-batched-tokens $MBT --max-num-seqs $MNS --warmup 4"
echo "### vocab=$VOCAB  prompt=$PROMPT out=$OUT  aligned envelope mbt=$MBT mns=$MNS gpu=$GPU"

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
kill_tree() { local pid=$1 k; k=$(pgrep -P "$pid" 2>/dev/null || true); for x in $k; do kill_tree "$x"; done; kill "$pid" 2>/dev/null || true; }
wait_free() { for _ in $(seq 1 60); do local u; u=$(gpu_used); [ "$u" -lt 900 ] && return 0; sleep 2; done; echo "!! gpu busy ${u}MiB"; }

cat > /tmp/c2_health.py <<'PY'
import sys, grpc
sys.path.insert(0, "/home/adam/nano-vllm-plus")
from nanovllm.serving import nanovllm_pb2, nanovllm_pb2_grpc
try:
    r = nanovllm_pb2_grpc.NanoVLLMServiceStub(grpc.insecure_channel("localhost:50051")).Health(
        nanovllm_pb2.HealthRequest(), timeout=5.0)
    sys.exit(0 if r.ready else 1)
except Exception:
    sys.exit(1)
PY
wait_grpc() { local pid=$1; for _ in $(seq 1 150); do python /tmp/c2_health.py >/dev/null 2>&1 && return 0
    kill -0 "$pid" 2>/dev/null || { echo "!! ours died"; tail -20 logs/ours_server.log; return 1; }; sleep 2; done; return 1; }
wait_http() { local url=$1 pid=$2 log=$3; for _ in $(seq 1 200); do curl -sf "$url" >/dev/null 2>&1 && return 0
    kill -0 "$pid" 2>/dev/null || { echo "!! vllm died"; tail -25 "$log"; return 1; }; sleep 2; done; return 1; }

########################################################################
echo; echo "### OFFLINE curve (n=8,16,32)"
for N in 8 16 32; do
  python benchmarks/cmp_offline_nanovllm.py $COMMON --requests "$N" \
    --out "results/ours_offline_n${N}.json" --label "offline-ours-n$N"
  wait_free
  python benchmarks/cmp_offline_vllm.py $COMMON --requests "$N" \
    --out "results/vllm_offline_n${N}.json" --label "offline-vllm-n$N"
  wait_free
done
# keep the plain names the report/README refer to (n=32 is the headline batch)
cp results/ours_offline_n32.json results/ours_offline.json
cp results/vllm_offline_n32.json results/vllm_offline.json

########################################################################
echo; echo "### ONLINE sweep, nano-vllm"
python benchmarks/cmp_serve_nanovllm.py --model "$MODEL" --port 50051 --metrics-port 9100 \
  --max-workers 256 --max-qps 10000 --max-num-batched-tokens "$MBT" --max-num-seqs "$MNS" \
  --max-model-len "$MML" --gpu-memory-utilization "$GPU" > logs/ours_server.log 2>&1 &
OURPID=$!
if wait_grpc "$OURPID"; then
  echo "ours ready (gpu=$(gpu_used) MiB)"
  for C in 1 8 32; do
    python benchmarks/cmp_online_client.py --engine nanovllm --target localhost:50051 \
      --metrics-url http://localhost:9100/metrics --requests 64 --concurrency "$C" \
      --prompt-len "$PROMPT" --output-len "$OUT" --vocab-size "$VOCAB" \
      --out "results/ours_online_c${C}.json" --label "online-ours-c$C"
  done
else echo "!! skipping ours online"; fi
kill_tree "$OURPID"; sleep 5; wait_free

########################################################################
echo; echo "### ONLINE sweep, vLLM"
VLOG="--no-enable-log-requests --disable-uvicorn-access-log"
vllm serve "$MODEL" --port 8000 --gpu-memory-utilization $GPU --max-model-len $MML \
  --max-num-batched-tokens $MBT --max-num-seqs $MNS $VLOG > logs/vllm_server.log 2>&1 &
VPID=$!
if wait_http http://localhost:8000/health "$VPID" logs/vllm_server.log; then
  echo "vllm ready (gpu=$(gpu_used) MiB)"
  for C in 1 8 32; do
    python benchmarks/cmp_online_client.py --engine vllm --target http://localhost:8000 \
      --metrics-url http://localhost:8000/metrics --requests 64 --concurrency "$C" \
      --prompt-len "$PROMPT" --output-len "$OUT" --vocab-size "$VOCAB" \
      --out "results/vllm_online_c${C}.json" --label "online-vllm-c$C"
  done
else echo "!! skipping vllm online"; fi
kill_tree "$VPID"; sleep 5
pkill -f "vllm serve $MODEL" 2>/dev/null || true
sleep 5; wait_free

########################################################################
echo; echo "### BONUS: vLLM at its stock max_num_batched_tokens=2048"
vllm serve "$MODEL" --port 8000 --gpu-memory-utilization $GPU --max-model-len $MML \
  --max-num-batched-tokens 2048 --max-num-seqs $MNS $VLOG > logs/vllm_server_stock.log 2>&1 &
SPID=$!
if wait_http http://localhost:8000/health "$SPID" logs/vllm_server_stock.log; then
  python benchmarks/cmp_online_client.py --engine vllm --target http://localhost:8000 \
    --metrics-url http://localhost:8000/metrics --requests 64 --concurrency 32 \
    --prompt-len "$PROMPT" --output-len "$OUT" --vocab-size "$VOCAB" \
    --out results/vllm_stock_online_c32.json --label "stock-2048-c32"
else echo "!! skipping stock point"; fi
kill_tree "$SPID"; sleep 5
pkill -f "vllm serve $MODEL" 2>/dev/null || true
wait_free

echo; echo "### ALL DONE"
ls results/*.json
