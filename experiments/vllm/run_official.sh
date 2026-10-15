#!/bin/bash
# OFFICIAL bench.py workload (256 seqs, input 100-1024, output 100-1024,
# seed 0 -> 133,966 total output tokens) on BOTH engines, offline and online.
#
# Envelope identical to the agreed main table:
#   KV = 144 nano blocks x 28 MiB = 4,227,858,432 B = 36,864 tokens for both
#   max_num_batched_tokens 16384, max_num_seqs 512, max_model_len 4096,
#   gpu_memory_utilization 0.80, CUDA graphs on, bf16, official entry points only.
#
# The official dataset is FIXED, so a second run in the same process/server
# would read the prefix cache from the first.  => one measured run per process
# (offline) and one concurrency level per server start (online).
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

REPO=/mnt/c/Users/Adam/PycharmProjects/nano-vllm-plus
bash "$REPO/.tmp/sync_to_wsl.sh" >/dev/null
cd ~/nano-vllm-plus
mkdir -p results_official logs

MODEL=/home/adam/models/Qwen3-0.6B
KV_BLOCKS=144
KV_BYTES=4227858432
UTIL=0.80
MBT=16384
MNS=512
MML=4096
TEMP=0.6
WARM_SEQS=16
ON_CONCS="16 32"

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
kill_tree() { local pid=$1 k; k=$(pgrep -P "$pid" 2>/dev/null || true); for x in $k; do kill_tree "$x"; done; kill "$pid" 2>/dev/null || true; }
wait_free() { for _ in $(seq 1 60); do local u; u=$(gpu_used); [ "$u" -lt 900 ] && return 0; sleep 2; done; echo "!! gpu busy ${u}MiB"; }

cat > /tmp/of_health.py <<'PY'
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
wait_grpc() { local pid=$1; for _ in $(seq 1 200); do python /tmp/of_health.py >/dev/null 2>&1 && return 0
    kill -0 "$pid" 2>/dev/null || { echo "!! ours died"; tail -20 logs/of_ours_server.log; return 1; }; sleep 2; done; return 1; }
wait_http() { local url=$1 pid=$2 log=$3; for _ in $(seq 1 220); do curl -sf "$url" >/dev/null 2>&1 && return 0
    kill -0 "$pid" 2>/dev/null || { echo "!! vllm died"; tail -25 "$log"; return 1; }; sleep 2; done; return 1; }

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
  --format=csv,noheader -l 2 > logs/of_gpu_sampler.csv 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

########################################################################
echo; echo "############ 1/6  OFFLINE  nano-vllm  (official workload) ############"
python benchmarks/cmp_official_offline.py --engine nanovllm --model "$MODEL" \
  --temperature "$TEMP" --warmup-seqs "$WARM_SEQS" \
  --num-kvcache-blocks "$KV_BLOCKS" --gpu-memory-utilization "$UTIL" \
  --max-num-batched-tokens "$MBT" --max-num-seqs "$MNS" --max-model-len "$MML" \
  --outdir results_official --label official-offline-nanovllm
wait_free

echo; echo "############ 2/6  OFFLINE  vLLM  (official workload) ############"
python benchmarks/cmp_official_offline.py --engine vllm --model "$MODEL" \
  --temperature "$TEMP" --warmup-seqs "$WARM_SEQS" \
  --kv-cache-bytes "$KV_BYTES" --gpu-memory-utilization "$UTIL" \
  --max-num-batched-tokens "$MBT" --max-num-seqs "$MNS" --max-model-len "$MML" \
  --outdir results_official --label official-offline-vllm
wait_free

########################################################################
# one server start per (engine, concurrency) because the dataset is fixed
VLOG="--no-enable-log-requests --disable-uvicorn-access-log"

for C in $ON_CONCS; do
  echo; echo "############ ONLINE nano-vllm  c=$C  (official workload) ############"
  python benchmarks/cmp_serve_nanovllm.py --model "$MODEL" --port 50051 --metrics-port 9100 \
    --max-workers 256 --max-qps 10000 --high-watermark 0.98 --low-watermark 0.95 \
    --num-kvcache-blocks "$KV_BLOCKS" --gpu-memory-utilization "$UTIL" \
    --max-num-batched-tokens "$MBT" --max-num-seqs "$MNS" --max-model-len "$MML" \
    > logs/of_ours_server.log 2>&1 &
  OURPID=$!
  if wait_grpc "$OURPID"; then
    echo "ours ready (gpu=$(gpu_used) MiB)"
    python benchmarks/cmp_online_client.py --engine nanovllm --target localhost:50051 \
      --metrics-url http://localhost:9100/metrics --dataset official \
      --warmup-seqs "$WARM_SEQS" --concurrency "$C" --rep 0 \
      --output-len 0 --prompt-len 0 --vocab-size 151643 \
      --outdir results_official --label "official-online-nanovllm-c$C"
  else echo "!! skipping ours c=$C"; fi
  kill_tree "$OURPID"; sleep 5; wait_free
done

for C in $ON_CONCS; do
  echo; echo "############ ONLINE vLLM  c=$C  (official workload) ############"
  vllm serve "$MODEL" --port 8000 --max-model-len $MML \
    --max-num-batched-tokens $MBT --max-num-seqs $MNS \
    --gpu-memory-utilization $UTIL --kv-cache-memory "$KV_BYTES" $VLOG \
    > logs/of_vllm_server.log 2>&1 &
  VPID=$!
  if wait_http http://localhost:8000/health "$VPID" logs/of_vllm_server.log; then
    echo "vllm ready (gpu=$(gpu_used) MiB)"
    python benchmarks/cmp_online_client.py --engine vllm --target http://localhost:8000 \
      --metrics-url http://localhost:8000/metrics --dataset official \
      --warmup-seqs "$WARM_SEQS" --concurrency "$C" --rep 0 \
      --output-len 0 --prompt-len 0 --vocab-size 151643 \
      --outdir results_official --label "official-online-vllm-c$C"
  else echo "!! skipping vllm c=$C"; fi
  kill_tree "$VPID"; sleep 5
  pkill -f "vllm serve $MODEL" 2>/dev/null || true
  sleep 5; wait_free
done

kill $SAMPLER 2>/dev/null || true
echo; echo "############ ALL DONE ############"
ls -1 results_official/*.json
python - <<'PY'
import csv, statistics
rows = []
try:
    for r in csv.reader(open("logs/of_gpu_sampler.csv")):
        if len(r) >= 3:
            try:
                rows.append((float(r[1].strip().rstrip(" %")), float(r[2].strip().rstrip(" MiB"))))
            except ValueError:
                pass
except FileNotFoundError:
    pass
if rows:
    u = [a for a, _ in rows]; m = [b for _, b in rows]
    print(f"gpu sampler: samples={len(rows)} util mean={statistics.fmean(u):.1f}% max={max(u):.0f}% "
          f"| mem mean={statistics.fmean(m):.0f} max={max(m):.0f} MiB")
PY
