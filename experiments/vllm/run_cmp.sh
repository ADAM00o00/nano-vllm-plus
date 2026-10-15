#!/bin/bash
# Head-to-head: nano-vllm vs vLLM 0.31.0 on the same GPU / model / workload.
#
# Envelope aligned across both engines so the comparison measures the
# implementation, not a config default:
#   gpu_memory_utilization=0.80   (desktop shares the 8 GB GPU; 0.9 is refused)
#   max_num_batched_tokens=16384  (vLLM's own default here is 2048)
#   max_num_seqs=512              (vLLM's own default here is 128)
#   max_model_len=4096            (nano-vllm's default)
#   CUDA graphs on (neither engine in eager mode)
# Workload: prompt=512 tokens, output=128 tokens, ignore_eos=True,
# temperature=1.0, raw token ids (no tokenizer in the path).
# Every run draws a fresh random dataset, so prefixes never repeat.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

REPO=/mnt/c/Users/Adam/PycharmProjects/nano-vllm-plus
bash "$REPO/.tmp/sync_to_wsl.sh" >/dev/null
cd ~/nano-vllm-plus
mkdir -p results logs

MODEL=/home/adam/models/Qwen3-0.6B
PROMPT=512
OUT=128
GPU=0.80
MBT=16384
MNS=512
MML=4096
REQS_OFF=32
REQS_ON=64
CONCS="1 8 32"
VOCAB=$(python -c "from transformers import AutoTokenizer; print(AutoTokenizer.from_pretrained('$MODEL').vocab_size)")
echo "### vocab_size=$VOCAB prompt=$PROMPT out=$OUT off_n=$REQS_OFF on_n=$REQS_ON conc='$CONCS'"

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

kill_tree() {
  local pid=$1
  local kids
  kids=$(pgrep -P "$pid" 2>/dev/null || true)
  for k in $kids; do kill_tree "$k"; done
  kill "$pid" 2>/dev/null || true
}

wait_gpu_free() {
  for _ in $(seq 1 60); do
    local used; used=$(gpu_used)
    if [ "$used" -lt 900 ]; then return 0; fi
    sleep 2
  done
  echo "!! GPU still holds ${used} MiB"
}

cat > /tmp/cmp_health_grpc.py <<'PY'
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

wait_grpc() {
  local pid=$1
  for _ in $(seq 1 150); do
    if python /tmp/cmp_health_grpc.py >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "!! nanovllm server died"; tail -20 logs/ours_server.log; return 1; fi
    sleep 2
  done
  echo "!! nanovllm server never became ready"; tail -20 logs/ours_server.log; return 1
}

wait_http() {
  local url=$1 pid=$2 log=$3
  for _ in $(seq 1 200); do
    if curl -sf "$url" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "!! vllm server died"; tail -25 "$log"; return 1; fi
    sleep 2
  done
  echo "!! vllm server never became ready"; tail -25 "$log"; return 1
}

run_online_sweep() {
  local engine=$1 target=$2 metrics=$3 tag=$4
  for C in $CONCS; do
    python benchmarks/cmp_online_client.py \
      --engine "$engine" --target "$target" --metrics-url "$metrics" \
      --requests "$REQS_ON" --concurrency "$C" \
      --prompt-len "$PROMPT" --output-len "$OUT" --vocab-size "$VOCAB" \
      --out "results/${tag}_online_c${C}.json" --label "$tag"
  done
}

########################################################################
echo; echo "=========== 1/4  OFFLINE  nano-vllm ==========="
python benchmarks/cmp_offline_nanovllm.py \
  --model "$MODEL" --requests "$REQS_OFF" --prompt-len "$PROMPT" --output-len "$OUT" \
  --gpu-memory-utilization "$GPU" --max-num-batched-tokens "$MBT" --max-num-seqs "$MNS" \
  --max-model-len "$MML" --out results/ours_offline.json --label offline-ours
wait_gpu_free

echo; echo "=========== 2/4  OFFLINE  vLLM ==========="
python benchmarks/cmp_offline_vllm.py \
  --model "$MODEL" --requests "$REQS_OFF" --prompt-len "$PROMPT" --output-len "$OUT" \
  --gpu-memory-utilization "$GPU" --max-num-batched-tokens "$MBT" --max-num-seqs "$MNS" \
  --max-model-len "$MML" --out results/vllm_offline.json --label offline-vllm
wait_gpu_free

########################################################################
echo; echo "=========== 3/4  ONLINE  nano-vllm ==========="
python benchmarks/cmp_serve_nanovllm.py \
  --model "$MODEL" --port 50051 --metrics-port 9100 --max-workers 256 --max-qps 10000 \
  --max-num-batched-tokens "$MBT" --max-num-seqs "$MNS" --max-model-len "$MML" \
  --gpu-memory-utilization "$GPU" > logs/ours_server.log 2>&1 &
OURPID=$!
if wait_grpc "$OURPID"; then
  echo "ours server ready (pid=$OURPID), gpu=$(gpu_used) MiB"
  run_online_sweep nanovllm localhost:50051 http://localhost:9100/metrics ours
else
  echo "!! skipping nano-vllm online sweep"
fi
kill_tree "$OURPID"
sleep 5
wait_gpu_free

########################################################################
echo; echo "=========== 4/4  ONLINE  vLLM ==========="
VLLM_LOG="--no-enable-log-requests --disable-uvicorn-access-log"
vllm serve "$MODEL" --port 8000 --gpu-memory-utilization $GPU --max-model-len $MML \
  --max-num-batched-tokens $MBT --max-num-seqs $MNS $VLLM_LOG > logs/vllm_server.log 2>&1 &
VPID=$!
if wait_http http://localhost:8000/health "$VPID" logs/vllm_server.log; then
  echo "vllm server ready (pid=$VPID), gpu=$(gpu_used) MiB"
  run_online_sweep vllm http://localhost:8000 http://localhost:8000/metrics vllm
else
  echo "!! skipping vLLM online sweep"
fi
kill_tree "$VPID"
sleep 5
# vLLM's EngineCore lives in its own process group; make sure it is gone.
pkill -f "vllm serve $MODEL" 2>/dev/null || true
sleep 5
wait_gpu_free

########################################################################
echo; echo "=========== BONUS  vLLM at its stock max_num_batched_tokens=2048 ==========="
vllm serve "$MODEL" --port 8000 --gpu-memory-utilization $GPU --max-model-len $MML \
  --max-num-batched-tokens 2048 --max-num-seqs $MNS $VLLM_LOG \
  > logs/vllm_server_stock.log 2>&1 &
SPID=$!
if wait_http http://localhost:8000/health "$SPID" logs/vllm_server_stock.log; then
  python benchmarks/cmp_online_client.py \
    --engine vllm --target http://localhost:8000 --metrics-url http://localhost:8000/metrics \
    --requests "$REQS_ON" --concurrency 32 \
    --prompt-len "$PROMPT" --output-len "$OUT" --vocab-size "$VOCAB" \
    --out results/vllm_stock_online_c32.json --label stock-2048-c32
else
  echo "!! skipping stock vLLM point"
fi
kill_tree "$SPID"
sleep 5
pkill -f "vllm serve $MODEL" 2>/dev/null || true
wait_gpu_free

echo; echo "=========== ALL DONE ==========="
ls -la results/
