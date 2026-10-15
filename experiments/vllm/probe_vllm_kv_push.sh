#!/bin/bash
# Push vLLM's KV as high as it goes while keeping a real safety margin.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd ~/nano-vllm-plus
mkdir -p logs
MODEL=/home/adam/models/Qwen3-0.6B
kill_tree() { local pid=$1 k; k=$(pgrep -P "$pid" 2>/dev/null || true); for x in $k; do kill_tree "$x"; done; kill "$pid" 2>/dev/null || true; }

run_one() {
  local B=$1 UTIL=$2
  local BYTES=$(( B * 29360128 ))
  echo "=== vLLM KV=$(( B * 28 ))MiB ($B nano blocks) util=$UTIL ==="
  vllm serve "$MODEL" --port 8000 --max-model-len 4096 \
    --max-num-batched-tokens 16384 --max-num-seqs 512 \
    --gpu-memory-utilization "$UTIL" --kv-cache-memory "$BYTES" \
    --no-enable-log-requests --disable-uvicorn-access-log \
    > "logs/probe_vllm_kv${B}_u${UTIL}.log" 2>&1 &
  local PID=$!
  local ok=0
  for i in $(seq 1 90); do
    curl -sf http://localhost:8000/health >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "$PID" 2>/dev/null || break
    sleep 2
  done
  if [ "$ok" = 1 ]; then
    echo "STARTED-OK; gpu: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
    grep -E "GPU KV cache size" "logs/probe_vllm_kv${B}_u${UTIL}.log" | tail -1
  else
    echo "FAILED"; grep -E "ValueError|RuntimeError|out of memory|No available memory" "logs/probe_vllm_kv${B}_u${UTIL}.log" | tail -3
  fi
  kill_tree "$PID"; sleep 4
  pkill -f "vllm serve $MODEL" 2>/dev/null || true
  for i in $(seq 1 40); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$u" -lt 900 ] && break; sleep 2
  done
}

run_one 168 0.86
run_one 176 0.86
run_one 168 0.80
echo "done"
