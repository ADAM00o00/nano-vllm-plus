#!/bin/bash
# Verify the memory-pressure hypothesis: vLLM's prefill collapses when the KV
# reservation leaves no room for the 16384-token prefill activation.
# Test smaller aligned KV sizes and watch TTFT at c=32 recover.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd ~/nano-vllm-plus
mkdir -p logs
MODEL=/home/adam/models/Qwen3-0.6B
kill_tree() { local pid=$1 k; k=$(pgrep -P "$pid" 2>/dev/null || true); for x in $k; do kill_tree "$x"; done; kill "$pid" 2>/dev/null || true; }
wait_free() { for _ in $(seq 1 60); do local u; u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "$u" -lt 900 ] && return 0; sleep 2; done; }

for B in 144 136; do
  BYTES=$(( B * 29360128 ))
  echo "=========== vLLM KV=$(( B*28 ))MiB ($B nano blocks = $(( B*256 )) tokens) ==========="
  vllm serve "$MODEL" --port 8000 --max-model-len 4096 \
    --max-num-batched-tokens 16384 --max-num-seqs 512 \
    --gpu-memory-utilization 0.80 --kv-cache-memory "$BYTES" \
    --no-enable-log-requests --disable-uvicorn-access-log \
    > "logs/vllm_kv${B}.log" 2>&1 &
  PID=$!
  ok=0
  for i in $(seq 1 120); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && { ok=1; break; }
    kill -0 "$PID" 2>/dev/null || break; sleep 2; done
  if [ "$ok" = 1 ]; then
    echo "startup gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    grep -E "GPU KV cache size" "logs/vllm_kv${B}.log" | tail -1
    python benchmarks/cmp_online_client.py --engine vllm --target http://localhost:8000 \
      --metrics-url http://localhost:8000/metrics --requests 64 --concurrency 32 --rep 0 \
      --prompt-len 512 --output-len 128 --vocab-size 151643 \
      --outdir results/probe_kv --label "vllm-kv${B}-c32"
    echo "peak gpu mem during run: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  else
    echo "FAILED to start"; tail -5 "logs/vllm_kv${B}.log"
  fi
  kill_tree "$PID"; sleep 4
  pkill -f "vllm serve $MODEL" 2>/dev/null || true
  wait_free
done
echo "done"
