#!/bin/bash
# DEFINITIVE comparison: nano-vllm vs vLLM 0.31.0, official usage only.
#
# Envelope (aligned so the comparison measures the implementations):
#   KV cache  : 144 nano-vllm blocks x 28 MiB = 4,227,858,432 B = 36,864 tokens
#               nano-vllm --num-kvcache-blocks 144   -> 144 x 256 = 36,864
#               vLLM      --kv-cache-memory 4227858432 -> 2304 x 16 = 36,864
#               (256 % 16 == 0, so byte equality gives EXACTLY equal capacity)
#
#   Why 144 and not the largest that fits: pushing the KV pool to the physical
#   limit (168 blocks / 4704 MiB) starved vLLM of the activation buffer its
#   16,384-token prefill needs and made its prefill 3x slower (TTFT 1839 ms,
#   996 tok/s) while decode stayed fine (TPOT 17.8 ms).  Measured at c=32:
#       168 blocks -> TTFT 1839.8 ms, TPOT 17.82 ms, 996.3 tok/s, peak 7859 MiB
#       144 blocks -> TTFT  611.3 ms, TPOT 17.33 ms, 1452.3 tok/s, peak 7430 MiB
#       136 blocks -> TTFT  644.9 ms, TPOT 17.09 ms, 1450.9 tok/s, peak 7408 MiB
#   vLLM sizes its own KV by profiling precisely to avoid this; nano-vllm does
#   not, so a forced byte-equal pool must stay inside what both can afford.
#   144 blocks leaves ~750 MiB free at peak for vLLM and ~2.1 GiB for nano-vllm.
#   max_num_batched_tokens 16384, max_num_seqs 512, max_model_len 4096 (both)
#   CUDA graphs on, bf16, prefix caching on (both defaults)
#   vLLM also needs --gpu-memory-utilization <= ~0.86 for its free-memory check;
#   the value does not affect the KV size once --kv-cache-memory is given (verified).
#
# Protocol: OFFICIAL entry points only.
#   offline -> vllm.LLM.generate() / nanovllm LLM.generate()   (throughput only)
#   online  -> unmodified `vllm serve` + OpenAI API; nano-vllm serve() + gRPC
# Workload: prompt 512 / output 128 token ids, ignore_eos, temperature 1.0.
# Every measurement uses a fresh random dataset -> prefix reuse is impossible
# (verified 0 hits in every run).
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

REPO=/mnt/c/Users/Adam/PycharmProjects/nano-vllm-plus
bash "$REPO/.tmp/sync_to_wsl.sh" >/dev/null
cd ~/nano-vllm-plus
mkdir -p results logs
# park anything from earlier runs so the new results are unambiguous
mkdir -p "results/prev_$(date +%H%M%S)"
mv results/*.json "results/prev_$(date +%H%M%S)/" 2>/dev/null || true

MODEL=/home/adam/models/Qwen3-0.6B
PROMPT=512
OUT=128
KV_BLOCKS=144
KV_BYTES=4227858432          # 144 * 29360128
UTIL=0.80
MBT=16384
MNS=512
MML=4096
OFF_REQS="32 128"
OFF_REPS=2
ON_REQS=64
ON_CONCS="1 8 32 48"
ON_REPS=2
VOCAB=$(python -c "from transformers import AutoTokenizer; print(AutoTokenizer.from_pretrained('$MODEL').vocab_size)")
echo "### prompt=$PROMPT out=$OUT kv=$KV_BLOCKS blocks/$KV_BYTES B (43008 tokens) vocab=$VOCAB"

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
kill_tree() { local pid=$1 k; k=$(pgrep -P "$pid" 2>/dev/null || true); for x in $k; do kill_tree "$x"; done; kill "$pid" 2>/dev/null || true; }
wait_free() { for _ in $(seq 1 60); do local u; u=$(gpu_used); [ "$u" -lt 900 ] && return 0; sleep 2; done; echo "!! gpu busy ${u}MiB"; }

cat > /tmp/c3_health.py <<'PY'
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
wait_grpc() { local pid=$1; for _ in $(seq 1 150); do python /tmp/c3_health.py >/dev/null 2>&1 && return 0
    kill -0 "$pid" 2>/dev/null || { echo "!! ours died"; tail -20 logs/ours_server.log; return 1; }; sleep 2; done; return 1; }
wait_http() { local url=$1 pid=$2 log=$3; for _ in $(seq 1 200); do curl -sf "$url" >/dev/null 2>&1 && return 0
    kill -0 "$pid" 2>/dev/null || { echo "!! vllm died"; tail -25 "$log"; return 1; }; sleep 2; done; return 1; }

# --- background GPU sampler -------------------------------------------------
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
  --format=csv,noheader -l 2 > logs/gpu_sampler.csv 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

########################################################################
echo; echo "############ 1/5  OFFLINE  nano-vllm ############"
python benchmarks/cmp_offline_nanovllm.py \
  --model "$MODEL" --requests $OFF_REQS --reps $OFF_REPS \
  --prompt-len "$PROMPT" --output-len "$OUT" \
  --num-kvcache-blocks "$KV_BLOCKS" --gpu-memory-utilization "$UTIL" \
  --max-num-batched-tokens "$MBT" --max-num-seqs "$MNS" --max-model-len "$MML" \
  --outdir results --label offline-ours
wait_free

echo; echo "############ 2/5  OFFLINE  vLLM ############"
python benchmarks/cmp_offline_vllm.py \
  --model "$MODEL" --requests $OFF_REQS --reps $OFF_REPS \
  --prompt-len "$PROMPT" --output-len "$OUT" \
  --kv-cache-bytes "$KV_BYTES" --gpu-memory-utilization "$UTIL" \
  --max-num-batched-tokens "$MBT" --max-num-seqs "$MNS" --max-model-len "$MML" \
  --outdir results --label offline-vllm
wait_free

########################################################################
echo; echo "############ 3/5  ONLINE  nano-vllm ############"
python benchmarks/cmp_serve_nanovllm.py \
  --model "$MODEL" --port 50051 --metrics-port 9100 --max-workers 256 --max-qps 10000 \
  --num-kvcache-blocks "$KV_BLOCKS" --gpu-memory-utilization "$UTIL" \
  --max-num-batched-tokens "$MBT" --max-num-seqs "$MNS" --max-model-len "$MML" \
  > logs/ours_server.log 2>&1 &
OURPID=$!
if wait_grpc "$OURPID"; then
  echo "ours ready (gpu=$(gpu_used) MiB)"
  grep -E "kv|block" logs/ours_server.log | tail -2
  for R in $(seq 0 $((ON_REPS-1))); do
    for C in $ON_CONCS; do
      python benchmarks/cmp_online_client.py --engine nanovllm --target localhost:50051 \
        --metrics-url http://localhost:9100/metrics --requests "$ON_REQS" \
        --concurrency "$C" --rep "$R" --prompt-len "$PROMPT" --output-len "$OUT" \
        --vocab-size "$VOCAB" --outdir results --label ours-online-c$C
    done
  done
else echo "!! skipping ours online"; fi
kill_tree "$OURPID"; sleep 5; wait_free

########################################################################
echo; echo "############ 4/5  ONLINE  vLLM ############"
VLOG="--no-enable-log-requests --disable-uvicorn-access-log"
vllm serve "$MODEL" --port 8000 --max-model-len $MML \
  --max-num-batched-tokens $MBT --max-num-seqs $MNS \
  --gpu-memory-utilization $UTIL --kv-cache-memory "$KV_BYTES" $VLOG \
  > logs/vllm_server.log 2>&1 &
VPID=$!
if wait_http http://localhost:8000/health "$VPID" logs/vllm_server.log; then
  echo "vllm ready (gpu=$(gpu_used) MiB)"
  grep -E "GPU KV cache size" logs/vllm_server.log | tail -1
  for R in $(seq 0 $((ON_REPS-1))); do
    for C in $ON_CONCS; do
      python benchmarks/cmp_online_client.py --engine vllm --target http://localhost:8000 \
        --metrics-url http://localhost:8000/metrics --requests "$ON_REQS" \
        --concurrency "$C" --rep "$R" --prompt-len "$PROMPT" --output-len "$OUT" \
        --vocab-size "$VOCAB" --outdir results --label vllm-online-c$C
    done
  done
else echo "!! skipping vllm online"; fi
kill_tree "$VPID"; sleep 5
pkill -f "vllm serve $MODEL" 2>/dev/null || true
sleep 5; wait_free

########################################################################
echo; echo "############ 5/5  CONTROL: vLLM at stock max_num_batched_tokens=2048 ############"
vllm serve "$MODEL" --port 8000 --max-model-len $MML \
  --max-num-batched-tokens 2048 --max-num-seqs $MNS \
  --gpu-memory-utilization $UTIL --kv-cache-memory "$KV_BYTES" $VLOG \
  > logs/vllm_server_stock.log 2>&1 &
SPID=$!
if wait_http http://localhost:8000/health "$SPID" logs/vllm_server_stock.log; then
  python benchmarks/cmp_online_client.py --engine vllm --target http://localhost:8000 \
    --metrics-url http://localhost:8000/metrics --requests "$ON_REQS" \
    --concurrency 32 --rep 0 --prompt-len "$PROMPT" --output-len "$OUT" \
    --vocab-size "$VOCAB" --outdir results --label vllm-stock2048-online-c32
else echo "!! skipping stock point"; fi
kill_tree "$SPID"; sleep 5
pkill -f "vllm serve $MODEL" 2>/dev/null || true
wait_free

kill $SAMPLER 2>/dev/null || true
echo; echo "############ ALL DONE ############"
ls results/*.json
echo "--- gpu sampler summary ---"
python - <<'PY'
import csv, statistics
rows = []
try:
    with open("logs/gpu_sampler.csv") as f:
        for r in csv.reader(f):
            if len(r) >= 3:
                try:
                    rows.append((float(r[1].strip().rstrip(" %")), float(r[2].strip().rstrip(" MiB"))))
                except ValueError:
                    pass
except FileNotFoundError:
    pass
if rows:
    util = [u for u, _ in rows]
    mem = [m for _, m in rows]
    print(f"samples={len(rows)} gpu_util mean={statistics.fmean(util):.1f}% max={max(util):.0f}% "
          f"| gpu_mem mean={statistics.fmean(mem):.0f}MiB max={max(mem):.0f}MiB")
PY
