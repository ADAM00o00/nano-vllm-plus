#!/bin/bash
# P1: where did the remaining 1.9x between achieved (2930 tok/s) and engine
# capability (5690 tok/s at batch 64) go?
#
# One instrumented server, five load phases. Each phase is bracketed by a
# CLOCK_MONOTONIC stamp from the driver (system-wide on Linux, so comparable with
# the stamps the server records per step). During each phase we also sample
# GPU utilisation and the engine's own Health RPC (num_running / num_waiting /
# cache_usage_ratio), because the ceiling may be the KV pool rather than
# max_workers.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd ~/nano-vllm-plus
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
PY=/home/adam/miniconda3/envs/ml/bin/python

rm -f /tmp/markers.jsonl /tmp/sampling.jsonl /tmp/steps_p1.json

cat > /tmp/sample.py <<'PY'
import json, sys, time, grpc
sys.path.insert(0, "/home/adam/nano-vllm-plus")
from nanovllm.serving import nanovllm_pb2, nanovllm_pb2_grpc
ch = grpc.insecure_channel("localhost:50051")
stub = nanovllm_pb2_grpc.NanoVLLMServiceStub(ch)
out = open("/tmp/sampling.jsonl", "a")
while True:
    import subprocess
    util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                           "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
    try:
        h = stub.Health(nanovllm_pb2.HealthRequest(), timeout=2.0)
        rec = dict(t=time.monotonic(), util=util, running=h.num_running,
                   waiting=h.num_waiting, cache=h.cache_usage_ratio)
    except Exception as e:
        rec = dict(t=time.monotonic(), util=util, err=type(e).__name__)
    out.write(json.dumps(rec) + "\n"); out.flush()
    time.sleep(0.1)
PY

echo "### starting instrumented server: max_workers=256, max_num_seqs=256, max_qps unbounded"
$PY benchmarks/profile_serving_step.py --model /home/adam/models/Qwen3-0.6B \
    --port 50051 --metrics-port 9100 --max-workers 256 --max-num-seqs 256 \
    --write /tmp/steps_p1.json > /tmp/server_p1.log 2>&1 &
SERVER=$!

$PY - <<'PY'
import sys, time, grpc
sys.path.insert(0, "/home/adam/nano-vllm-plus")
from nanovllm.serving import nanovllm_pb2, nanovllm_pb2_grpc
for _ in range(120):
    try:
        r = nanovllm_pb2_grpc.NanoVLLMServiceStub(grpc.insecure_channel("localhost:50051")).Health(
            nanovllm_pb2.HealthRequest(), timeout=5.0)
        if r.ready:
            print("server ready"); break
    except Exception: pass
    time.sleep(2)
else:
    sys.exit("server never became ready")
PY

$PY /tmp/sample.py &
SAMPLER=$!
sleep 1

for CONC in 8 32 64 128 256; do
  echo
  echo "==================== concurrency=$CONC ===================="
  $PY - "$CONC" <<'PY'
import json, sys, time
open("/tmp/markers.jsonl", "a").write(json.dumps(
    dict(phase=f"c{sys.argv[1]}", kind="start", t=time.monotonic())) + "\n")
PY
  $PY benchmarks/benchmark_grpc.py --target localhost:50051 \
      --requests 256 --concurrency "$CONC" --max-tokens 64
  $PY - "$CONC" <<'PY'
import json, sys, time
open("/tmp/markers.jsonl", "a").write(json.dumps(
    dict(phase=f"c{sys.argv[1]}", kind="end", t=time.monotonic())) + "\n")
PY
  sleep 2
done

kill $SAMPLER 2>/dev/null
kill -TERM $SERVER 2>/dev/null
sleep 10
kill -9 $SERVER 2>/dev/null
wait $SERVER 2>/dev/null
echo
echo "############ per-phase analysis ############"
$PY - <<'PY'
import json, statistics, collections
steps = json.load(open("/tmp/steps_p1.json"))["steps"]   # [t, host_ms, gpu_ms, batch]
marks = [json.loads(l) for l in open("/tmp/markers.jsonl")]
samples = []
try:
    samples = [json.loads(l) for l in open("/tmp/sampling.jsonl")]
except FileNotFoundError:
    pass

phases = []
pending = {}
for m in marks:
    if m["kind"] == "start":
        pending[m["phase"]] = m["t"]
    else:
        phases.append((m["phase"], pending.pop(m["phase"]), m["t"]))

hdr = (f"{'phase':>7} {'steps':>6} {'batch med/max':>14} {'step ms':>8} "
       f"{'tok/s(step)':>12} {'running':>8} {'waiting':>8} {'cache':>7} {'GPU%':>6}")
print(hdr); print("-" * len(hdr))
for name, t0, t1 in phases:
    ph = [s for s in steps if t0 <= s[0] <= t1]
    dec = [s for s in ph if s[3] > 0]
    if not dec:
        print(f"{name:>7} {len(ph):>6}  (no decode steps)"); continue
    batches = [s[3] for s in dec]
    gpu = [s[2] for s in dec]
    med_gpu = statistics.median(gpu)
    toks = sum(batches) / (sum(s[1] for s in dec) / 1e3)  # tokens / engine-thread time
    sm = [s for s in samples if t0 <= s["t"] <= t1]
    run = [s["running"] for s in sm if "running" in s]
    wait = [s["waiting"] for s in sm if "waiting" in s]
    cache = [s["cache"] for s in sm if "cache" in s]
    util = []
    for s in sm:
        try: util.append(int(s["util"].split(",")[0]))
        except Exception: pass
    print(f"{name:>7} {len(dec):>6} {statistics.median(batches):>7.0f}/{max(batches):<6} "
          f"{med_gpu:>8.2f} {toks:>12.0f} "
          f"{statistics.fmean(run) if run else float('nan'):>8.1f} "
          f"{statistics.fmean(wait) if wait else float('nan'):>8.1f} "
          f"{statistics.fmean(cache) if cache else float('nan'):>7.3f} "
          f"{statistics.fmean(util) if util else float('nan'):>6.0f}")
print()
print("tok/s(step) = tokens / engine-thread busy time (excludes driver-side gaps)")
print("running/waiting/cache = engine Health RPC averages during the phase")
PY
