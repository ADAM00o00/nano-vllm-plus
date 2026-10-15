#!/bin/bash
# P1 corrected: the first attempt was invalidated at concurrency 128/256 by the
# admissions QPS token bucket (default max_qps=100 -> RESOURCE_EXHAUSTED
# "backpressure: qps"). Raise it and re-measure the top end.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd ~/nano-vllm-plus
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
PY=/home/adam/miniconda3/envs/ml/bin/python

rm -f /tmp/markers2.jsonl /tmp/sampling2.jsonl /tmp/steps_p1b.json

cat > /tmp/sample2.py <<'PY'
import json, subprocess, sys, time, grpc
sys.path.insert(0, "/home/adam/nano-vllm-plus")
from nanovllm.serving import nanovllm_pb2, nanovllm_pb2_grpc
stub = nanovllm_pb2_grpc.NanoVLLMServiceStub(grpc.insecure_channel("localhost:50051"))
out = open("/tmp/sampling2.jsonl", "a")
while True:
    util = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                           "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
    try:
        h = stub.Health(nanovllm_pb2.HealthRequest(), timeout=2.0)
        rec = dict(t=time.monotonic(), util=util, running=h.num_running,
                   waiting=h.num_waiting, cache=h.cache_usage_ratio)
    except Exception:
        rec = dict(t=time.monotonic(), util=util)
    out.write(json.dumps(rec) + "\n"); out.flush()
    time.sleep(0.05)
PY

echo "### server: max_workers=256, max_num_seqs=256, max_qps=1000000 (the fix for this run)"
$PY benchmarks/profile_serving_step.py --model /home/adam/models/Qwen3-0.6B \
    --port 50051 --metrics-port 9100 --max-workers 256 --max-num-seqs 256 \
    --max-qps 1000000 \
    --write /tmp/steps_p1b.json > /tmp/server_p1b.log 2>&1 &
SERVER=$!
$PY - <<'PY'
import sys, time, grpc
sys.path.insert(0, "/home/adam/nano-vllm-plus")
from nanovllm.serving import nanovllm_pb2, nanovllm_pb2_grpc
for _ in range(150):
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
$PY /tmp/sample2.py &
SAMPLER=$!
sleep 1

for CONC in 64 128 256; do
  echo
  echo "==================== concurrency=$CONC ===================="
  $PY -c "import json,time;open('/tmp/markers2.jsonl','a').write(json.dumps(dict(phase='c$CONC',kind='start',t=time.monotonic()))+'\n')"
  $PY benchmarks/benchmark_grpc.py --target localhost:50051 \
      --requests 256 --concurrency "$CONC" --max-tokens 64 || echo "  (benchmark failed)"
  $PY -c "import json,time;open('/tmp/markers2.jsonl','a').write(json.dumps(dict(phase='c$CONC',kind='end',t=time.monotonic()))+'\n')"
  sleep 2
done

kill $SAMPLER 2>/dev/null
kill -TERM $SERVER 2>/dev/null
sleep 12
kill -9 $SERVER 2>/dev/null
wait $SERVER 2>/dev/null
echo
echo "############ per-phase analysis (corrected) ############"
$PY - <<'PY'
import json, statistics
steps = json.load(open("/tmp/steps_p1b.json"))["steps"]   # [t, host_ms, gpu_ms, batch]
marks = [json.loads(l) for l in open("/tmp/markers2.jsonl")]
try:
    samples = [json.loads(l) for l in open("/tmp/sampling2.jsonl")]
except FileNotFoundError:
    samples = []

phases, pending = [], {}
for m in marks:
    (pending.__setitem__(m["phase"], m["t"]) if m["kind"] == "start"
     else phases.append((m["phase"], pending.pop(m["phase"]), m["t"])))

hdr = (f"{'phase':>6} {'dec':>5} {'pre':>5} {'batch med/max':>13} {'step ms':>8} "
       f"{'tok/s(step)':>11} {'tok/s(wall)':>11} {'GPUeng%':>8} {'run':>6} {'wait':>6} {'cache':>6} {'smi%':>5}")
print(hdr); print("-" * len(hdr))
for name, t0, t1 in phases:
    ph = [s for s in steps if t0 <= s[0] <= t1]
    dec = [s for s in ph if s[3] > 0]
    pre = [s for s in ph if s[3] == 0]
    if not dec:
        print(f"{name:>6}  (no decode steps)"); continue
    batches = [s[3] for s in dec]
    toks = sum(batches)
    busy = sum(s[1] for s in dec) / 1e3            # engine-thread seconds
    wall = t1 - t0
    sm = [s for s in samples if t0 <= s["t"] <= t1]
    run = [s["running"] for s in sm if "running" in s]
    wait = [s["waiting"] for s in sm if "waiting" in s]
    cache = [s["cache"] for s in sm if "cache" in s]
    util = []
    for s in sm:
        try: util.append(int(s["util"].split(",")[0]))
        except Exception: pass
    print(f"{name:>6} {len(dec):>5} {len(pre):>5} "
          f"{statistics.median(batches):>6.0f}/{max(batches):<6} "
          f"{statistics.median([s[2] for s in dec]):>8.2f} "
          f"{toks/busy:>11.0f} {toks/wall:>11.0f} "
          f"{100*sum(s[2] for s in dec)/1e3/wall:>7.0f}% "
          f"{statistics.fmean(run) if run else float('nan'):>6.1f} "
          f"{statistics.fmean(wait) if wait else float('nan'):>6.1f} "
          f"{statistics.fmean(cache) if cache else float('nan'):>6.3f} "
          f"{statistics.fmean(util) if util else float('nan'):>5.0f}")
print()
print("tok/s(step) = tokens / engine-thread busy time    <- how fast the engine itself goes")
print("tok/s(wall) = tokens / phase wall time            <- what a client sees (incl. turnaround)")
print("GPUeng%     = sum(step gpu_ms) / wall             <- wall time 'engaged' by a decode step")
PY
