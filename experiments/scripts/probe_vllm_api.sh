#!/bin/bash
# Probe vLLM 0.31.0 API surface for the comparison benchmark.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONUNBUFFERED=1
cd ~/nano-vllm-plus

cat > /tmp/probe_vllm_api.py <<'PY'
import inspect
import vllm
print("vllm", vllm.__version__)

from vllm import LLM, SamplingParams
print("LLM sig:", inspect.signature(LLM.__init__))
print()
print("SamplingParams fields:", sorted(SamplingParams.__dataclass_fields__.keys()) if hasattr(SamplingParams, "__dataclass_fields__") else "n/a")
print()

# metrics surface
try:
    from vllm.outputs import RequestOutput, CompletionOutput
    print("RequestOutput fields:", [f for f in dir(RequestOutput) if not f.startswith("_")])
except Exception as e:
    print("RequestOutput import failed:", e)

try:
    import vllm.v1.metrics.stats as st
    print("stats module:", [n for n in dir(st) if not n.startswith("_")])
except Exception as e:
    print("stats import failed:", e)

for modname, attr in [("vllm.metrics", "Metrics"), ("vllm.v1.metrics.loggers", None),
                      ("vllm.engine.metrics_types", None), ("vllm.v1.metrics.reader", None)]:
    try:
        m = __import__(modname, fromlist=["*"])
        print(f"OK {modname}")
    except Exception as e:
        print(f"-- {modname}: {type(e).__name__}")

# CLI / server entrypoints
import shutil
print()
print("vllm exe:", shutil.which("vllm"))
PY
python /tmp/probe_vllm_api.py 2>&1 | tail -40

echo "=== vllm serve --help (first 60 lines) ==="
vllm serve --help 2>&1 | head -60
