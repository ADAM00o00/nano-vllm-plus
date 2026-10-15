#!/bin/bash
# Cheap introspection of vLLM APIs (no model load).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONUNBUFFERED=1

cat > /tmp/introspect.py <<'PY'
import inspect
from vllm import LLM, SamplingParams

print("=== LLM.generate ===")
print(inspect.signature(LLM.generate))
print(inspect.getdoc(LLM.generate)[:1500])
print()
print("=== LLM methods ===")
print([m for m in dir(LLM) if not m.startswith('_')])
print()

try:
    from vllm.v1.engine.llm_engine import LLMEngine
    print("=== LLMEngine.add_request ===")
    print(inspect.signature(LLMEngine.add_request))
    print("=== LLMEngine.step ===")
    print(inspect.signature(LLMEngine.step))
    print("=== LLMEngine methods ===")
    print([m for m in dir(LLMEngine) if not m.startswith('_')])
except Exception as e:
    print("LLMEngine introspect failed:", e)
print()

try:
    from vllm import AsyncLLM
    print("=== AsyncLLM ===")
    print(inspect.signature(AsyncLLM.__init__))
    print([m for m in dir(AsyncLLM) if not m.startswith('_')])
except Exception as e:
    print("AsyncLLM:", type(e).__name__, e)
print()

from vllm import SamplingParams as SP
print("=== SamplingParams ===")
print(inspect.signature(SP.__init__) if hasattr(SP, "__init__") else "")
try:
    print(SP.__struct_fields__ if hasattr(SP, "__struct_fields__") else "")
except Exception as e:
    print(e)
PY
python /tmp/introspect.py 2>&1 | grep -v -E "^(INFO|WARNING)" | head -80
