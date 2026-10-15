#!/bin/bash
# End-to-end validation of the integrated decode-attention backend.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd ~/nano-vllm-plus
export PYTHONUNBUFFERED=1

echo "############ unit tests (decode attention) ############"
python -m unittest tests.test_decode_attention 2>&1 | tail -5

echo
echo "############ full test suite ############"
python -m unittest discover -s tests 2>&1 | tail -5

cat > /tmp/e2e_check.py <<'PY'
import json, os, sys, time, torch
sys.path.insert(0, "/home/adam/nano-vllm-plus")
from nanovllm import LLM, SamplingParams
from nanovllm.layers.attention import DECODE_ATTENTION_BACKEND

torch.manual_seed(0)
backend = DECODE_ATTENTION_BACKEND
eager = os.environ.get("EAGER") == "1"
print(f"backend={backend} enforce_eager={eager}")
llm = LLM("/home/adam/models/Qwen3-0.6B", enforce_eager=eager)
# The repo forbids true greedy sampling, so use a temperature small enough to be
# effectively argmax: that makes the two backends' token streams comparable.
sp = SamplingParams(temperature=1e-6, max_tokens=32)
prompts = [
    "The capital of France is",
    "1 + 1 =",
    "def fib(n):",
    "Explain briefly why the sky is blue:",
    "List three prime numbers:",
]
outs = llm.generate(prompts, sp, use_tqdm=False)
result = {}
for p, o in zip(prompts, outs):
    result[p] = o["token_ids"]
    print(f"  {p!r} -> {o['text']!r}")
print("  peak mem MiB:", torch.cuda.max_memory_allocated() // 2**20)
with open(f"/tmp/e2e_{backend}_{'eager' if eager else 'graph'}.json", "w") as f:
    json.dump(result, f)
PY

for mode in graph eager; do
  for backend in flash_attn triton; do
    echo
    echo "---- backend=$backend mode=$mode ----"
    if [ "$mode" = eager ]; then
      NANOVLLM_DECODE_ATTN=$backend EAGER=1 python /tmp/e2e_check.py 2>&1 | tail -12
    else
      NANOVLLM_DECODE_ATTN=$backend python /tmp/e2e_check.py 2>&1 | tail -12
    fi
  done
done

echo
echo "############ token-stream agreement between backends ############"
python - <<'PY'
import json
for mode in ("graph", "eager"):
    a = json.load(open(f"/tmp/e2e_flash_attn_{mode}.json"))
    b = json.load(open(f"/tmp/e2e_triton_{mode}.json"))
    same = sum(1 for k in a if a[k] == b[k])
    print(f"{mode}: {same}/{len(a)} prompts produced identical token ids")
    for k in a:
        if a[k] != b[k]:
            n = min(len(a[k]), len(b[k]))
            first = next((i for i in range(n) if a[k][i] != b[k][i]), n)
            print(f"  differs at token {first}: {k!r}")
PY
