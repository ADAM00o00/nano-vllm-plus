#!/bin/bash
# Smoke test v2: vLLM offline API at gpu_memory_utilization=0.8 (desktop shares the GPU).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONUNBUFFERED=1
cd ~/nano-vllm-plus

cat > /tmp/smoke_vllm2.py <<'PY'
import time

def main():
    from vllm import LLM, SamplingParams
    MODEL = "/home/adam/models/Qwen3-0.6B"
    t0 = time.perf_counter()
    llm = LLM(MODEL, max_model_len=4096, gpu_memory_utilization=0.80,
              enforce_eager=False)
    print(f"== loaded in {time.perf_counter()-t0:.1f}s ==", flush=True)

    sp = SamplingParams(temperature=1.0, max_tokens=8, ignore_eos=True)
    outs = llm.generate(["Hello, tell me a story about", "Count from one to"], sp)
    o = outs[0]
    print("type:", type(o).__mro__)
    print("attrs:", [a for a in dir(o) if not a.startswith('_')])
    m = getattr(o, "metrics", None)
    print("metrics:", type(m), m)
    if m is not None:
        print("metrics attrs:", [a for a in dir(m) if not a.startswith('_')])
        for a in dir(m):
            if a.startswith('_'): continue
            try: print(f"   m.{a} = {getattr(m, a)}")
            except Exception as e: print(f"   m.{a} err {e}")
    print("prompt_token_ids len:", len(o.prompt_token_ids or []))

    # token-id prompts
    outs2 = llm.generate(prompt_token_ids=[[1000]*16, [2000]*16], sampling_params=sp)
    print("token-id prompt ok, len:", len(outs2[0].prompt_token_ids or []))

if __name__ == "__main__":
    main()
PY
python /tmp/smoke_vllm2.py 2>&1 | grep -v -E "^(INFO|WARNING)" | tail -35
