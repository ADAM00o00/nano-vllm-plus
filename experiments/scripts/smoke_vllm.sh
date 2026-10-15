#!/bin/bash
# Smoke test: vLLM offline LLM API + metric surface that exposes TTFT.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
export PYTHONUNBUFFERED=1
cd ~/nano-vllm-plus

cat > /tmp/smoke_vllm.py <<'PY'
import time

def main():
    from vllm import LLM, SamplingParams
    MODEL = "/home/adam/models/Qwen3-0.6B"
    t0 = time.perf_counter()
    llm = LLM(MODEL, max_model_len=4096, gpu_memory_utilization=0.90,
              enforce_eager=False, disable_log_stats=False)
    print(f"== engine loaded in {time.perf_counter()-t0:.1f}s ==", flush=True)

    sp = SamplingParams(temperature=1.0, max_tokens=8, ignore_eos=True)
    outs = llm.generate(["Hello, tell me a story about", "Count from one to"], sp)
    print("num outputs:", len(outs))
    o = outs[0]
    print("type:", type(o))
    print("repr:", repr(o)[:700])
    print("attrs:", [a for a in dir(o) if not a.startswith('_')])
    for a in ("metrics", "request_id", "outputs", "prompt_token_ids", "finished"):
        print(f"  {a} =", getattr(o, a, "<missing>"))
    m = getattr(o, "metrics", None)
    if m is not None:
        print("metrics type:", type(m))
        print("metrics attrs:", [a for a in dir(m) if not a.startswith('_')])
        print("metrics repr:", repr(m)[:900])
        for a in ("first_token_time", "first_token_latency", "arrival_time",
                  "queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts"):
            print(f"    m.{a} =", getattr(m, a, "<missing>"))
    co = o.outputs[0] if getattr(o, "outputs", None) else None
    if co is not None:
        print("completion attrs:", [a for a in dir(co) if not a.startswith('_')])
        print("  cum_logprob=", getattr(co, "cumulative_logprob", None),
              "len token_ids=", len(getattr(co, "token_ids", []) or []))

if __name__ == "__main__":
    main()
PY
python /tmp/smoke_vllm.py 2>&1 | grep -v -E "^(INFO|WARNING)" | tail -40
