# -*- coding: utf-8 -*-
"""Diagnose the n=8 offline TTFT anomaly: cold-start cost vs batch-size effect."""
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/adam/nano-vllm-plus")
from benchmarks.cmp_common import build_prompts, prompt_seed
from nanovllm import LLM, SamplingParams

MODEL = "/home/adam/models/Qwen3-0.6B"


def measure(engine, n, prompt_len, out_len, seed, tag):
    prompts = build_prompts(n, prompt_len, seed, 151643)
    sp = SamplingParams(temperature=1.0, max_tokens=out_len, ignore_eos=True)
    seqs = []
    step_times = []
    t0 = time.perf_counter()
    for ids in prompts:
        seqs.append(engine.add_request(ids, sp))
    first = {}
    prev = t0
    nsteps = 0
    while not engine.is_finished():
        engine.step()
        now = time.perf_counter()
        if nsteps < 4:
            step_times.append((nsteps, (now - prev) * 1000))
        nsteps += 1
        prev = now
        for s in seqs:
            if s.num_completion_tokens >= 1 and s.seq_id not in first:
                first[s.seq_id] = now - t0
    wall = time.perf_counter() - t0
    ttfts = sorted(first.values())
    print(f"[{tag}] n={n} out={out_len} steps={nsteps} wall={wall:.3f}s "
          f"TTFT mean={sum(ttfts)/len(ttfts)*1000:.1f}ms "
          f"first4_step_ms={[(i, round(ms,1)) for i, ms in step_times]}", flush=True)
    return ttfts[0]


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    v = tok.vocab_size
    print("vocab", v, flush=True)

    engine = LLM(MODEL, gpu_memory_utilization=0.80, max_num_batched_tokens=16384,
                 max_num_seqs=512, max_model_len=4096)
    print("loaded", flush=True)

    # (a) warm up with a *different batch size* than we measure
    measure(engine, 4, 512, 128, prompt_seed("d", "w4", 4), "warmup-n4")
    # (b) three consecutive n=8 measurements: is run 1 the slow one?
    for rep in range(3):
        measure(engine, 8, 512, 128, prompt_seed("d", "n8", rep), f"n8-rep{rep}")
    # (c) n=16 and n=32 for reference
    measure(engine, 16, 512, 128, prompt_seed("d", "n16", 0), "n16")
    measure(engine, 32, 512, 128, prompt_seed("d", "n32", 0), "n32")
    # (d) n=8 again after the big batches
    measure(engine, 8, 512, 128, prompt_seed("d", "n8", 99), "n8-after-big")


if __name__ == "__main__":
    main()
