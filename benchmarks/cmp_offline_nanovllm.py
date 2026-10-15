# -*- coding: utf-8 -*-
"""Offline (batch) throughput benchmark for nano-vllm -- official usage only.

Uses ``LLM.generate()``, i.e. exactly the API a user would call for offline
batch inference.  Per the agreed protocol an offline run reports **throughput
only**: there is no client-observable time-to-first-token offline (all requests
are submitted at once and the caller receives complete outputs), and the only
"first token" timestamp you could extract is a single batch-level number whose
distribution is degenerate.

    python benchmarks/cmp_offline_nanovllm.py --model /home/adam/models/Qwen3-0.6B \
        --requests 32 128 --prompt-len 512 --output-len 128 \
        --num-kvcache-blocks 168 --outdir results
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.cmp_common import (  # noqa: E402
    DEFAULT_OUTPUT_LEN,
    DEFAULT_PROMPT_LEN,
    build_prompts,
    prompt_seed,
)

from nanovllm import LLM, SamplingParams  # noqa: E402


def measure(llm, prompts, output_len, tag):
    sp = SamplingParams(temperature=1.0, max_tokens=output_len, ignore_eos=True)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    gen = sum(len(o["token_ids"]) for o in outs)
    prompt_tokens = sum(len(p) for p in prompts)
    per_req = sorted(len(o["token_ids"]) for o in outs)
    res = {
        "engine": "nanovllm",
        "mode": "offline",
        "tag": tag,
        "n_requests": len(prompts),
        "prompt_len": len(prompts[0]),
        "prompt_tokens": prompt_tokens,
        "gen_tokens": gen,
        "tokens_per_request_min": per_req[0],
        "tokens_per_request_max": per_req[-1],
        "wall_s": wall,
        "gen_throughput_tok_s": gen / wall,
        "total_throughput_tok_s": (gen + prompt_tokens) / wall,
        "e2e_s_per_request_mean": wall / len(prompts),
    }
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/adam/models/Qwen3-0.6B")
    p.add_argument("--requests", type=int, nargs="+", default=[32])
    p.add_argument("--prompt-len", type=int, default=DEFAULT_PROMPT_LEN)
    p.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN)
    p.add_argument("--warmup-rounds", type=int, default=2)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--num-kvcache-blocks", type=int, default=168)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--outdir", default=None)
    p.add_argument("--label", default="offline-ours")
    args = p.parse_args()

    from transformers import AutoTokenizer
    vocab_size = AutoTokenizer.from_pretrained(args.model).vocab_size

    t_load = time.perf_counter()
    llm = LLM(args.model,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=args.max_num_batched_tokens,
              num_kvcache_blocks=args.num_kvcache_blocks)
    load_s = time.perf_counter() - t_load
    blocks = llm.scheduler.block_manager.num_blocks
    print(f"[nanovllm] loaded in {load_s:.1f}s  kv_blocks={blocks} "
          f"kv_tokens={blocks * 256} max_num_batched_tokens={args.max_num_batched_tokens}",
          flush=True)

    results = []
    for rep in range(args.reps):
      for n in args.requests:
        # Warm up at the exact shape, twice: the engine's own warmup_model()
        # runs before allocate_kv_cache(), so it never touches the real paged
        # prefill path and the first genuine prefill is ~16x slower.
        for rnd in range(args.warmup_rounds):
            ws = prompt_seed("nanovllm", f"offline-warm-{n}", rnd)
            measure(llm, build_prompts(n, args.prompt_len, ws, vocab_size),
                    args.output_len, "warmup")
        seed = prompt_seed("nanovllm", f"offline-{n}", 0) + rep
        prompts = build_prompts(n, args.prompt_len, seed, vocab_size)
        res = measure(llm, prompts, args.output_len, f"offline-n{n}")
        res.update({
            "rep": rep,
            "load_s": load_s,
            "kv_blocks": blocks,
            "kv_tokens": blocks * 256,
            "block_size": llm.scheduler.block_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "num_kvcache_blocks": args.num_kvcache_blocks,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_model_len": args.max_model_len,
            "dataset_seed": seed,
        })
        print("RESULT " + args.label + f"-n{n}-rep{rep} " + json.dumps(
            {k: res[k] for k in ("n_requests", "wall_s", "gen_throughput_tok_s",
                                 "total_throughput_tok_s")}), flush=True)
        results.append(res)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        with open(os.path.join(args.outdir, "offline_nanovllm.json"), "w",
                  encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.outdir}/offline_nanovllm.json", flush=True)


if __name__ == "__main__":
    main()
