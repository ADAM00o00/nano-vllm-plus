# -*- coding: utf-8 -*-
"""Offline (batch) throughput benchmark for vLLM -- official usage only.

Uses ``vllm.LLM.generate()``, the documented offline API.  The earlier version
of this file reached into ``llm.llm_engine.add_request()/step()`` to synthesise a
per-request TTFT; that is not an official entry point, does not correspond to
anything a caller can observe offline, and has been removed.  Offline reports
throughput only.

    python benchmarks/cmp_offline_vllm.py --model /home/adam/models/Qwen3-0.6B \
        --requests 32 128 --prompt-len 512 --output-len 128 \
        --kv-cache-bytes 4932501504 --outdir results
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


def measure(llm, prompts, sp, tag):
    toks = [{"prompt_token_ids": ids} for ids in prompts]
    t0 = time.perf_counter()
    outs = llm.generate(toks, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    per_req = sorted(len(o.outputs[0].token_ids) for o in outs)
    gen = sum(per_req)
    prompt_tokens = sum(len(p) for p in prompts)
    return {
        "engine": "vllm",
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/adam/models/Qwen3-0.6B")
    p.add_argument("--requests", type=int, nargs="+", default=[32])
    p.add_argument("--prompt-len", type=int, default=DEFAULT_PROMPT_LEN)
    p.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN)
    p.add_argument("--warmup-rounds", type=int, default=2)
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--kv-cache-bytes", type=int, default=4932501504)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--outdir", default=None)
    p.add_argument("--label", default="offline-vllm")
    args = p.parse_args()

    from vllm import LLM, SamplingParams

    t_load = time.perf_counter()
    llm = LLM(args.model,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=args.max_num_batched_tokens,
              kv_cache_memory_bytes=args.kv_cache_bytes)
    load_s = time.perf_counter() - t_load
    cc = llm.llm_engine.vllm_config.cache_config
    kv_tokens = getattr(cc, "num_gpu_blocks", 0) * getattr(cc, "block_size", 0)
    print(f"[vllm] loaded in {load_s:.1f}s  num_gpu_blocks={cc.num_gpu_blocks} "
          f"block_size={cc.block_size} kv_tokens={kv_tokens}", flush=True)

    vocab_size = llm.get_tokenizer().vocab_size
    sp = SamplingParams(temperature=1.0, max_tokens=args.output_len,
                        ignore_eos=True, detokenize=False)

    results = []
    for rep in range(args.reps):
      for n in args.requests:
        for rnd in range(args.warmup_rounds):
            ws = prompt_seed("vllm", f"offline-warm-{n}", rnd)
            llm.generate([{"prompt_token_ids": ids} for ids in
                          build_prompts(n, args.prompt_len, ws, vocab_size)],
                         sp, use_tqdm=False)
            try:
                llm.reset_prefix_cache()
            except Exception:
                pass
        seed = prompt_seed("vllm", f"offline-{n}", 0) + rep
        prompts = build_prompts(n, args.prompt_len, seed, vocab_size)
        res = measure(llm, prompts, sp, f"offline-n{n}")
        res.update({
            "rep": rep,
            "load_s": load_s,
            "num_gpu_blocks": cc.num_gpu_blocks,
            "block_size": cc.block_size,
            "kv_tokens": kv_tokens,
            "kv_cache_bytes": args.kv_cache_bytes,
            "gpu_memory_utilization": args.gpu_memory_utilization,
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
        with open(os.path.join(args.outdir, "offline_vllm.json"), "w",
                  encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.outdir}/offline_vllm.json", flush=True)


if __name__ == "__main__":
    main()
