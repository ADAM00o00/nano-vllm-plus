# -*- coding: utf-8 -*-
"""Offline throughput on the OFFICIAL bench.py workload, both engines.

Protocol notes:
* Dataset = `benchmarks/cmp_official_common.build_official_workload()`, which
  reproduces `bench.py` seed-for-seed (verified: sum(max_tokens) == 133,966,
  matching the README table).
* Reports **total output tokens** and **throughput = output_tokens / wall**,
  exactly the README's definition.  This is offline mode, so no TTFT.
* The workload is FIXED (seed 0), so a second run in the same process would hit
  the prefix cache.  One measured run per process; repeat by restarting.
* Warmup uses a DIFFERENT dataset (seed 1) so it cannot warm the official
  prompts into the prefix cache.  Without any shape-relevant warmup the first
  real paged prefill costs ~2.6 s (see docs/cmp_vllm_vs_nanovllm_worklog.md).

    python benchmarks/cmp_official_offline.py --engine nanovllm \
        --num-kvcache-blocks 144 --outdir results_official
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.cmp_official_common import (  # noqa: E402
    build_official_workload,
    describe,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["nanovllm", "vllm"], required=True)
    p.add_argument("--model", default="/home/adam/models/Qwen3-0.6B")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--warmup-seqs", type=int, default=16)
    p.add_argument("--num-kvcache-blocks", type=int, default=144)
    p.add_argument("--kv-cache-bytes", type=int, default=4227858432)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--outdir", default=None)
    p.add_argument("--label", default="")
    args = p.parse_args()

    prompts, out_lens = build_official_workload()
    info = describe(prompts, out_lens)

    if args.engine == "nanovllm":
        from nanovllm import LLM, SamplingParams
        t0 = time.perf_counter()
        llm = LLM(args.model,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  max_model_len=args.max_model_len,
                  max_num_seqs=args.max_num_seqs,
                  max_num_batched_tokens=args.max_num_batched_tokens,
                  num_kvcache_blocks=args.num_kvcache_blocks)
        cfg = {"num_kvcache_blocks": args.num_kvcache_blocks,
               "kv_tokens": llm.scheduler.block_manager.num_blocks
                            * llm.scheduler.block_size,
               "block_size": llm.scheduler.block_size}
    else:
        from vllm import LLM, SamplingParams
        t0 = time.perf_counter()
        llm = LLM(args.model,
                  max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  max_num_seqs=args.max_num_seqs,
                  max_num_batched_tokens=args.max_num_batched_tokens,
                  kv_cache_memory_bytes=args.kv_cache_bytes)
        cc = llm.llm_engine.vllm_config.cache_config
        cfg = {"num_gpu_blocks": cc.num_gpu_blocks, "block_size": cc.block_size,
               "kv_tokens": cc.num_gpu_blocks * cc.block_size}
    load_s = time.perf_counter() - t0
    print(f"[{args.engine}] loaded in {load_s:.1f}s {cfg}", flush=True)

    def run(pr, ol):
        if args.engine == "nanovllm":
            sps = [SamplingParams(temperature=args.temperature, max_tokens=n,
                                  ignore_eos=True) for n in ol]
        else:
            sps = [SamplingParams(temperature=args.temperature, max_tokens=n,
                                  ignore_eos=True, detokenize=False) for n in ol]
            pr = [{"prompt_token_ids": ids} for ids in pr]
        t = time.perf_counter()
        outs = llm.generate(pr, sps, use_tqdm=False)
        return time.perf_counter() - t, outs

    # warmup on a non-overlapping dataset (seed 1), so the official prompts are
    # never populated into the prefix cache
    if args.warmup_seqs > 0:
        from random import randint, seed as _seed
        _seed(1)
        wpr = [[randint(0, 10000) for _ in range(randint(100, 1024))]
               for _ in range(args.warmup_seqs)]
        wol = [randint(100, 1024) for _ in range(args.warmup_seqs)]
        run(wpr, wol)
        print(f"[{args.engine}] warmup done ({args.warmup_seqs} seqs, seed 1)",
              flush=True)

    wall, outs = run(prompts, out_lens)
    if args.engine == "nanovllm":
        actual = sum(len(o["token_ids"]) for o in outs)
    else:
        actual = sum(len(o.outputs[0].token_ids) for o in outs)

    res = {
        "engine": args.engine,
        "mode": "offline",
        "dataset": "official bench.py (seed 0)",
        "n_requests": len(prompts),
        "prompt_tokens": info["prompt_tokens"],
        "total_output_tokens": info["output_tokens"],
        "actual_output_tokens": actual,
        "output_tokens_match": actual == info["output_tokens"],
        "matches_official_dataset": info["matches_official"],
        "wall_s": wall,
        "throughput_tok_s": info["output_tokens"] / wall,
        "load_s": load_s,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "temperature": args.temperature,
        "warmup_seqs": args.warmup_seqs,
        **cfg,
    }
    print(f"RESULT {args.label or args.engine+'-official-offline'} " + json.dumps(
        {k: res[k] for k in ("total_output_tokens", "wall_s", "throughput_tok_s",
                             "output_tokens_match")}), flush=True)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        path = os.path.join(args.outdir, f"official_offline_{args.engine}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
