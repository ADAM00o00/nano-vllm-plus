# -*- coding: utf-8 -*-
"""Probe how large the pinned KV pool can be on this 8 GB card."""
import argparse
import subprocess
import sys

sys.path.insert(0, "/home/adam/nano-vllm-plus")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", type=int, required=True)
    p.add_argument("--util", type=float, default=0.90)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--max-num-seqs", type=int, default=512)
    args = p.parse_args()

    from nanovllm import LLM

    llm = LLM("/home/adam/models/Qwen3-0.6B",
              gpu_memory_utilization=args.util,
              max_model_len=4096,
              max_num_batched_tokens=args.max_num_batched_tokens,
              max_num_seqs=args.max_num_seqs,
              num_kvcache_blocks=args.blocks)
    got = llm.scheduler.block_manager.num_blocks
    kv_gib = got * 256 * 114688 / 1024**3
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    used = int(out.split(",")[0].strip().rstrip(" MiB"))
    total = int(out.split(",")[1].strip().rstrip(" MiB"))
    print(f"PROBE-OK requested={args.blocks} got={got} tokens={got*256} "
          f"kv={kv_gib:.3f}GiB | gpu_used={used}MiB total={total}MiB "
          f"leftover={total-used}MiB")


if __name__ == "__main__":
    main()
