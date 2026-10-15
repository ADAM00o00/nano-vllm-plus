# -*- coding: utf-8 -*-
"""Shared helpers for the nano-vllm vs vLLM head-to-head benchmark.

Design notes (why this is built the way it is):

* **Every measurement uses a freshly generated dataset.**  Both engines keep a
  prefix cache whose entries survive after the owning request is released
  (``BlockManager.hash_to_block_id`` is never cleared in nano-vllm; vLLM
  likewise retains cached blocks).  Re-running the same prompts would therefore
  turn the second run into a partial prefix-cache hit and silently inflate it.
  We sidestep that by deriving the dataset seed from
  ``(engine, mode, concurrency, repetition)`` so no two runs ever share tokens.
* **Prefixes cannot collide by construction.**  Prompts are drawn as uniform
  random token ids from a safe sub-range of the vocabulary, so request *i* and
  request *j* differ starting at token 0.  That is stronger than "different
  first block": it defeats prefix reuse at any block granularity, in both
  engines, within a run and across runs.
* **Fixed input and output length.**  ``ignore_eos`` / ``min_tokens`` keep the
  output length exact, so throughput numbers are directly comparable.
"""
import json
import os
import statistics
from typing import Iterable

import numpy as np

# Prompt/output length used for the head-to-head comparison.
# Qwen3-0.6B: 28 layers, hidden 1024, 32k positions, 8 KV heads x 128 head_dim.
# 512 in / 128 out is the point where this model stops being purely
# weight-byte-bound and KV bytes start to matter (the worklog's crossover
# B* = W/L is ~20 sequences at ctx 512), so TTFT and throughput both carry
# signal instead of one of them being a constant.
DEFAULT_PROMPT_LEN = 512
DEFAULT_OUTPUT_LEN = 128

# Token ids are sampled from [SAFE_LO, vocab_size - SAFE_HI_MARGIN) to avoid
# special/control tokens that some engines treat specially.
SAFE_LO = 1000
SAFE_HI_MARGIN = 1000


def prompt_seed(engine: str, mode: str, concurrency: int, rep: int = 0) -> int:
    """Deterministic-but-unique dataset seed for one measurement."""
    key = f"{engine}|{mode}|{concurrency}|{rep}"
    h = 1469598103934665603
    for b in key.encode():
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h % (2**31)


def build_prompts(n: int, prompt_len: int, seed: int, vocab_size: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    lo, hi = SAFE_LO, vocab_size - SAFE_HI_MARGIN
    if hi <= lo:
        raise ValueError(f"vocab_size {vocab_size} too small for safe id range")
    return [rng.integers(lo, hi, size=prompt_len).tolist() for _ in range(n)]


def summarize(values: Iterable[float]) -> dict:
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"count": 0}
    n = len(vals)

    def q(p):
        # nearest-rank percentile; adequate for the sample sizes used here
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return vals[idx]

    return {
        "count": n,
        "mean": statistics.fmean(vals),
        "min": vals[0],
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
        "max": vals[-1],
    }


def make_result(engine, mode, concurrency, prompts, metrics, extra=None) -> dict:
    ttfts = metrics["ttft_s"]
    e2es = metrics["e2e_s"]
    wall = metrics["wall_s"]
    gen_tokens = metrics["gen_tokens"]
    prompt_tokens = sum(len(p) for p in prompts)
    out = {
        "engine": engine,
        "mode": mode,
        "concurrency": concurrency,
        "n_requests": len(prompts),
        "prompt_len": len(prompts[0]) if prompts else 0,
        "prompt_len_min": min((len(p) for p in prompts), default=0),
        "prompt_len_max": max((len(p) for p in prompts), default=0),
        "dataset_seed": metrics.get("dataset_seed"),
        "ttft_ms": {k: v * 1000.0 for k, v in summarize(ttfts).items() if k != "count"},
        "ttft_count": len(ttfts),
        "e2e_ms": {k: v * 1000.0 for k, v in summarize(e2es).items() if k != "count"},
        "wall_s": wall,
        "gen_tokens": gen_tokens,
        "prompt_tokens": prompt_tokens,
        "gen_throughput_tok_s": gen_tokens / wall if wall > 0 else 0.0,
        "total_throughput_tok_s": (gen_tokens + prompt_tokens) / wall if wall > 0 else 0.0,
        "tokens_per_request": sorted(metrics.get("per_request_tokens", [])),
    }
    if extra:
        out.update(extra)
    return out


def fmt_row(r: dict) -> str:
    t = r["ttft_ms"]
    return (
        f"{r['engine']:>9} {r['mode']:>7} c={r['concurrency']:>3} "
        f"| TTFT mean {t.get('mean', float('nan')):8.2f} p50 {t.get('p50', float('nan')):8.2f} "
        f"p99 {t.get('p99', float('nan')):9.2f} ms "
        f"| wall {r['wall_s']:7.3f}s gen {r['gen_throughput_tok_s']:9.1f} tok/s "
        f"total {r['total_throughput_tok_s']:9.1f} tok/s"
    )


def save_result(path: str | None, obj: dict, label: str = "") -> None:
    line = fmt_row(obj)
    print(("RESULT " + label + " " + line).strip(), flush=True)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        print(f"wrote {path}", flush=True)
