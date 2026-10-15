# -*- coding: utf-8 -*-
"""Latency / bandwidth benchmark: sm_89 paged decode attention vs flash-attn.

Phase 1 / M3 of ``docs/sm89_operator_plan.md``. Runs the experiment matrix from
section 5.5 and reports:

* kernel latency in microseconds (median of the timed reps),
* achieved DRAM bandwidth, from the *logical* KV bytes (``logical_bytes``), which
  is the traffic the cache hierarchy must actually fetch from DRAM, against the
  peak measured on this machine rather than the marketing figure,
* the speedup against ``flash_attn_with_kvcache`` on identical inputs,
* the launch-config, split-KV and software-pipeline sweeps behind the shipped
  defaults.

Usage (WSL ``ml`` env, RTX 4060):

    python benchmarks/benchmark_decode_attention.py
    python benchmarks/benchmark_decode_attention.py --check --sweep
    python benchmarks/benchmark_decode_attention.py --matrix batch=1,4,8 ctx=512,4096
    python benchmarks/benchmark_decode_attention.py --markdown results.md

Both kernels are timed with the L2 cache flushed between reps, because a
bandwidth-bound decode kernel otherwise measures a cache-resident best case.

Note on the bandwidth figure: with GQA (16 query heads / 8 KV heads) the Phase-1
schedule runs one program per ``(seq, query_head)``, so every KV head is read
once per query head -- 2x the KV bytes that are logically required. The table
reports both the issued bytes (what the kernel really moves, used for the GB/s
number) and the logical bytes, so the GQA redundancy is visible rather than
hidden.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly as `python benchmarks/benchmark_decode_attention.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton

from nanovllm.layers.decode_attention import (
    PagedDecodeAttentionConfig,
    choose_launch_config,
    choose_num_splits,
    paged_decode_attention_reference,
    sm89_paged_decode_attention,
)

try:
    from flash_attn import flash_attn_with_kvcache

    HAS_FLASH_ATTN = True
except Exception:  # pragma: no cover - depends on the environment
    flash_attn_with_kvcache = None
    HAS_FLASH_ATTN = False

Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 256
SCALE = HEAD_DIM ** -0.5

# RTX 4060 Laptop nominal: 128-bit bus @ ~16 Gbps -> ~256 GB/s. The benchmark
# measures the achievable figure on the actual machine (see `measure_bandwidth`)
# and reports against that, rather than against this marketing number.
NOMINAL_PEAK_GBPS = 256.0


def measure_bandwidth(n_floats: int = 64 * 1024 * 1024) -> dict:
    """Empirically calibrate achievable DRAM bandwidth on this machine.

    ``read`` (a large reduction) is the right yardstick for a decode kernel,
    which is read-dominated. ``copy`` counts bytes read plus bytes written.
    """
    x = torch.randn(n_floats, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    copy_ms = time_ms(lambda: y.copy_(x))
    read_ms = time_ms(lambda: torch.sum(x))
    del x, y
    torch.cuda.empty_cache()
    return {
        "copy_gbps": 2 * n_floats * 4 / (copy_ms * 1e-3) / 1e9,
        "read_gbps": n_floats * 4 / (read_ms * 1e-3) / 1e9,
    }

DEFAULT_MATRIX = [
    (1, 512), (1, 2048), (1, 4096),
    (4, 512), (4, 2048), (4, 4096),
    (8, 512), (8, 2048), (8, 4096),
    (16, 512), (16, 2048),
]


def build_inputs(batch: int, ctx_len: int, seed: int = 0, device: str = "cuda"):
    """Random decode-step inputs with a paged KV cache, as nano-vllm builds them."""
    gen = torch.Generator(device=device).manual_seed(seed)
    blocks_per_seq = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = batch * blocks_per_seq + 4
    perm = torch.randperm(num_blocks, generator=gen, device=device)

    block_table = torch.empty(batch, blocks_per_seq, dtype=torch.int32, device=device)
    for s in range(batch):
        block_table[s] = perm[s * blocks_per_seq : (s + 1) * blocks_per_seq]

    shape = (num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_DIM)
    k_cache = torch.randn(shape, generator=gen, device=device, dtype=torch.float32).to(torch.bfloat16)
    v_cache = torch.randn(shape, generator=gen, device=device, dtype=torch.float32).to(torch.bfloat16)
    q = torch.randn(batch, Q_HEADS, HEAD_DIM, generator=gen, device=device, dtype=torch.float32).to(torch.bfloat16)
    context_lens = torch.full((batch,), ctx_len, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, block_table, context_lens


def issued_bytes(batch: int, ctx_len: int) -> int:
    """KV bytes this schedule reads at the L2: one K and one V pass per (seq, query_head).

    This is the traffic the load units issue, which is what the L2 must serve.
    """
    return 2 * batch * Q_HEADS * ctx_len * HEAD_DIM * 2


def logical_bytes(batch: int, ctx_len: int) -> int:
    """KV bytes that must come from DRAM: one pass per (seq, kv_head).

    With GQA the two query heads sharing a KV head read the same bytes a few
    hundred KB apart in the instruction stream, so the L2 (24 MB on this card)
    absorbs the duplication and DRAM only sees the logical traffic. Reporting
    issued bytes against a DRAM peak is therefore how you end up with >100%
    "of peak"; the headline GB/s columns below use this figure.
    """
    return 2 * batch * KV_HEADS * ctx_len * HEAD_DIM * 2


def time_ms(fn, warmup: float = 25.0, rep: float = 100.0) -> float:
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def run_case(batch: int, ctx_len: int, cfg: PagedDecodeAttentionConfig, warmup: float, rep: float) -> dict:
    q, k_cache, v_cache, block_table, context_lens = build_inputs(batch, ctx_len)

    # Both kernels allocate their output inside the timed region, so the
    # comparison is apples-to-apples with the real decode path.
    ours_ms = time_ms(
        lambda: sm89_paged_decode_attention(q, k_cache, v_cache, block_table, context_lens, SCALE, config=cfg),
        warmup,
        rep,
    )

    flash_ms = float("nan")
    if HAS_FLASH_ATTN:
        flash_ms = time_ms(
            lambda: flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context_lens,
                block_table=block_table,
                softmax_scale=SCALE,
                causal=True,
            ),
            warmup,
            rep,
        )

    # Record what the auto heuristics resolved to, so the report is
    # self-documenting about which kernel was actually timed.
    auto = cfg.block_n == 0 or cfg.num_warps == 0
    block_n, num_warps = choose_launch_config(batch, Q_HEADS)
    if not auto:
        block_n, num_warps = cfg.block_n, cfg.num_warps
    num_splits = cfg.num_splits if cfg.num_splits > 0 else choose_num_splits(batch, Q_HEADS)

    logical = logical_bytes(batch, ctx_len)
    return {
        "batch": batch,
        "ctx_len": ctx_len,
        "ours_ms": ours_ms,
        "flash_ms": flash_ms,
        "speedup": (flash_ms / ours_ms) if ours_ms else float("nan"),
        "ours_gbps": logical / (ours_ms * 1e-3) / 1e9,
        "flash_gbps": logical / (flash_ms * 1e-3) / 1e9,
        "block_n": block_n,
        "num_warps": num_warps,
        "num_splits": num_splits,
    }


def check_against_flash_attn(batch: int, ctx_len: int) -> float:
    q, k_cache, v_cache, block_table, context_lens = build_inputs(batch, ctx_len)
    got = sm89_paged_decode_attention(q, k_cache, v_cache, block_table, context_lens, SCALE)
    if not HAS_FLASH_ATTN:
        ref = paged_decode_attention_reference(q, k_cache, v_cache, block_table, context_lens, SCALE)
        return (got.float() - ref.float()).abs().max().item()
    expected = flash_attn_with_kvcache(
        q.unsqueeze(1),
        k_cache,
        v_cache,
        cache_seqlens=context_lens,
        block_table=block_table,
        softmax_scale=SCALE,
        causal=True,
    ).squeeze(1)
    return (got.float() - expected.float()).abs().max().item()


def format_table(rows: list[dict], peak_gbps: float) -> str:
    header = (
        f"{'batch':>5} {'ctx':>5} {'ours (us)':>10} {'flash (us)':>11} "
        f"{'speedup':>8} {'ours GB/s':>10} {'%peak':>6} {'flash GB/s':>11}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['batch']:>5} {r['ctx_len']:>5} {r['ours_ms'] * 1e3:>10.1f} "
            f"{r['flash_ms'] * 1e3:>11.1f} {r['speedup']:>7.2f}x "
            f"{r['ours_gbps']:>10.1f} {100 * r['ours_gbps'] / peak_gbps:>5.1f}% "
            f"{r['flash_gbps']:>11.1f}"
        )
    return "\n".join(lines)


def to_markdown(rows: list[dict], cfg: PagedDecodeAttentionConfig, gpu: str, peak_gbps: float) -> str:
    lines = [
        "| batch | ctx | cfg (block_n/warps/splits) | ours (us) | flash-attn (us) | speedup | ours GB/s | %peak |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['batch']} | {r['ctx_len']} | {r['block_n']}/{r['num_warps']}/{r['num_splits']} | "
            f"{r['ours_ms'] * 1e3:.1f} | "
            f"{r['flash_ms'] * 1e3:.1f} | {r['speedup']:.2f}x | "
            f"{r['ours_gbps']:.1f} | {100 * r['ours_gbps'] / peak_gbps:.1f}% |"
        )
    speedups = [r["speedup"] for r in rows]
    lines.append("")
    lines.append(
        f"`{gpu}`, requested config `{cfg}`, measured peak read BW {peak_gbps:.0f} GB/s. "
        f"GB/s counts logical KV bytes (one pass per KV head), which is the DRAM traffic; "
        f"the load units issue {Q_HEADS // KV_HEADS}x that because the two query heads of a "
        f"GQA group each read the same KV head, and the 24 MB L2 absorbs the duplicate. "
        f"Mean speedup {sum(speedups) / len(speedups):.2f}x "
        f"(min {min(speedups):.2f}x, max {max(speedups):.2f}x)."
    )
    return "\n".join(lines)


def sweep_configs(ctx_len: int, batch: int, warmup: float, rep: float, peak_gbps: float) -> None:
    q, k_cache, v_cache, block_table, context_lens = build_inputs(batch, ctx_len)
    print(f"\nlaunch-config sweep at batch={batch}, ctx={ctx_len} (median us / % of measured peak)")
    header = f"{'block_n':>8} " + " ".join(f"{'w=' + str(w):>14}" for w in (1, 2, 4, 8))
    print(header)
    print("-" * len(header))
    for block_n in (32, 64, 128, 256):
        cells = []
        for num_warps in (1, 2, 4, 8):
            cfg = PagedDecodeAttentionConfig(block_n=block_n, num_warps=num_warps, num_splits=1)
            try:
                ms = time_ms(
                    lambda: sm89_paged_decode_attention(
                        q, k_cache, v_cache, block_table, context_lens, SCALE, config=cfg
                    ),
                    warmup,
                    rep,
                )
            except Exception:  # pragma: no cover - invalid configs
                cells.append(f"{'n/a':>14}")
                continue
            gbps = logical_bytes(batch, ctx_len) / (ms * 1e-3) / 1e9
            cells.append(f"{ms * 1e3:>7.1f}/{100 * gbps / peak_gbps:>5.1f}%")
        print(f"{block_n:>8} " + " ".join(cells))
    print("(cell = latency us / achieved % of measured peak bandwidth)")


def sweep_splits(ctx_len: int, batch: int, warmup: float, rep: float, peak_gbps: float, block_n: int, num_warps: int) -> None:
    q, k_cache, v_cache, block_table, context_lens = build_inputs(batch, ctx_len)
    print(
        f"\nsplit-KV sweep at batch={batch}, ctx={ctx_len} "
        f"(block_n={block_n}, num_warps={num_warps}; auto would pick "
        f"{choose_num_splits(batch, Q_HEADS)})"
    )
    header = f"{'num_splits':>11} {'us':>9} {'GB/s':>8} {'%peak':>7}"
    print(header)
    print("-" * len(header))
    for num_splits in (1, 2, 3, 4, 6, 8, 12, 16, 32):
        cfg = PagedDecodeAttentionConfig(block_n=block_n, num_warps=num_warps, num_splits=num_splits)
        ms = time_ms(
            lambda: sm89_paged_decode_attention(
                q, k_cache, v_cache, block_table, context_lens, SCALE, config=cfg
            ),
            warmup,
            rep,
        )
        gbps = logical_bytes(batch, ctx_len) / (ms * 1e-3) / 1e9
        print(f"{num_splits:>11} {ms * 1e3:>9.1f} {gbps:>8.1f} {100 * gbps / peak_gbps:>6.1f}%")


def sweep_stages(ctx_len: int, batch: int, warmup: float, rep: float, peak_gbps: float, block_n: int, num_warps: int) -> None:
    q, k_cache, v_cache, block_table, context_lens = build_inputs(batch, ctx_len)
    print(
        f"\nsoftware-pipeline sweep at batch={batch}, ctx={ctx_len} "
        f"(block_n={block_n}, num_warps={num_warps}, num_splits=1)"
    )
    header = f"{'num_stages':>11} {'us':>9} {'GB/s':>8} {'%peak':>7}    best_cfg"
    print(header)
    print("-" * len(header))
    for stages in (1, 2, 3, 4):
        cfg = PagedDecodeAttentionConfig(
            block_n=block_n, num_warps=num_warps, num_stages=stages, num_splits=1
        )
        ms = time_ms(
            lambda: sm89_paged_decode_attention(
                q, k_cache, v_cache, block_table, context_lens, SCALE, config=cfg
            ),
            warmup,
            rep,
        )
        gbps = logical_bytes(batch, ctx_len) / (ms * 1e-3) / 1e9
        print(f"{stages:>11} {ms * 1e3:>9.1f} {gbps:>8.1f} {100 * gbps / peak_gbps:>6.1f}%    {cfg}")


def parse_pairs(text: str) -> list[int]:
    return [int(x) for x in text.replace(" ", "").split(",") if x]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default=None, help="override, e.g. 'batch=1,4,8 ctx=512,4096'")
    ap.add_argument("--block-n", type=int, default=0, help="0 = pick from the grid size (default)")
    ap.add_argument("--num-warps", type=int, default=0, help="0 = pick from the grid size (default)")
    ap.add_argument("--num-stages", type=int, default=1)
    ap.add_argument(
        "--num-splits", type=int, default=1,
        help="1 = single pass (default, measured fastest); 0 = auto heuristic (slower on the 4060)",
    )
    ap.add_argument("--warmup", type=float, default=25.0)
    ap.add_argument("--rep", type=float, default=100.0)
    ap.add_argument("--check", action="store_true", help="verify against flash-attn/reference before timing")
    ap.add_argument("--sweep", action="store_true", help="sweep block_n x num_warps for --sweep-batch/--sweep-ctx")
    ap.add_argument("--sweep-splits", action="store_true", help="sweep num_splits for --sweep-batch/--sweep-ctx")
    ap.add_argument("--sweep-stages", action="store_true", help="sweep num_stages for --sweep-batch/--sweep-ctx")
    ap.add_argument("--sweep-batch", type=int, default=1)
    ap.add_argument("--sweep-ctx", type=int, default=4096)
    ap.add_argument("--markdown", default=None, help="write the result table to this path")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available; run this in the WSL `ml` env on the RTX 4060.", file=sys.stderr)
        return 1

    gpu = torch.cuda.get_device_name(0)
    cfg = PagedDecodeAttentionConfig(
        block_n=args.block_n, num_warps=args.num_warps, num_stages=args.num_stages, num_splits=args.num_splits
    )
    print(f"device: {gpu} (sm_{''.join(str(x) for x in torch.cuda.get_device_capability(0))})")
    print(f"torch {torch.__version__}, triton {triton.__version__}, config {cfg}")
    if not HAS_FLASH_ATTN:
        print("WARNING: flash-attn not importable; the baseline column will be empty.", file=sys.stderr)

    matrix = DEFAULT_MATRIX
    if args.matrix:
        parts = dict(kv.split("=") for kv in args.matrix.split())
        matrix = [(b, c) for b in parse_pairs(parts["batch"]) for c in parse_pairs(parts["ctx"])]

    if args.check:
        print("\ncorrectness spot check (max abs error vs flash-attn):")
        worst = 0.0
        for batch, ctx_len in matrix:
            err = check_against_flash_attn(batch, ctx_len)
            worst = max(worst, err)
            print(f"  batch={batch:<3} ctx={ctx_len:<5} max_err={err:.3e}")
        print(f"  worst={worst:.3e} (target < 1e-2)\n")

    torch.cuda.empty_cache()
    bw = measure_bandwidth()
    peak_gbps = bw["read_gbps"]
    print(
        f"measured bandwidth: read {bw['read_gbps']:.0f} GB/s, copy {bw['copy_gbps']:.0f} GB/s "
        f"(nominal {NOMINAL_PEAK_GBPS:.0f} GB/s) -> %peak is against the measured read figure"
    )

    rows = []
    for batch, ctx_len in matrix:
        row = run_case(batch, ctx_len, cfg, args.warmup, args.rep)
        rows.append(row)
        print(f"  measured batch={batch} ctx={ctx_len}", flush=True)

    print()
    print(format_table(rows, peak_gbps))
    if HAS_FLASH_ATTN:
        speedups = [r["speedup"] for r in rows]
        print(f"\nmean speedup vs flash-attn: {sum(speedups) / len(speedups):.2f}x "
              f"(min {min(speedups):.2f}x, max {max(speedups):.2f}x)")
        print(f"mean achieved bandwidth: {sum(r['ours_gbps'] for r in rows) / len(rows):.1f} GB/s "
              f"({100 * sum(r['ours_gbps'] for r in rows) / len(rows) / peak_gbps:.1f}% of measured peak, "
              f"logical KV bytes)")
    print(f"GQA read amplification: the load units issue {Q_HEADS // KV_HEADS}x the logical KB "
          f"({issued_bytes(8, 4096) / 2**20:.0f} MiB vs {logical_bytes(8, 4096) / 2**20:.0f} MiB at "
          f"batch 8 / ctx 4096); the L2 absorbs it, so it costs DRAM bandwidth only indirectly.")

    if args.sweep:
        sweep_configs(args.sweep_ctx, args.sweep_batch, args.warmup, args.rep, peak_gbps)
    if args.sweep_splits:
        sweep_splits(
            args.sweep_ctx, args.sweep_batch, args.warmup, args.rep, peak_gbps,
            args.block_n, args.num_warps,
        )
    if args.sweep_stages:
        for block_n, num_warps in ((128, 4), (256, 8), (256, 4)):
            sweep_stages(
                args.sweep_ctx, args.sweep_batch, args.warmup, args.rep, peak_gbps,
                block_n, num_warps,
            )

    if args.markdown:
        md = to_markdown(rows, cfg, gpu, peak_gbps)
        Path(args.markdown).write_text(md + "\n", encoding="utf-8")
        print(f"\nwrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
