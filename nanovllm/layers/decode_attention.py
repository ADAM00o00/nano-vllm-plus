# -*- coding: utf-8 -*-
"""SM89 (Ada / RTX 4060) specialised paged decode attention.

Phase 1 of ``docs/sm89_operator_plan.md``: a Triton implementation of the decode
attention that nano-vllm currently gets from
``flash_attn.flash_attn_with_kvcache``.

Tensor contracts (identical to the layout nano-vllm's ``ModelRunner`` and
flash-attn already use, so the two are interchangeable):

    q            [num_seqs, num_q_heads, head_dim]                 bf16
    k_cache      [num_blocks, block_size, num_kv_heads, head_dim]  bf16
    v_cache      [num_blocks, block_size, num_kv_heads, head_dim]  bf16
    block_table  [num_seqs, max_blocks]                            int32
    context_lens [num_seqs]                                        int32
    scale        float  (1 / sqrt(head_dim))
    o            [num_seqs, num_q_heads, head_dim]                 bf16

Why paged + GQA is the interesting case on a 4060: decode reads the whole KV
cache every step and the card has ~256 GB/s of bandwidth (measured: 238 GB/s
read, 247 GB/s copy). With GQA (16 query heads / 8 KV heads) the schedule below
issues every KV head once per query head -- 2x the logical KV traffic -- but the
duplicate reads land a few hundred KB apart in the instruction stream, so the
24 MB L2 absorbs them and DRAM sees only the logical traffic. That, plus two
measured negative results, is what fixed the shipped defaults:

* **Split-KV is a net loss on the shapes that matter here.** The split kernel is
  implemented, tested and correct, but the combine pass and its scratch cost
  8-10us, and the split kernel only wins that back at one measured shape
  (batch 8 / ctx 4096, ~3% at ``num_splits=2``); it loses at batch 1 and batch 4
  (``--sweep-splits``). Default is a single pass.
* **Software pipelining does nothing.** ``num_stages`` 1-4 land within 1% at
  every shape (``--sweep-stages``), so the loop is not stalling on a
  load/compute dependency the pipeliner could hide.
* **What does matter is the tile width vs the grid size.** With 16 CTAs
  (batch 1) ``block_n=256 / num_warps=8`` is up to 50% faster than 128/4; from
  32 CTAs up, 128/4 wins. ``choose_launch_config`` encodes that crossover.

Large batches already run at ~90% of measured DRAM peak, so the remaining gap to
flash-attn is at batch 1 and comes from the L2/DRAM efficiency of the paged
gather pattern, not from occupancy. See ``docs/sm89_phase1_results.md``.

The kernel is written so that CUDA-graph capture is safe: the grid is derived
purely from tensor shapes, ``context_lens`` is consumed on-device, and no
device-to-host synchronisation or dynamic allocation happens in the launcher.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = [
    "sm89_paged_decode_attention",
    "paged_decode_attention_reference",
    "paged_kv_token",
    "PagedDecodeAttentionConfig",
    "choose_launch_config",
    "choose_num_splits",
]


@triton.jit
def _paged_decode_attention_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_table_ptr,
    context_lens_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    out_ptr,
    scale,
    stride_q_seq,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_bt_seq,
    stride_o_seq,
    stride_o_head,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    num_splits: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """One program per ``(seq, query_head)``, optionally per KV split.

    ``SPLIT=False``: a single program walks the whole context and writes the
    finished output. ``SPLIT=True``: program ``(pid, split)`` walks only its own
    slice of the block range and writes an unnormalised ``(m, l, acc)`` partial
    for the combine kernel to merge. Both modes share this body so the online
    softmax exists in exactly one place; the dead branch is eliminated at trace
    time because ``SPLIT`` is a constexpr.
    """
    pid = tl.program_id(0)
    split = tl.program_id(1)
    seq_idx = pid // num_q_heads
    q_head = pid % num_q_heads
    # GQA mapping: every (num_q_heads // num_kv_heads) query heads share a KV head.
    kv_head = q_head // (num_q_heads // num_kv_heads)

    ctx_len = tl.load(context_lens_ptr + seq_idx)

    d_offs = tl.arange(0, head_dim)
    n_offs = tl.arange(0, BLOCK_N)

    # Pre-scaling q by 1/sqrt(head_dim) folds the softmax scale into the dot
    # product and keeps the inner loop free of a per-score multiply. bf16 -> fp32
    # is exact, so this costs no precision relative to scaling the scores.
    q = tl.load(q_ptr + seq_idx * stride_q_seq + q_head * stride_q_head + d_offs)
    q = q.to(tl.float32) * scale

    k_head_ptr = k_cache_ptr + kv_head * stride_k_head
    v_head_ptr = v_cache_ptr + kv_head * stride_v_head
    bt_ptr = block_table_ptr + seq_idx * stride_bt_seq

    # Splitting on block boundaries keeps every program's inner loop identical to
    # the unsplit one. A split with no blocks left over runs zero iterations and
    # contributes (m=-inf, l=0, acc=0), which the combine step weights by zero.
    num_blocks = tl.cdiv(ctx_len, block_size)
    if SPLIT:
        per_split = tl.cdiv(num_blocks, num_splits)
        blk_start = split * per_split
        blk_end = tl.minimum(num_blocks, blk_start + per_split)
    else:
        blk_start = 0
        blk_end = num_blocks

    m_i = tl.full([1], float("-inf"), tl.float32)
    l_i = tl.zeros([1], tl.float32)
    acc = tl.zeros([head_dim], tl.float32)

    # Only blocks that hold a live token are visited, so block_table padding
    # (-1 within the batch, 0 past it in the CUDA-graph buffer) is never read.
    for blk in range(blk_start, blk_end):
        block_id = tl.load(bt_ptr + blk)
        k_base = k_head_ptr + block_id * stride_k_block
        v_base = v_head_ptr + block_id * stride_v_block
        for n0 in range(0, block_size, BLOCK_N):
            token_offs = n0 + n_offs
            # `pos < ctx_len` drops the tail of the final, partially filled
            # block. `token_offs < block_size` keeps BLOCK_N from spilling into
            # the next physical block when block_size is not a multiple of it.
            mask = (blk * block_size + token_offs < ctx_len) & (token_offs < block_size)
            k_tile = tl.load(
                k_base + token_offs[:, None] * stride_k_token + d_offs[None, :],
                mask=mask[:, None],
                other=0.0,
            )
            s = tl.sum(k_tile.to(tl.float32) * q[None, :], axis=1)
            s = tl.where(mask, s, float("-inf"))

            # Online softmax: rescale the running sum/accumulator by exp(m_old -
            # m_new) before folding in the new tile.
            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)

            v_tile = tl.load(
                v_base + token_offs[:, None] * stride_v_token + d_offs[None, :],
                mask=mask[:, None],
                other=0.0,
            )
            acc = acc * alpha + tl.sum(p[:, None] * v_tile.to(tl.float32), axis=0)
            m_i = m_new

    if SPLIT:
        slot = pid * num_splits + split
        tl.store(partial_m_ptr + slot + tl.arange(0, 1), m_i)
        tl.store(partial_l_ptr + slot + tl.arange(0, 1), l_i)
        tl.store(partial_acc_ptr + slot * head_dim + d_offs, acc)
    else:
        o = acc / l_i
        tl.store(
            out_ptr + seq_idx * stride_o_seq + q_head * stride_o_head + d_offs,
            o.to(out_ptr.dtype.element_ty),
        )


@triton.jit
def _paged_decode_attention_combine_kernel(
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    out_ptr,
    stride_o_seq,
    stride_o_head,
    num_q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Merge the per-split ``(m, l, acc)`` partials of one (seq, query_head).

    Standard split-softmax reduction: re-base every partial on the global max
    ``M`` and sum ``acc * exp(m - M)`` over ``l * exp(m - M)``. Splits that got no
    blocks carry ``m = -inf`` and therefore weight 0.
    """
    pid = tl.program_id(0)
    seq_idx = pid // num_q_heads
    q_head = pid % num_q_heads

    s_offs = tl.arange(0, BLOCK_S)
    s_mask = s_offs < NUM_SPLITS
    base = pid * NUM_SPLITS

    m = tl.load(partial_m_ptr + base + s_offs, mask=s_mask, other=float("-inf"))
    l = tl.load(partial_l_ptr + base + s_offs, mask=s_mask, other=0.0)
    m_max = tl.max(m, axis=0)
    w = tl.exp(m - m_max)

    d_offs = tl.arange(0, head_dim)
    a = tl.load(
        partial_acc_ptr + (base + s_offs)[:, None] * head_dim + d_offs[None, :],
        mask=s_mask[:, None],
        other=0.0,
    )
    num = tl.sum(a * w[:, None], axis=0)
    denom = tl.sum(w * l, axis=0)

    tl.store(
        out_ptr + seq_idx * stride_o_seq + q_head * stride_o_head + d_offs,
        (num / denom).to(out_ptr.dtype.element_ty),
    )


class PagedDecodeAttentionConfig:
    """Launch configuration for :func:`sm89_paged_decode_attention`.

    ``block_n``/``num_warps`` of ``0`` mean "pick from the grid size"
    (:func:`choose_launch_config`). ``num_splits = 0`` means "pick from the grid
    size" (:func:`choose_num_splits`), and ``1`` (the default) forces the
    single-pass kernel.

    The default is ``1`` because splitting measured as a net loss on the 4060 at
    every shape that matters (see ``benchmarks/benchmark_decode_attention.py
    --sweep-splits``: the combine pass plus its scratch costs 8-10us, and only
    batch 8 / ctx 4096 with ``num_splits=2`` won ~3% of that back).
    ``num_splits > 1`` is implemented, tested and correct -- it is just not the
    right answer on this card yet.
    """

    __slots__ = ("block_n", "num_warps", "num_stages", "num_splits")

    def __init__(self, block_n: int = 0, num_warps: int = 0, num_stages: int = 1, num_splits: int = 1):
        if num_splits < 0:
            raise ValueError("num_splits must be >= 0 (0 means auto)")
        if block_n < 0 or num_warps < 0:
            raise ValueError("block_n/num_warps must be >= 0 (0 means auto)")
        self.block_n = block_n
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_splits = num_splits

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"PagedDecodeAttentionConfig(block_n={self.block_n}, "
            f"num_warps={self.num_warps}, num_stages={self.num_stages}, "
            f"num_splits={self.num_splits})"
        )


_DEFAULT_CONFIG = PagedDecodeAttentionConfig()

# Grid-size target for the automatic split heuristic, in CTAs.
#
# Decode at small batch launches only ``num_seqs * num_q_heads`` programs, so
# with batch 1 and 16 heads a 24-SM 4060 runs 16 CTAs -- two thirds of the GPU
# idle and nothing left over to hide DRAM latency with. Splitting the context
# raises the grid without changing total KV traffic, since each split reads a
# disjoint block range.
#
# The heuristic is kept for explicit experiments (``num_splits=0``) and for
# wider GPUs, but it is *not* the default: on the 4060 it loses 8-10us to the
# combine pass at every shape measured and never wins it back, so the shipped
# default is a single pass.
_SPLIT_TARGET_WAVES = 4
_MAX_NUM_SPLITS = 16
_SM_COUNT: int | None = None

# Above this many CTAs the grid already covers the machine, and wider tiles stop
# paying. Measured crossover is between batch 1 (16 CTAs, where block_n=256 /
# num_warps=8 beats 128/4 by up to 50%) and batch 2 (32 CTAs, where 128/4 wins).
# A 256-wide tile needs its 8 warps: with fewer, the fp32 score tile no longer
# fits in registers and latency explodes (measured 1200us vs 112us at batch 1,
# ctx 4096), which is why block_n and num_warps are chosen as a pair.
_SMALL_GRID_PROGRAMS = 16


def _sm_count() -> int:
    global _SM_COUNT
    if _SM_COUNT is None:
        if torch.cuda.is_available():
            _SM_COUNT = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        else:
            _SM_COUNT = 1
    return _SM_COUNT


def choose_launch_config(num_seqs: int, num_q_heads: int) -> tuple[int, int]:
    """``(block_n, num_warps)`` for this grid size, from the measured sweep.

    Depends only on tensor shapes: ``context_lens`` lives on the device and
    reading it here would force a synchronisation, which would break CUDA-graph
    capture. The two values are always chosen together -- a 256-wide tile with
    fewer than 8 warps spills catastrophically.
    """
    if max(1, num_seqs * num_q_heads) <= _SMALL_GRID_PROGRAMS:
        return 256, 8
    return 128, 4


def choose_num_splits(num_seqs: int, num_q_heads: int, target_programs: int | None = None) -> int:
    """Split factor that lifts the decode grid to ~``_SPLIT_TARGET_WAVES`` waves.

    Depends only on tensor shapes, so it is safe to call while a CUDA graph is
    being captured. Only used when a caller explicitly asks for ``num_splits=0``;
    see the note above on why the default config does not.
    """
    programs = max(1, num_seqs * num_q_heads)
    target = target_programs if target_programs is not None else _sm_count() * _SPLIT_TARGET_WAVES
    return max(1, min(_MAX_NUM_SPLITS, -(-target // programs)))


def sm89_paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    out: torch.Tensor | None = None,
    config: PagedDecodeAttentionConfig | None = None,
) -> torch.Tensor:
    """Paged decode attention (one query position per sequence).

    Drop-in replacement for ``flash_attn_with_kvcache`` in the decode path:
    same KV-cache layout, same ``block_table`` semantics, same ``context_lens``
    meaning. Returns ``[num_seqs, num_q_heads, head_dim]`` in ``q.dtype``.

    ``q`` may also be ``[num_seqs, 1, num_q_heads, head_dim]`` (the shape
    flash-attn takes); the extra axis is squeezed.
    """
    if q.dim() == 4:
        if q.size(1) != 1:
            raise ValueError(f"expected q_len == 1 for decode, got {q.size(1)}")
        q = q.squeeze(1)
    if q.dim() != 3:
        raise ValueError(f"q must be [num_seqs, num_heads, head_dim], got {tuple(q.shape)}")
    if not q.is_cuda:
        raise ValueError("sm89_paged_decode_attention requires CUDA tensors")

    num_seqs, num_q_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, cache_head_dim = k_cache.shape
    if cache_head_dim != head_dim:
        raise ValueError(f"head_dim mismatch: q={head_dim}, cache={cache_head_dim}")
    if v_cache.shape != k_cache.shape:
        raise ValueError(f"v_cache shape {tuple(v_cache.shape)} != k_cache {tuple(k_cache.shape)}")
    if num_q_heads % num_kv_heads:
        raise ValueError(f"num_q_heads ({num_q_heads}) must be a multiple of num_kv_heads ({num_kv_heads})")
    if head_dim & (head_dim - 1):
        raise ValueError(f"head_dim must be a power of two, got {head_dim}")
    if block_table.dtype != torch.int32 or context_lens.dtype != torch.int32:
        raise ValueError("block_table and context_lens must be int32")
    if block_table.dim() != 2 or block_table.size(0) != num_seqs:
        raise ValueError(f"block_table must be [num_seqs, max_blocks], got {tuple(block_table.shape)}")
    if context_lens.numel() != num_seqs:
        raise ValueError(f"context_lens must have {num_seqs} entries, got {context_lens.numel()}")
    for name, t in (("q", q), ("k_cache", k_cache), ("v_cache", v_cache), ("block_table", block_table)):
        if t.stride(-1) != 1:
            raise ValueError(f"{name} must be contiguous in the last dimension")
    if context_lens.stride(0) != 1:
        raise ValueError("context_lens must be contiguous")

    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype or out.stride(-1) != 1:
        raise ValueError("out must match q in shape/dtype and be contiguous in the last dimension")

    if num_seqs == 0:
        return out

    cfg = config or _DEFAULT_CONFIG
    auto_block_n, auto_num_warps = choose_launch_config(num_seqs, num_q_heads)
    block_n = cfg.block_n or auto_block_n
    num_warps = cfg.num_warps or auto_num_warps
    num_splits = cfg.num_splits if cfg.num_splits > 0 else choose_num_splits(num_seqs, num_q_heads)
    num_programs = num_seqs * num_q_heads

    common = dict(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=cfg.num_stages,
    )
    strides = (
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
    )

    if num_splits <= 1:
        # Single pass: no scratch buffers, no combine kernel. This is the path
        # large batches take, where the grid already saturates the SMs.
        _paged_decode_attention_kernel[(num_programs, 1)](
            q, k_cache, v_cache, block_table, context_lens, out, out, out, out,
            scale, *strides,
            num_splits=1, SPLIT=False, **common,
        )
        return out

    # Split pass. The scratch is fp32 and sized num_programs * num_splits *
    # head_dim; the auto heuristic keeps num_splits small whenever the grid is
    # already large, which bounds this at a few MB in every realistic case.
    partial_acc = torch.empty(num_programs * num_splits, head_dim, dtype=torch.float32, device=q.device)
    partial_m = torch.empty(num_programs * num_splits, dtype=torch.float32, device=q.device)
    partial_l = torch.empty(num_programs * num_splits, dtype=torch.float32, device=q.device)

    _paged_decode_attention_kernel[(num_programs, num_splits)](
        q, k_cache, v_cache, block_table, context_lens,
        partial_acc, partial_m, partial_l, out,
        scale, *strides,
        num_splits=num_splits, SPLIT=True, **common,
    )
    _paged_decode_attention_combine_kernel[(num_programs,)](
        partial_acc, partial_m, partial_l, out,
        out.stride(0), out.stride(1),
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        NUM_SPLITS=num_splits,
        BLOCK_S=triton.next_power_of_2(num_splits),
        num_warps=4,
        num_stages=1,
    )
    return out
    return out


def paged_kv_token(block_table: torch.Tensor, seq_idx: int, position: int, block_size: int) -> tuple[int, int]:
    """``(block_id, offset_in_block)`` holding ``position`` of ``seq_idx``.

    Mirrors the addressing the Triton kernel does, and is what the tests use to
    place probe values at known places in the paged cache.
    """
    return int(block_table[seq_idx, position // block_size]), position % block_size


def paged_decode_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Straightforward PyTorch oracle for paged decode attention.

    Gathers each sequence's live KV tokens out of the paged cache, then runs a
    dense softmax attention with fp32 accumulation. Deliberately simple and
    slow: it exists to be obviously correct, not fast. It is the reference the
    Triton kernel and flash-attn are both checked against.
    """
    if q.dim() == 4:
        q = q.squeeze(1)
    num_seqs, num_q_heads, head_dim = q.shape
    block_size = k_cache.size(1)
    num_kv_heads = k_cache.size(2)
    group_size = num_q_heads // num_kv_heads

    out = torch.empty(num_seqs, num_q_heads, head_dim, dtype=torch.float32, device=q.device)
    for seq_idx in range(num_seqs):
        ctx_len = int(context_lens[seq_idx])
        n_blocks = (ctx_len + block_size - 1) // block_size
        block_ids = block_table[seq_idx, :n_blocks].to(torch.long)
        # [n_blocks, block_size, Hkv, D] -> [ctx_len, Hkv, D]
        k = k_cache.index_select(0, block_ids).reshape(-1, num_kv_heads, head_dim)[:ctx_len]
        v = v_cache.index_select(0, block_ids).reshape(-1, num_kv_heads, head_dim)[:ctx_len]
        for q_head in range(num_q_heads):
            kv_head = q_head // group_size
            scores = (q[seq_idx, q_head].to(torch.float32) @ k[:, kv_head, :].to(torch.float32).T) * scale
            probs = torch.softmax(scores, dim=-1)
            out[seq_idx, q_head] = probs @ v[:, kv_head, :].to(torch.float32)
    return out.to(q.dtype)
