# -*- coding: utf-8 -*-
"""Correctness tests for the sm_89 paged decode attention kernel.

Phase 1 / M1 of ``docs/sm89_operator_plan.md``.

Two kinds of checks live here:

* **Parity checks** against ``paged_decode_attention_reference`` and against
  ``flash_attn.flash_attn_with_kvcache`` on random data (the plan's
  ``max abs error < 1e-2`` criterion).
* **Exact-structure checks** built so that a subtly wrong kernel cannot pass
  them: uniform-softmax probes that encode every KV position into ``v`` (so the
  *set* of positions the kernel reads is verified), one-hot probes placed at
  chosen paged coordinates (so block/offset addressing is verified), and
  per-KV-head constants (so the GQA head mapping is verified).

Run in the WSL ``ml`` env (RTX 4060 / sm_89):

    python tests/test_decode_attention.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Allow running directly as `python tests/test_decode_attention.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from nanovllm.layers.decode_attention import (
    PagedDecodeAttentionConfig,
    choose_launch_config,
    choose_num_splits,
    paged_decode_attention_reference,
    paged_kv_token,
    sm89_paged_decode_attention,
)

try:  # flash-attn is the baseline we must match; not required to run the tests.
    from flash_attn import flash_attn_with_kvcache

    HAS_FLASH_ATTN = True
except Exception:  # pragma: no cover - depends on the environment
    flash_attn_with_kvcache = None
    HAS_FLASH_ATTN = False

CUDA = torch.cuda.is_available()
DEVICE = "cuda"

# Qwen3-0.6B attention geometry, plus nano-vllm's default paged-KV block size.
Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 256
SCALE = HEAD_DIM ** -0.5

# The plan's acceptance criterion for bf16.
ATOL = 1e-2


def _build_paged_inputs(
    ctx_lens,
    num_q_heads=Q_HEADS,
    num_kv_heads=KV_HEADS,
    head_dim=HEAD_DIM,
    block_size=BLOCK_SIZE,
    seed=0,
    unused_fill=None,
):
    """Random q / paged KV cache / block_table for a batch of context lengths.

    Physical block ids are handed out from a permutation so that a kernel which
    silently assumed identity block ordering would fail, and the table is padded
    with ``-1`` the same way ``ModelRunner.prepare_block_tables`` does.
    """
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    lens = torch.as_tensor(list(ctx_lens), dtype=torch.int32, device=DEVICE)
    batch = lens.numel()
    blocks_per_seq = [max(1, (int(x) + block_size - 1) // block_size) for x in ctx_lens]
    max_blocks = max(blocks_per_seq)

    # Keep the live blocks and the unused blocks disjoint so probes written into
    # "unused" slots can never bleed into a sequence's live range.
    num_live = sum(blocks_per_seq)
    spare = 3
    num_blocks = num_live + spare
    perm = torch.randperm(num_blocks, generator=gen, device=DEVICE)

    block_table = torch.full((batch, max_blocks), -1, dtype=torch.int32, device=DEVICE)
    cursor = 0
    for s, nb in enumerate(blocks_per_seq):
        block_table[s, :nb] = perm[cursor : cursor + nb]
        cursor += nb

    def rand_cache():
        t = torch.randn(
            num_blocks, block_size, num_kv_heads, head_dim, generator=gen, device=DEVICE, dtype=torch.float32
        )
        return t.to(torch.bfloat16)

    q = torch.randn(batch, num_q_heads, head_dim, generator=gen, device=DEVICE, dtype=torch.float32).to(torch.bfloat16)
    k_cache = rand_cache()
    v_cache = rand_cache()
    if unused_fill is not None:
        live = set()
        for s in range(batch):
            for p in range(int(lens[s])):
                live.add(paged_kv_token(block_table, s, p, block_size))
        live_slots = torch.zeros(num_blocks, block_size, dtype=torch.bool, device=DEVICE)
        for blk, off in live:
            live_slots[blk, off] = True
        for cache in (k_cache, v_cache):
            cache[~live_slots] = unused_fill
    return q, k_cache, v_cache, block_table, lens


def _reference_kwargs(inputs):
    q, k_cache, v_cache, block_table, lens = inputs
    return dict(q=q, k_cache=k_cache, v_cache=v_cache, block_table=block_table, context_lens=lens, scale=SCALE)


@unittest.skipUnless(CUDA, "requires a CUDA device")
class ReferenceParityTest(unittest.TestCase):
    """Kernel vs the PyTorch oracle on random data."""

    def _check(self, ctx_lens, **kwargs):
        inputs = _build_paged_inputs(ctx_lens, **kwargs)
        ref = paged_decode_attention_reference(**_reference_kwargs(inputs))
        got = sm89_paged_decode_attention(**_reference_kwargs(inputs))
        self.assertEqual(got.shape, ref.shape)
        self.assertEqual(got.dtype, ref.dtype)
        max_err = (got.float() - ref.float()).abs().max().item()
        torch.testing.assert_close(got.float(), ref.float(), atol=ATOL, rtol=0)
        return max_err

    def test_various_batch_and_context_lengths(self):
        # 1 and 17 exercise "shorter than one block"; 256 / 257 straddle the
        # block boundary; 255 / 511 are partial final blocks.
        for ctx_lens in ([1], [17], [255], [256], [257], [255, 256, 257], [1, 512, 513]):
            with self.subTest(ctx_lens=ctx_lens):
                err = self._check(ctx_lens)
                self.assertLess(err, ATOL)

    def test_batch_sizes(self):
        for batch in (1, 2, 4, 8, 16):
            with self.subTest(batch=batch):
                self._check([1024] * batch)

    def test_long_context(self):
        for ctx in (2048, 4096):
            with self.subTest(ctx=ctx):
                self._check([ctx, ctx // 2])

    def test_ragged_batch_mixed_lengths(self):
        self._check([1, 256, 4096, 777, 2048, 129])

    def test_other_head_geometries(self):
        # MHA and a wider GQA group must both work; the kernel derives the group
        # size from num_q_heads // num_kv_heads.
        for num_q_heads, num_kv_heads in ((8, 8), (16, 4), (16, 8), (32, 8)):
            with self.subTest(q_heads=num_q_heads, kv_heads=num_kv_heads):
                self._check([700, 1300], num_q_heads=num_q_heads, num_kv_heads=num_kv_heads)

    def test_smaller_block_size(self):
        # The kernel takes block_size as a constexpr; check a non-default one.
        self._check([300, 1024], block_size=64)

    def test_nan_in_unused_cache_slots_is_ignored(self):
        # Unused cache slots hold NaN. Masking must drop them before they reach
        # the softmax, otherwise the whole output turns NaN.
        inputs = _build_paged_inputs([257, 1000], unused_fill=float("nan"))
        got = sm89_paged_decode_attention(**_reference_kwargs(inputs))
        self.assertFalse(torch.isnan(got).any(), "NaN leaked in from unused KV slots")

    def test_block_table_padding_is_ignored(self):
        # Cells past each sequence's live block count must never be
        # dereferenced: poisoning them with different junk has to give the
        # identical answer. 0 and -1 are the two values that actually appear in
        # nano-vllm's buffers (`prepare_block_tables` pads with -1, the CUDA
        # graph buffer is zero-filled), and block 0 is live for another
        # sequence, so read-through is detectable.
        inputs = _build_paged_inputs([300, 1000])
        q, k_cache, v_cache, block_table, lens = inputs
        ref = paged_decode_attention_reference(**_reference_kwargs(inputs))

        for tail in (-1, 0):
            with self.subTest(tail=tail):
                bt = torch.full(
                    (block_table.size(0), block_table.size(1) + 4), tail, dtype=torch.int32, device=DEVICE
                )
                bt[:, : block_table.size(1)] = block_table
                got = sm89_paged_decode_attention(q, k_cache, v_cache, bt, lens, SCALE)
                torch.testing.assert_close(got.float(), ref.float(), atol=ATOL, rtol=0)

    def test_flash_attn_parity(self):
        if not HAS_FLASH_ATTN:
            self.skipTest("flash-attn not installed")
        for ctx_lens in ([1], [128], [2048, 300], [4096]):
            with self.subTest(ctx_lens=ctx_lens):
                q, k_cache, v_cache, block_table, lens = _build_paged_inputs(ctx_lens)
                got = sm89_paged_decode_attention(q, k_cache, v_cache, block_table, lens, SCALE)
                expected = flash_attn_with_kvcache(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    cache_seqlens=lens,
                    block_table=block_table,
                    softmax_scale=SCALE,
                    causal=True,
                )
                expected = expected.squeeze(1)
                torch.testing.assert_close(got.float(), expected.float(), atol=ATOL, rtol=0)


@unittest.skipUnless(CUDA, "requires a CUDA device")
class ExactStructureTest(unittest.TestCase):
    """Probes that pin down positions, paged addressing and the GQA mapping."""

    def _empty_case(self, ctx_lens, block_size=BLOCK_SIZE, num_q_heads=Q_HEADS, num_kv_heads=KV_HEADS):
        gen = torch.Generator(device=DEVICE).manual_seed(1234)
        lens = torch.as_tensor(list(ctx_lens), dtype=torch.int32, device=DEVICE)
        batch = lens.numel()
        blocks_per_seq = [max(1, (int(x) + block_size - 1) // block_size) for x in ctx_lens]
        max_blocks = max(blocks_per_seq)
        num_blocks = sum(blocks_per_seq) + 2
        perm = torch.randperm(num_blocks, generator=gen, device=DEVICE)
        block_table = torch.full((batch, max_blocks), -1, dtype=torch.int32, device=DEVICE)
        cursor = 0
        for s, nb in enumerate(blocks_per_seq):
            block_table[s, :nb] = perm[cursor : cursor + nb]
            cursor += nb
        shape = (num_blocks, block_size, num_kv_heads, HEAD_DIM)
        k_cache = torch.zeros(shape, dtype=torch.bfloat16, device=DEVICE)
        v_cache = torch.zeros(shape, dtype=torch.bfloat16, device=DEVICE)
        return k_cache, v_cache, block_table, lens

    def test_attends_exactly_the_live_positions(self):
        """Uniform softmax + position-encoded ``v`` reveals the position set.

        ``k == 0`` makes every score 0, so the softmax is uniform over whatever
        the kernel attends to. ``v`` encodes the *global* position of each live
        token (as ``p / block_size``) while every non-live slot holds a large
        sentinel. Any over-read, mis-masked tail, or stray block shows up as a
        shift far larger than bf16 noise.

        Because the output is bf16, the tolerance is one bf16 ulp at the
        expected magnitude: ``(ctx_len - 1) / (2 * block_size)`` can land
        exactly on a tie (511/512 needs nine mantissa bits), and either
        neighbour is then a correct rounding. Single-token errors at long
        context are caught by the quantisation-proof probes below instead.
        """
        block_size = BLOCK_SIZE
        ctx_lens = [1, 2, 3, 17, 255, 256, 257, 511, 512, 513]
        k_cache, v_cache, block_table, lens = self._empty_case(ctx_lens, block_size=block_size)
        sentinel = 4.0
        v_cache.fill_(sentinel)

        for s, ctx_len in enumerate(ctx_lens):
            for p in range(ctx_len):
                blk, off = paged_kv_token(block_table, s, p, block_size)
                v_cache[blk, off, :, :] = p / block_size

        q = torch.full((len(ctx_lens), Q_HEADS, HEAD_DIM), 0.5, dtype=torch.bfloat16, device=DEVICE)
        got = sm89_paged_decode_attention(q, k_cache, v_cache, block_table, lens, SCALE).float()

        for s, ctx_len in enumerate(ctx_lens):
            # Mean of p/block_size over p in [0, ctx_len) == (ctx_len - 1) / (2 * block_size).
            expected = (ctx_len - 1) / (2.0 * block_size)
            atol = 2.0 ** -8 * max(1.0, abs(expected))  # one bf16 ulp
            torch.testing.assert_close(
                got[s],
                torch.full_like(got[s], expected),
                atol=atol,
                rtol=0,
                msg=f"seq {s} (ctx_len={ctx_len}) attended the wrong set of positions",
            )

    def test_last_live_token_is_included(self):
        """A spike on the final live token catches a dropped tail token.

        Every live token is 0 except the very last one, so the mean is exactly
        ``1 / ctx_len``; non-live slots hold a 4.0 sentinel. Dropping the tail
        token zeroes the result and sweeping in a sentinel adds ``4 / ctx_len``
        -- both far outside bf16 noise even at ``ctx_len = 1000``, unlike a
        probe that spreads its signal across all positions.
        """
        block_size = BLOCK_SIZE
        ctx_lens = [1, 2, 17, 256, 257, 512, 513, 1000]
        k_cache, v_cache, block_table, lens = self._empty_case(ctx_lens, block_size=block_size)
        v_cache.fill_(4.0)

        for s, ctx_len in enumerate(ctx_lens):
            for p in range(ctx_len - 1):
                blk, off = paged_kv_token(block_table, s, p, block_size)
                v_cache[blk, off, :, :] = 0.0
            blk, off = paged_kv_token(block_table, s, ctx_len - 1, block_size)
            v_cache[blk, off, :, :] = 1.0

        q = torch.full((len(ctx_lens), Q_HEADS, HEAD_DIM), 0.5, dtype=torch.bfloat16, device=DEVICE)
        got = sm89_paged_decode_attention(q, k_cache, v_cache, block_table, lens, SCALE).float()

        for s, ctx_len in enumerate(ctx_lens):
            want = torch.tensor(1.0 / ctx_len, dtype=torch.bfloat16, device=DEVICE).float()
            torch.testing.assert_close(
                got[s],
                torch.full_like(got[s], want),
                atol=1e-4,
                rtol=0,
                msg=f"seq {s} (ctx_len={ctx_len}) did not include the final live token",
            )

    def test_paged_addressing_selects_the_exact_token(self):
        """A one-hot ``k`` at a chosen paged coordinate must win the softmax.

        The winning token's ``v`` is a one-hot vector over KV-head index, so the
        output must be exactly that one-hot vector -- which verifies both the
        block/offset arithmetic and the query-head -> KV-head mapping.

        ``exp(40)`` against ``ctx_len - 1`` zero scores makes the winning weight
        ``1 - ctx_len * exp(-40)``, so this stays exact at long context where a
        mean-based probe would drown in bf16 quantisation.
        """
        block_size = BLOCK_SIZE
        ctx_lens = [1, 17, 256, 257, 1000]
        for ctx_len in ctx_lens:
            # 0 / 1 pin the very first token, block_size - 1 and block_size the
            # block seam, ctx_len - 1 the last token of a partial final block.
            targets = {0, 1, block_size - 1, block_size, block_size + 5, ctx_len - 1}
            for target in sorted(t for t in targets if t < ctx_len):
                with self.subTest(ctx_len=ctx_len, target=target):
                    k_cache, v_cache, block_table, lens = self._empty_case(
                        [ctx_len], block_size=block_size
                    )
                    blk, off = paged_kv_token(block_table, 0, target, block_size)
                    # k is a unit vector on dim 0 at the target only, so every
                    # other live score is exactly 0.
                    k_cache[blk, off, :, 0] = 1.0
                    v_cache[blk, off, :, :KV_HEADS] = torch.eye(KV_HEADS, dtype=torch.bfloat16, device=DEVICE)

                    # q on dim 0 too, sized so the *scaled* score at the target
                    # is ~40 -- exp(40) swamps the ctx_len - 1 zero scores.
                    q = torch.zeros(1, Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
                    q[0, :, 0] = 40.0 / SCALE
                    got = sm89_paged_decode_attention(q, k_cache, v_cache, block_table, lens, SCALE).float()

                    expected = torch.zeros(Q_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE)
                    heads = torch.arange(Q_HEADS, device=DEVICE)
                    expected[heads, heads // (Q_HEADS // KV_HEADS)] = 1.0
                    torch.testing.assert_close(got[0], expected, atol=1e-3, rtol=0)

    def test_gqa_head_mapping(self):
        """Each KV head owns a constant ``v``; uniform softmax must return it.

        The expected output for query head ``h`` is ``kv_head(h) + 1`` and the
        four KV heads are far apart, so any off-by-one in the head mapping
        fails loudly instead of averaging out.
        """
        ctx_lens = [1, 300, 1024]
        k_cache, v_cache, block_table, lens = self._empty_case(ctx_lens)
        for kv_head in range(KV_HEADS):
            v_cache[:, :, kv_head, :] = float(kv_head + 1)

        q = torch.ones(len(ctx_lens), Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        got = sm89_paged_decode_attention(q, k_cache, v_cache, block_table, lens, SCALE).float()

        expected_heads = torch.arange(Q_HEADS, device=DEVICE) // (Q_HEADS // KV_HEADS)
        expected = (expected_heads + 1).to(torch.float32)
        for s in range(len(ctx_lens)):
            torch.testing.assert_close(got[s], expected[:, None].expand(Q_HEADS, HEAD_DIM), atol=1e-3, rtol=0)


@unittest.skipUnless(CUDA, "requires a CUDA device")
class SplitKvTest(unittest.TestCase):
    """Split-KV path: partial softmax statistics + the combine kernel."""

    def _check(self, ctx_lens, num_splits, **kwargs):
        inputs = _build_paged_inputs(ctx_lens, **kwargs)
        ref = paged_decode_attention_reference(**_reference_kwargs(inputs))
        cfg = PagedDecodeAttentionConfig(num_splits=num_splits)
        got = sm89_paged_decode_attention(**_reference_kwargs(inputs), config=cfg)
        torch.testing.assert_close(got.float(), ref.float(), atol=ATOL, rtol=0)
        return got

    def test_split_matches_single_pass(self):
        # num_splits deliberately includes values that are not powers of two
        # (BLOCK_S is rounded up) and values far larger than the block count
        # (most splits get no blocks at all and must weigh in as zero).
        for ctx_lens in ([256], [1000], [300, 700, 4096]):
            for num_splits in (1, 2, 3, 4, 5, 8, 16, 33, 64):
                with self.subTest(ctx_lens=ctx_lens, num_splits=num_splits):
                    self._check(ctx_lens, num_splits)

    def test_split_covers_exact_block_ranges(self):
        # 7 blocks over 3 splits -> per_split = 3 -> ranges 0:3, 3:6, 6:7. Ragged
        # sequences get their own per_split, so this also checks the device-side
        # range arithmetic rather than a host loop.
        self._check([7 * BLOCK_SIZE, 3 * BLOCK_SIZE, BLOCK_SIZE], 3)

    def test_split_single_token_context(self):
        # One live token: every split but the first must contribute nothing.
        self._check([1], 8)

    def test_auto_split_matches_reference(self):
        # num_splits=0 takes the shape-derived heuristic path. It is not the
        # shipped default (it measured slower), but it must stay correct.
        for ctx_lens in ([4096], [512, 512], [1000] * 8):
            with self.subTest(ctx_lens=ctx_lens):
                self._check(ctx_lens, 0)

    def test_shipped_default_is_single_pass(self):
        # Guards the measured decision: the default must not pay for a combine.
        self.assertEqual(PagedDecodeAttentionConfig().num_splits, 1)
        cfg = PagedDecodeAttentionConfig()
        self.assertEqual(cfg.block_n, 0)  # resolved per grid size at launch
        self.assertEqual(cfg.num_warps, 0)

    def test_split_with_unused_cache_nan(self):
        inputs = _build_paged_inputs([2000, 300], unused_fill=float("nan"))
        cfg = PagedDecodeAttentionConfig(num_splits=4)
        got = sm89_paged_decode_attention(**_reference_kwargs(inputs), config=cfg)
        self.assertFalse(torch.isnan(got).any())

    def test_choose_num_splits_is_shape_only_and_monotonic(self):
        # Fixed target so the assertion does not depend on this machine's SM count.
        self.assertEqual(choose_num_splits(1, 16, target_programs=96), 6)
        self.assertEqual(choose_num_splits(2, 16, target_programs=96), 3)
        self.assertEqual(choose_num_splits(8, 16, target_programs=96), 1)
        self.assertEqual(choose_num_splits(64, 16, target_programs=96), 1)
        # Never exceeds the cap, never drops below 1.
        self.assertEqual(choose_num_splits(1, 1, target_programs=1 << 20), 16)
        previous = None
        for batch in (1, 2, 3, 4, 8, 16, 32):
            got = choose_num_splits(batch, 16)
            self.assertGreaterEqual(got, 1)
            self.assertLessEqual(got, 16)
            if previous is not None:
                self.assertLessEqual(got, previous)  # bigger batch needs no more splits
            previous = got


@unittest.skipUnless(CUDA, "requires a CUDA device")
class AutoConfigTest(unittest.TestCase):
    """The shape-only launch heuristics, and that they actually get used."""

    def test_choose_launch_config_crossover(self):
        # 16 CTAs (batch 1 x 16 heads) is the small-grid case; 32 is not.
        self.assertEqual(choose_launch_config(1, 16), (256, 8))
        self.assertEqual(choose_launch_config(2, 8), (256, 8))
        self.assertEqual(choose_launch_config(2, 16), (128, 4))
        self.assertEqual(choose_launch_config(16, 16), (128, 4))
        # block_n 256 is never paired with fewer than 8 warps.
        for batch in range(1, 33):
            block_n, num_warps = choose_launch_config(batch, 16)
            if block_n == 256:
                self.assertEqual(num_warps, 8)

    def test_auto_config_matches_reference_at_both_grid_sizes(self):
        # Exercises both branches of the heuristic through the real launcher.
        for ctx_lens in ([4096], [4096, 4096], [777] * 8):
            with self.subTest(ctx_lens=ctx_lens):
                inputs = _build_paged_inputs(ctx_lens)
                ref = paged_decode_attention_reference(**_reference_kwargs(inputs))
                got = sm89_paged_decode_attention(**_reference_kwargs(inputs))
                torch.testing.assert_close(got.float(), ref.float(), atol=ATOL, rtol=0)

    def test_rejects_negative_config(self):
        with self.assertRaises(ValueError):
            PagedDecodeAttentionConfig(num_splits=-1)
        with self.assertRaises(ValueError):
            PagedDecodeAttentionConfig(block_n=-8)
        with self.assertRaises(ValueError):
            PagedDecodeAttentionConfig(num_warps=-1)


@unittest.skipUnless(CUDA, "requires a CUDA device")
class InterfaceTest(unittest.TestCase):
    """Shape, dtype and launch-configuration contracts."""

    def test_output_contract(self):
        q, k_cache, v_cache, block_table, lens = _build_paged_inputs([100, 300])
        out = sm89_paged_decode_attention(q, k_cache, v_cache, block_table, lens, SCALE)
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(out.dtype, q.dtype)
        self.assertTrue(out.is_contiguous())

    def test_flash_attn_style_4d_q(self):
        inputs = _build_paged_inputs([100, 300])
        q4 = inputs[0].unsqueeze(1)
        got = sm89_paged_decode_attention(q4, inputs[1], inputs[2], inputs[3], inputs[4], SCALE)
        self.assertEqual(got.shape, inputs[0].shape)

    def test_preallocated_output_buffer(self):
        inputs = _build_paged_inputs([512])
        out = torch.empty_like(inputs[0])
        ret = sm89_paged_decode_attention(*inputs[:4], inputs[4], SCALE, out=out)
        self.assertIs(ret, out)
        ref = paged_decode_attention_reference(**_reference_kwargs(inputs))
        torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=0)

    def test_non_contiguous_q_and_row_padded_block_table(self):
        inputs = _build_paged_inputs([300, 700])
        q, k_cache, v_cache, block_table, lens = inputs
        # A wider q whose rows are embedded in a larger buffer: strides, not
        # contiguity, must drive the addressing.
        q_padded = torch.zeros(q.size(0), q.size(1), HEAD_DIM * 2, dtype=q.dtype, device=DEVICE)
        q_padded[:, :, :HEAD_DIM] = q
        q_view = q_padded[:, :, :HEAD_DIM]
        self.assertFalse(q_view.is_contiguous())

        # block_table with padding in the fast dimension but a wider row stride.
        bt_padded = torch.full((block_table.size(0), block_table.size(1) + 7), -1, dtype=torch.int32, device=DEVICE)
        bt_padded[:, : block_table.size(1)] = block_table
        bt_view = bt_padded[:, : block_table.size(1)]
        self.assertFalse(bt_view.is_contiguous())

        ref = paged_decode_attention_reference(q, k_cache, v_cache, block_table, lens, SCALE)
        got = sm89_paged_decode_attention(q_view, k_cache, v_cache, bt_view, lens, SCALE)
        torch.testing.assert_close(got.float(), ref.float(), atol=ATOL, rtol=0)

    def test_launch_configs_agree(self):
        inputs = _build_paged_inputs([4096, 900])
        ref = paged_decode_attention_reference(**_reference_kwargs(inputs))
        for block_n in (16, 32, 64, 128, 256):
            for num_warps in (1, 2, 4, 8):
                for num_splits in (1, 3):
                    with self.subTest(block_n=block_n, num_warps=num_warps, num_splits=num_splits):
                        cfg = PagedDecodeAttentionConfig(
                            block_n=block_n, num_warps=num_warps, num_splits=num_splits
                        )
                        got = sm89_paged_decode_attention(**_reference_kwargs(inputs), config=cfg)
                        torch.testing.assert_close(got.float(), ref.float(), atol=ATOL, rtol=0)

    def test_rejects_bad_inputs(self):
        q, k_cache, v_cache, block_table, lens = _build_paged_inputs([100])
        narrow_cache = k_cache[:, :, :, :64].contiguous()
        with self.assertRaises(ValueError):  # q/cache head_dim mismatch
            sm89_paged_decode_attention(q, narrow_cache, narrow_cache, block_table, lens, SCALE)
        with self.assertRaises(ValueError):  # non-int32 block_table
            sm89_paged_decode_attention(q, k_cache, v_cache, block_table.long(), lens, SCALE)
        with self.assertRaises(ValueError):  # non-int32 context_lens
            sm89_paged_decode_attention(q, k_cache, v_cache, block_table, lens.float(), SCALE)
        with self.assertRaises(ValueError):  # block_table shorter than the batch
            sm89_paged_decode_attention(q, k_cache, v_cache, block_table[:0], lens, SCALE)
        with self.assertRaises(ValueError):  # non-power-of-two head_dim
            sm89_paged_decode_attention(q[:, :, :96], k_cache, v_cache, block_table, lens, SCALE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
