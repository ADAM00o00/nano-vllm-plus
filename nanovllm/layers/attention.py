import os
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.layers.decode_attention import sm89_paged_decode_attention
from nanovllm.utils.context import get_context


# Which kernel runs the decode step.
#
#   NANOVLLM_DECODE_ATTN=flash_attn  (default) flash_attn_with_kvcache
#   NANOVLLM_DECODE_ATTN=triton      nanovllm/layers/decode_attention.py
#
# Read once at import: the decode loop runs this on every layer of every step,
# and an env lookup per layer would show up in a ~100us step. Switching backends
# therefore means restarting the engine, which is how the A/B runs in
# docs/sm89_operator_plan.md section 5.6 are done anyway.
DECODE_ATTENTION_BACKENDS = ("flash_attn", "triton")


def _resolve_decode_backend() -> str:
    backend = os.environ.get("NANOVLLM_DECODE_ATTN", "flash_attn").strip().lower()
    if backend not in DECODE_ATTENTION_BACKENDS:
        raise ValueError(
            f"NANOVLLM_DECODE_ATTN={backend!r} is not one of {DECODE_ATTENTION_BACKENDS}"
        )
    return backend


DECODE_ATTENTION_BACKEND = _resolve_decode_backend()


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            if DECODE_ATTENTION_BACKEND == "triton":
                # Same paged-KV layout and block_table semantics as flash-attn,
                # so this only differs in which kernel reads them.
                o = sm89_paged_decode_attention(q, k_cache, v_cache,
                                                context.block_tables, context.context_lens,
                                                self.scale)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True)
        return o
