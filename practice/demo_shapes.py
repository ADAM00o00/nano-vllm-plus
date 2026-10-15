# -*- coding: utf-8 -*-
"""
第 4 层概念演示: Qwen3-0.6B 张量维度计算器
==========================================
纯 Python 标准库, 不用 torch。把模型每一层输入输出的张量形状
一行行算出来 —— 相当于"手动前向"一遍。

用法:  python practice/demo_shapes.py
对照阅读: nanovllm/models/qwen3.py + nanovllm/layers/*
模型下载后可用下面命令核对配置(指南第 3.3 节):
  python -c "from transformers import AutoConfig; \
  c=AutoConfig.from_pretrained('~/huggingface/Qwen3-0.6B/'); print(c)"
"""

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------- Qwen3-0.6B 官方配置 (下载模型后可核对) ----------------
CFG = dict(
    vocab_size=151936,
    hidden_size=1024,
    intermediate_size=4096,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=4,
    head_dim=64,            # hidden 1024 / 16 heads
    max_position_embeddings=32768,
    rope_theta=1_000_000.0,
    rms_norm_eps=1e-6,
)

# 模拟一个 prefill 批次: 2 条请求, 长度 5 和 8 (共 13 个 token)
SEQ_LENS = [5, 8]
TOTAL_TOKENS = sum(SEQ_LENS)
N_SEQS = len(SEQ_LENS)

H = CFG["hidden_size"]
V = CFG["vocab_size"]
L = CFG["num_hidden_layers"]
NH = CFG["num_attention_heads"]
NKV = CFG["num_key_value_heads"]
HD = CFG["head_dim"]
INT = CFG["intermediate_size"]


def show(title, shape):
    print(f"  {title:<46} {shape}")


def main():
    print("=" * 76)
    print(f"Qwen3-0.6B 配置: vocab={V} hidden={H} layers={L} heads={NH} "
          f"kv_heads={NKV} head_dim={HD}")
    print(f"  intermediate={INT} max_pos={CFG['max_position_embeddings']} "
          f"rope_theta={CFG['rope_theta']:g}")
    print(f"模拟 prefill 批次: 2 条请求, 长度 {SEQ_LENS}, 共 {TOTAL_TOKENS} 个 token")
    print("=" * 76)

    # ---------- Embedding ----------
    print("\n[1] VocabParallelEmbedding (embed_tokens)")
    show("input token ids (一维, 所有序列拼在一起)", f"({TOTAL_TOKENS},)")
    show("embedding 权重", f"({V}, {H})")
    show("hidden_states", f"({TOTAL_TOKENS}, {H})")

    # ---------- 每个 DecoderLayer ----------
    print(f"\n[2] Qwen3DecoderLayer × {L} (每层结构相同)")
    x = (TOTAL_TOKENS, H)
    show("输入 hidden_states", x)

    print("\n[2a] RMSNorm + 残差 (input_layernorm)")
    show("norm 输出 / 残差流", f"({TOTAL_TOKENS}, {H})")

    print("\n[2b] Qwen3Attention")
    qkv_out = (TOTAL_TOKENS, (NH + 2 * NKV) * HD)
    show("qkv_proj 输出 (一次算出 Q/K/V)", qkv_out)
    show("  Q 部分", f"({TOTAL_TOKENS}, {NH * HD})  = ({TOTAL_TOKENS}, {NH}, {HD})")
    show("  K 部分", f"({TOTAL_TOKENS}, {NKV * HD})  = ({TOTAL_TOKENS}, {NKV}, {HD})")
    show("  V 部分", f"({TOTAL_TOKENS}, {NKV * HD})  = ({TOTAL_TOKENS}, {NKV}, {HD})")
    show("q_norm/k_norm 后 (Qwen3 特性, 每头单独归一)", f"({TOTAL_TOKENS}, {NH}, {HD})")
    show("RoPE 旋转后", f"({TOTAL_TOKENS}, {NH}, {HD})")
    show("Attention 输出 (flash-attn)", f"({TOTAL_TOKENS}, {NH}, {HD})")
    show("展平 flatten(1,-1)", f"({TOTAL_TOKENS}, {NH * HD})")
    show("o_proj 输出", f"({TOTAL_TOKENS}, {H})")

    print("\n[2c] RMSNorm + 残差 (post_attention_layernorm)")
    show("norm 输出 / 残差流", f"({TOTAL_TOKENS}, {H})")

    print("\n[2d] Qwen3MLP (SwiGLU)")
    show("gate_up_proj 输出 (gate 和 up 拼一起)", f"({TOTAL_TOKENS}, {2 * INT})")
    show("chunk(2) 后: gate / up", f"({TOTAL_TOKENS}, {INT}) 各一份")
    show("silu(gate) * up", f"({TOTAL_TOKENS}, {INT})")
    show("down_proj 输出", f"({TOTAL_TOKENS}, {H})")

    print("\n[2e] 残差相加 → 本层输出")
    show("hidden_states", f"({TOTAL_TOKENS}, {H})")

    # ---------- 最终 Norm + LM Head ----------
    print("\n[3] 最终 RMSNorm (norm)")
    show("hidden_states", f"({TOTAL_TOKENS}, {H})")

    print("\n[4] ParallelLMHead (lm_head) —— 生成 logits")
    show("全量 logits", f"({TOTAL_TOKENS}, {V})")
    show("prefill 时只取每条序列最后一个位置 (cu_seqlens_q[1:]-1)",
         f"({N_SEQS}, {V})")
    print(f"  {'>> 给 Sampler: 每条序列 1 个 logits 向量 → 采样出新 token':<46}")

    # ---------- 权重参数量 ----------
    print("\n[5] 参数量估算 (bf16)")
    embed = V * H
    attn_qkv = H * (NH + 2 * NKV) * HD
    attn_o = NH * HD * H
    mlp_gate_up = H * 2 * INT
    mlp_down = INT * H
    per_layer = attn_qkv + attn_o + mlp_gate_up + mlp_down
    total = embed + L * per_layer + H * (V + 1)   # +lm_head (V*H) + 各 norm
    show("embedding 权重", f"{embed/1e6:.1f}M")
    show("单层 attention+mlp", f"{per_layer/1e6:.1f}M")
    show("lm_head (与 embedding 同形, 可 tie)", f"{V*H/1e6:.1f}M")
    show("总参数量 ≈", f"{total/1e9:.3f}B")
    show("bf16 权重占用 ≈", f"{total*2/1e9:.2f} GB")

    # ---------- KV Cache 块大小 ----------
    print("\n[6] KV Cache 单块显存 (allocate_kv_cache 的 block_bytes 公式)")
    block_size = 256
    dtype_bytes = 2  # bf16
    block_bytes = 2 * L * block_size * NKV * HD * dtype_bytes
    show("每块 = 2(K+V) × 层数 × 块大小 × kv头 × head_dim × 字节",
         f"{block_bytes/2**20:.1f} MB")
    total_mem_gb = 8.0   # 假设 8GB 显卡
    usable = total_mem_gb * 0.9 - 1.2   # gpu_memory_utilization=0.9, 减模型约 1.2GB
    show("8GB 卡可用 KV 空间 ≈", f"{usable:.2f} GB")
    show("可分配块数 ≈", f"{int(usable*1024/ (block_bytes/2**20))} 块 "
         f"(每块 256 token → 最多缓存约 {int(usable*1024/(block_bytes/2**20))*256:,} token)")

    print("\n" + "=" * 76)
    print("观察: 每层都保持 (N, 1024) 的'主干'形状 —— 这就是残差流(residual stream)")
    print("Attention/MLP 只是在主干上'借道'算完再还回去。")
    print("KV Cache 才是显存大头: 6 号公式决定了引擎能同时服务多少请求。")


if __name__ == "__main__":
    main()
