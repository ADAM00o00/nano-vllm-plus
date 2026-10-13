# SM89 / RTX 4060 专用算子实现计划

> 目标：为 RTX 4060（Ada Lovelace，sm_89）实现一个专用推理算子，优先替换 nano-vllm-plus 中最热的 **decode attention** 路径。
> 本文档是给新对话的开工说明，包含环境、现状、算子设计、验证指标、集成点和里程碑。

---

## 0. 一句话目标

在 nano-vllm-plus 里，把 decode 阶段的 attention 从通用 flash-attn 替换为一个 **RTX 4060 / sm_89 专用 kernel**，减少显存带宽压力、降低 decode 延迟、提升在线服务吞吐。

推荐路线：

```text
Phase 1: Triton 写 paged decode attention
Phase 2: 用 Nsight 调优，对比 flash-attn
Phase 3: 视收益决定是否重写成 CUDA C++ / CUTLASS
Phase 4: 集成进 nanovllm/layers/attention.py 的 decode 路径
```

---

## 1. 环境信息

| 项目 | 值 |
|---|---|
| 系统 | WSL2 Ubuntu 26.04 |
| Conda 环境 | `ml` |
| Python | 3.12.15 |
| PyTorch | 2.13.0+cu130 |
| CUDA | 13.0，系统路径 `/usr/local/cuda` |
| nvcc | 13.0.88 |
| flash-attn | 2.8.3.post1（已定制编译，详见第 12 节） |
| 模型 | `/home/adam/models/Qwen3-0.6B` |
| WSL 项目副本 | `/home/adam/nano-vllm-plus` |
| Windows 原仓库 | `C:\Users\Adam\PycharmProjects\nano-vllm-plus` |
| GPU | RTX 4060 Laptop，8GB，sm_89，约 24 SM，约 256 GB/s 带宽 |
| vLLM | 0.31.0 已安装，但本项目主要用 nano-vllm |

### 环境检查命令

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd ~/nano-vllm-plus

python - <<'PY'
import torch, flash_attn
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
print("flash_attn", flash_attn.__version__)
PY
```

---

## 2. 模型形状（Qwen3-0.6B）

| 参数 | 值 |
|---|---|
| hidden_size | 1024 |
| num_hidden_layers | 28 |
| num_attention_heads | 16 |
| num_key_value_heads | 8（GQA，16:8 = 2:1） |
| head_dim | 128 |
| dtype | bf16 |
| kvcache_block_size | 256（`nanovllm/config.py` 默认） |
| max_model_len | 4096（默认，可配置） |

Decode 阶段典型 batch：1 / 2 / 4 / 8 / 16。
Decode 阶段 context 长度：512 / 1024 / 2048 / 4096。

---

## 3. 为什么优先做 decode attention

1. **在线服务里 decode 是主要路径**：每个请求会跑几十到几百步 decode，prefill 只有一次。
2. **4060 是带宽瓶颈**：decode 阶段每生成 1 个 token，都要把整个 KV cache 读一遍。
3. **优化空间明确**：分页 KV、GQA、split-KV、FP8/INT8、L2 复用。
4. **收益可量化**：和 `flash_attn_with_kvcache` 对比延迟 / 吞吐。

---

## 4. 现有代码入口

### 4.1 `nanovllm/layers/attention.py`

当前实现：

```python
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

class Attention(nn.Module):
    def forward(self, q, k, v):
        context = get_context()
        if context.is_prefill:
            # prefill: flash_attn_varlen_func
        else:
            # decode: flash_attn_with_kvcache
```

你要做的是：

- 保留 prefill 走 flash-attn
- 把 decode 分支替换成你的 `sm89_paged_decode_attention(q, k_cache, v_cache, block_table, context_lens, ...)`
- 或者把 `store_kvcache` 也一起替换掉（Phase 2）

### 4.2 `nanovllm/utils/context.py`

`Context` 里有：

- `context_lens`：每个 seq 的 KV 长度
- `block_tables`：每个 seq 的 KV block 索引表
- `slot_mapping`：prefill 写 KV cache 用

### 4.3 `nanovllm/engine/model_runner.py`

重点函数：

- `prepare_decode(seqs)`：构造 decode 的 `context_lens` / `block_tables`
- `prepare_prefill(seqs)`：prefill 的 `slot_mapping` / `cu_seqlens`
- `allocate_kv_cache()`：KV cache 布局

现有 KV cache 形状：

```python
self.kv_cache = torch.empty(
    2,                      # k / v
    num_hidden_layers,
    num_kvcache_blocks,
    block_size,
    num_kv_heads,
    head_dim,
)
```

### 4.4 服务层

- `nanovllm/serving/grpc_server.py`：gRPC 服务
- `nanovllm/serving/engine_worker.py`：后台 step 循环
- `nanovllm/serving/gateway.py`：Redis 路由网关
- 压测脚本：`benchmarks/benchmark_grpc.py`

---

## 5. Phase 1：Paged Decode Attention

### 5.1 输入 / 输出

输入：

```text
q            [num_seqs, num_q_heads, head_dim]       bf16
k_cache      [num_blocks, block_size, num_kv_heads, head_dim]  bf16
v_cache      [num_blocks, block_size, num_kv_heads, head_dim]  bf16
block_table  [num_seqs, max_blocks]                  int32
context_lens [num_seqs]                              int32
scale        float32  (1 / sqrt(head_dim))
```

输出：

```text
o            [num_seqs, num_q_heads, head_dim]       bf16
```

### 5.2 算法

对每个 `(seq, q_head)`：

```text
kv_head = q_head // (num_q_heads // num_kv_heads)   # GQA 映射

m = -inf        # running max
l = 0           # running sum
acc = 0         # running output

for block_idx in range(0, ceil(context_len / block_size)):
    block_id = block_table[seq, block_idx]
    for j in range(block_size):
        pos = block_idx * block_size + j
        if pos >= context_len:
            break
        k = k_cache[block_id, j, kv_head, :]
        v = v_cache[block_id, j, kv_head, :]
        s = dot(q, k) * scale
        m_new = max(m, s)
        l = l * exp(m - m_new) + exp(s - m_new)
        acc = acc * exp(m - m_new) + exp(s - m_new) * v
        m = m_new

o = acc / l
```

### 5.3 Triton kernel 设计

建议先写 Triton 版本，文件：

```text
nanovllm/layers/decode_attention.py
```

设计要点：

1. **Grid**
   - 第一版：`grid = (num_seqs * num_q_heads,)`
   - 每个 program 处理一个 `(seq, q_head)`
   - 后续优化：`grid = (num_seqs * num_q_heads, num_kv_splits)` 做 split-KV

2. **分页索引**
   - 用 `tl.load(block_table + seq * max_blocks + block_idx)` 拿到 block_id
   - 计算 `k_cache` 的地址偏移
   - 避免除法/取模，block_size 用 2 的幂（当前 256）

3. **Online softmax**
   - 维护 `m_i`、`l_i`、`acc`
   - 用 `tl.exp`，注意 `-inf` 处理

4. **GQA**
   - `num_q_heads=16`，`num_kv_heads=8`
   - `kv_head = q_head // 2`

5. **mask**
   - 超过 `context_len` 的位置不参与

6. **调参**
   - `BLOCK_N`：每次处理多少个 KV token，建议 32 / 64 / 128
   - `num_warps`：4 / 8
   - split-KV：`num_splits` 2 / 4 / 8

### 5.4 正确性验证

写一个 reference：

```python
def torch_reference(q, k_cache, v_cache, block_table, context_lens, scale):
    # 用 PyTorch 手动 gather KV，做 attention
    ...
```

对比：

- max abs error < 1e-2（bf16 容差）
- 和 `flash_attn_with_kvcache` 的输出对比
- 覆盖 batch=1/2/4/8，ctx=512/2048/4096

### 5.5 性能评测

实验矩阵：

| batch | context_len | 对比对象 |
|---|---|---|
| 1 | 512 / 2048 / 4096 | flash-attn |
| 4 | 512 / 2048 / 4096 | flash-attn |
| 8 | 512 / 2048 / 4096 | flash-attn |
| 16 | 512 / 2048 | flash-attn |

指标：

- kernel latency（us / ms）
- 端到端 decode 延迟
- 吞吐（tok/s）
- Nsight Compute 的显存带宽利用率
- Nsight Compute 的 occupancy / stall

### 5.6 集成到 nano-vllm

方案：

1. 新增 `nanovllm/layers/decode_attention.py`
2. 在 `Attention.forward` 的 decode 分支调用：

```python
if context.is_prefill:
    o = flash_attn_varlen_func(...)
else:
    o = sm89_paged_decode_attention(
        q, k_cache, v_cache,
        context.block_tables,
        context.context_lens,
        self.scale,
    )
```

3. 保持 prefill 走 flash-attn，先不动
4. 跑 `python -m unittest discover -s tests`
5. 启动真实服务，跑 `benchmarks/benchmark_grpc.py` 对比前后

---

## 6. Phase 2：Fused QKV + RoPE + KV Store

Decode 阶段每步还有：

- QKV projection
- RoPE
- KV cache store（当前是 Triton `store_kvcache_kernel`）

可以做融合：

```text
one kernel:
  input hidden_states
  -> QKV linear
  -> apply RoPE
  -> write K/V to paged cache
  -> output Q for attention
```

收益：

- 少一次显存读写
- 少一次 kernel launch

风险：

- 线性层权重在大矩阵乘里，融合到 Triton 可能不如 cuBLAS
- 更现实的融合：`RoPE + KV store`，QKV 仍走 cuBLAS

---

## 7. Phase 3：KV Cache 量化

4060 带宽只有约 256 GB/s，KV cache 是最大带宽来源。

可以尝试：

- KV cache 存 FP8（e4m3）
- 或存 INT8 / INT4
- attention kernel 里在 shared memory 反量化

收益：

- KV 读取量减半或更多
- decode 延迟可能明显下降

注意：

- Qwen3-0.6B 对 KV 量化比较敏感，要先验证精度
- 需要 `scale` / `zero_point`
- 可以先做 per-head 或 per-block 量化

---

## 8. 实验与验收标准

### 功能验收

- [ ] 正确性：和 flash-attn 输出 max error < 1e-2
- [ ] 覆盖 batch=1/2/4/8/16，ctx=512/2048/4096
- [ ] 集成后 `tests/test_serving_core.py` 全部通过
- [ ] gRPC 在线服务能正常返回真实 token

### 性能验收

- [ ] decode kernel 延迟低于 flash-attn（至少持平）
- [ ] 8 并发在线吞吐不低于当前基线（约 237 tok/s）
- [ ] TTFT p99 不明显变差（当前约 82ms）
- [ ] 显存占用不增加

### 可观测性

- [ ] 新增 kernel 的延迟 / 吞吐指标
- [ ] 和 flash-attn 的对比数据
- [ ] Nsight 截图或数据

---

## 9. 里程碑

| 阶段 | 内容 | 产出 |
|---|---|---|
| M1 | 写 PyTorch reference + 正确性测试 | `tests/test_decode_attention.py` |
| M2 | Triton kernel 第一版 | `nanovllm/layers/decode_attention.py` |
| M3 | 正确性对齐 + 基础 benchmark | 正确性数据 + 延迟对比 |
| M4 | 调优（split-KV / block size / num_warps） | 性能数据 |
| M5 | 集成进 `Attention.forward` | 端到端服务可跑 |
| M6 | 在线压测对比 | benchmark 报告 |
| M7（可选） | 重写 CUDA C++ / CUTLASS | 更高性能版本 |
| M8（可选） | KV 量化 | 额外收益 |

---

## 10. 风险与注意事项

1. **CUDA Graph 兼容性**
   - nano-vllm 开启了 CUDA graph
   - 你的 kernel 必须支持固定 shape / 固定地址
   - 第一版可以先 `enforce_eager=True` 跑通，再考虑 graph

2. **WSL 内存**
   - WSL 只有约 7.7GB 内存
   - 编译时用 `MAX_JOBS=1`、`NVCC_THREADS=1`
   - 不要同时跑 vLLM 和大编译

3. **Triton 对 paged 索引的支持**
   - 动态 block_table 索引在 Triton 里要注意边界
   - 先保证正确，再优化

4. **GQA**
   - Qwen3-0.6B 是 16 Q heads / 8 KV heads
   - GQA 映射写错会静默出错，必须有 reference 对齐

5. **bf16 精度**
   - 累加建议用 fp32
   - 输出再转 bf16

6. **block_size**
   - nano-vllm 默认 256
   - kernel 里 block_size 可以作为 constexpr 传入

7. **sm_89 编译**
   - 专用 kernel 用 `TORCH_CUDA_ARCH_LIST=8.9` 或 `-gencode arch=compute_89,code=sm_89`
   - 不要只编 sm_80 靠 PTX JIT

---

## 11. 简历表述模板

> 针对 RTX 4060（sm_89）实现专用 paged decode attention kernel，针对 GQA + 分页 KV cache 优化访存，decode 延迟降低 X%，在线服务 8 并发吞吐从 A tok/s 提升到 B tok/s。

可补充：

- Triton / CUDA 实现
- split-KV、online softmax、L2 复用
- 和 flash-attn 的对比数据
- 集成进 gRPC 在线服务后的端到端 p99

---

## 12. 工具链现状备注

当前 WSL `ml` 环境里的 flash-attn 是**定制编译版**：

- 只编译了 `head_dim=128`
- 去掉了 32/64/96/192/256
- `HEADDIM_SWITCH` 被改成只支持 128
- 编译时用了 `FLASH_ATTN_CUDA_ARCHS=80`、`MAX_JOBS=1`、`NVCC_THREADS=1`
- 修复了 CUDA 13.0 + glibc 2.43 的 `rsqrt` 冲突

如果新算子要重新编译 flash-attn 或 C++ extension，注意沿用这些设置。

系统 CUDA 头文件已打补丁：

```text
/usr/local/cuda/include/crt/math_functions.h
```

给 `rsqrt` / `rsqrtf` 加了 `noexcept`（仅 `__cplusplus` 下）。

---

## 13. 新对话开工清单

新对话开始时，按这个顺序做：

1. 读本文档
2. 确认环境：`conda activate ml`，跑一次真实 `LLM.generate`
3. 确认当前 baseline：
   ```bash
   python benchmarks/benchmark_grpc.py --target localhost:50051 --requests 32 --concurrency 8 --max-tokens 32
   ```
4. 写 PyTorch reference + 正确性测试
5. 写 Triton kernel 第一版
6. 对齐正确性
7. 跑 benchmark 对比 flash-attn
8. 集成进 `Attention.forward` decode 分支
9. 重跑在线服务压测
10. 产出数据和简历表述
