# 第 4 层讲义：Qwen3 模型结构与 layers 算子

> 配套: `python practice/demo_shapes.py`(张量维度计算器, 先跑它!)
> 阅读顺序: `models/qwen3.py`(216行) → `layers/` 下 8 个文件
> 本层是 PyTorch 进阶主战场。所有算子的形状变化都在 demo_shapes.py 里。

---

## 0. 先建立"残差流"心智模型

```
token ids
  → VocabParallelEmbedding  (151936 → 1024)
  → 28 × DecoderLayer:
       主干 hidden_states (N, 1024) 一直保持这个形状
       每层内部: RMSNorm → Attention → RMSNorm → MLP, 全部加回残差
  → RMSNorm
  → ParallelLMHead → logits (N, 151936)
```

**残差流(residual stream)**: 每一层都在"主干"(N, 1024)上借道计算, 算完再加回去。
这也是为什么 demo_shapes.py 里所有中间结果都以 1024 结尾——只有 Attention/MLP 内部会临时放大维度。

**Qwen3-0.6B 配置速记** (下载模型后可用指南 3.3 的命令核对):
vocab 151936 | hidden 1024 | 28 层 | 16 Q 头 / 4 KV 头 (GQA 4:1) | head_dim 64 | intermediate 4096 | max_pos 32768 | rope_theta 1e6

## 1. Qwen3Model —— 主干

```python
class Qwen3Model(nn.Module):
    def __init__(self, config):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions):
        hidden_states = self.embed_tokens(input_ids)
        residual = None                       # 残差流从 None 开始
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

**不常见①: `nn.ModuleList`** —— 就是"装了 nn.Module 的普通 list"。为什么不用 Python list? 因为 `nn.ModuleList` 里的子模块会被自动注册进父模块, `model.parameters()`/`.to("cuda")` 才找得到它们。用 `nn.Sequential` 也可以, 但这里每层还要传 `positions` 和 `residual`, 需要手动循环, 所以用 ModuleList。

**`forward(input_ids, positions)` 只有两个输入**: 没有 attention mask! 因为 varlen flash-attn 用 `cu_seqlens` 表达每条序列的边界(第 5 层), 不需要传统 mask 矩阵。

## 2. Qwen3DecoderLayer —— 一层的骨架

```python
def forward(self, positions, hidden_states, residual):
    if residual is None:                    # 第一层: 没有残差可加
        hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(positions, hidden_states)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

**残差融合(fused residual)**: 注意 RMSNorm 的输入是 `(hidden_states, residual)`, 输出是 `(norm后的, 更新后的residual)`——norm 把"残差相加"和"归一化"合成一步做(见下节)。这是省一次显存读写的优化, 也是"普通开发者不会这么写"的推理引擎风格。

## 3. RMSNorm (layernorm.py) —— 不用均值, 只用均方根

```python
@torch.compile
def rms_forward(self, x):
    orig_dtype = x.dtype
    x = x.float()                                   # ① 转 float32 算, 防精度丢失
    var = x.pow(2).mean(dim=-1, keepdim=True)       # ② 均方
    x.mul_(torch.rsqrt(var + self.eps))             # ③ 除以 RMS (rsqrt = 1/sqrt)
    x = x.to(orig_dtype).mul_(self.weight)          # ④ 乘可学习缩放, 转回原精度
    return x

@torch.compile
def add_rms_forward(self, x, residual):
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())            # ⑤ 残差相加(融合!)
    residual = x.to(orig_dtype)                     #   更新残差流
    ...(同上 ②③④)...
    return x, residual
```

**要点**:
- **为什么用 float32 算**: bf16/fp16 只有 ~3 位有效数字, 直接算方差会丢精度。norm 这种"先算统计量再缩放"的操作, 业界标准做法是升到 fp32 算完再降回来。
- **`rsqrt`**: `1/sqrt(x)` 的向量化版本, 比 `1.0 / torch.sqrt(x)` 快。
- **原地算子 `mul_`/`add_`**: 下划线 = 原地修改, 不产生新张量(省显存)。注意 `x = x.to(orig_dtype).mul_(self.weight)` 的链式调用。
- **RMSNorm vs LayerNorm**: 不做减均值, 只做除均方根——数学上更省, 且对"偏移"不敏感, 是 Llama/Qwen 系标配。
- `@torch.compile` 第 5 层细讲, 先知道"它让这段 Python 编译成更快的 kernel"即可。

## 4. Qwen3Attention —— GQA + QK-Norm

```python
self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim,
                                  self.total_num_heads, self.total_num_kv_heads, bias=qkv_bias)
self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size)
self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim,
                           max_position=max_position, base=rope_theta)
self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)
if not self.qkv_bias:
    self.q_norm = RMSNorm(self.head_dim)     # ★ Qwen3 特性: QK-Norm
    self.k_norm = RMSNorm(self.head_dim)
```

```python
def forward(self, positions, hidden_states):
    qkv = self.qkv_proj(hidden_states)                    # (N, 1536)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q = q.view(-1, self.num_heads, self.head_dim)         # (N, 16, 64)
    k = k.view(-1, self.num_kv_heads, self.head_dim)      # (N, 4, 64)
    v = v.view(-1, self.num_kv_heads, self.head_dim)
    if not self.qkv_bias:
        q = self.q_norm(q)                                # 每头单独归一
        k = self.k_norm(k)
    q, k = self.rotary_emb(positions, q, k)               # RoPE
    o = self.attn(q, k, v)                                # (N, 16, 64)
    return self.o_proj(o.flatten(1, -1))                  # (N, 1024)
```

**① 一次线性层算 Q/K/V**: `qkv_proj` 把三个投影矩阵拼成一个 (1024 → 1536), 一次矩阵乘出 Q/K/V 再 `split`。为什么? 矩阵乘是显存带宽瓶颈, 拼在一起能减少 kernel 启动次数、充分利用缓存。这是推理引擎的经典优化(训练代码常写成三个 `nn.Linear`)。

**② `split([...], dim=-1)` 不常见③**: 按长度把最后一维切成三段(Q: 1024, K: 256, V: 256)。和 `chunk(3)` 的区别: `split` 支持**不等长**切分。

**③ `view(-1, ...)` 不常见④**: 只改形状不改数据(要求内存连续, `reshape` 更宽容)。`-1` = 让 PyTorch 自己推。注意 q/k/v 的头数是不同的(16 vs 4)——这就是 GQA。

**④ GQA(分组查询注意力)**: 16 个 Q 头共享 4 组 K/V 头(每组 4 个 Q 头)。为什么? KV Cache 的大小只和 K/V 头数挂钩——KV 头从 16 砍到 4, **KV Cache 直接缩小 4 倍**(还记得 demo_shapes 第 6 节的 7MB/块吗? 那是按 4 头算的)。

**⑤ QK-Norm(不常见⑤, Qwen3 的招牌特性)**: 对 Q 和 K 每头做一次 RMSNorm。作用: 稳定训练、降低注意力分数对尺度变化的敏感度。`q_norm = RMSNorm(head_dim)` —— norm 作用在最后一维 head_dim=64 上。只有 `qkv_bias=False` 时才需要(有 bias 时权重已自带偏移修正)。

## 5. RotaryEmbedding —— 位置信息怎么进模型 (rotary_embedding.py)

```python
inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
t = torch.arange(max_position_embeddings, dtype=torch.float)
freqs = torch.einsum("i,j -> ij", t, inv_freq)      # 外积: (max_pos, rotary_dim/2)
cos, sin = freqs.cos(), freqs.sin()
cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1) # (max_pos, 1, rotary_dim)
self.register_buffer("cos_sin_cache", cache, persistent=False)
```

```python
def apply_rotary_emb(x, cos, sin):
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)      # 切成两半
    y1 = x1 * cos - x2 * sin                        # 旋转公式
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)
```

**① 原理一句话**: RoPE 把"位置 p"编码成旋转角度 p×ωᵢ(ωᵢ 是不同频率), 把每个 head 的向量按位置旋转。这样两个 token 的**相对位置**会体现在它们向量的**夹角**里, 注意力天然感知距离。频率 `inv_freq` 从 1/θ⁰ 到 1/θ^(dim/2) 指数递减(θ=1e6), 和正弦位置编码同源。

**② `torch.einsum("i,j -> ij", t, inv_freq)` 不常见⑥**: Einstein 求和记号, 这里就是**外积**——`t` 的每个位置 × `inv_freq` 的每个频率。等价于 `t[:, None] * inv_freq[None, :]`, 但更直观地表达"把所有自由维度都乘一遍"。

**③ `register_buffer(persistent=False)` 不常见⑦**: 把张量注册成模块的 buffer(随模块 .to("cuda") 迁移、参与 state_dict 保存——除非 `persistent=False`)。`persistent=False` 表示**不进 state_dict**: 因为 cos/sin 表是纯函数预计算(可由 max_pos/θ 推导), 没必要存进权重文件。

**④ `get_rope` 用 `@lru_cache(1)` 包一层 不常见⑧**: `functools.lru_cache` 缓存函数返回值。`lru_cache(1)` = 只缓存最近 1 个结果。因为 28 层用的 RoPE 完全一样, 只构造一次, 后面 27 层直接拿缓存。

**⑤ 为什么 forward 里先 `chunk` 成两半再旋转**: 数学上, 旋转操作把向量分成 (x1, x2) 两半分别旋转再拼回——这是 RoPE 的**偶数/奇数分半**实现, 比复数实现更省显存。

## 6. Qwen3MLP —— SwiGLU

```python
self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False)
self.down_proj = RowParallelLinear(intermediate_size, hidden_size)
self.act_fn = SiluAndMul()

def forward(self, x):
    gate_up = self.gate_up_proj(x)      # (N, 8192)
    x = self.act_fn(gate_up)            # silu(gate) * up
    x = self.down_proj(x)               # (N, 1024)
```

```python
class SiluAndMul(nn.Module):
    @torch.compile
    def forward(self, x):
        x, y = x.chunk(2, -1)           # (N, 4096) 两份
        return F.silu(x) * y            # SwiGLU = silu(gate) ⊙ up
```

**SwiGLU 公式**: `silu(gate) * up`。gate 和 up 是同一个线性层(拼接成 8192)切出来的两半。`F.silu` 是 SiLU 激活(≈x·sigmoid(x))。为什么用 SwiGLU: 比 ReLU 表达能力强, 是 Llama/Qwen/DeepSeek 系 MLP 的标配。

## 7. linear.py —— 四种并行线性层 (重点!)

### 7.1 基类: 把 weight_loader 挂到参数上

```python
class LinearBase(nn.Module):
    def __init__(self, input_size, output_size, bias=False, tp_dim=None):
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader   # ★ 方法挂到 Parameter 上!
        ...
```

**不常见⑨: `self.weight.weight_loader = self.weight_loader`** —— 把一个**实例方法当作属性**挂到 nn.Parameter 对象上。PyTorch 的 Parameter 本质是 Tensor, 可以挂任意属性。这样 `loader.py` 里 `param = model.get_parameter(name); weight_loader = getattr(param, "weight_loader")` 就能拿到**这个层专属的**权重加载逻辑。这是 vLLM 风格的"多态分发", 比在 loader 里写一堆 if/elif 干净得多。

**`torch.empty` 不初始化**: 权重随后由 load_model 从 safetensors 拷入。未初始化就前向会得到垃圾值——所以加载必须在推理前完成。

### 7.2 四种切分方式(张量并行基础, 第 6 层细讲)

| 类 | 怎么切 | 何时用 |
|---|---|---|
| `ColumnParallelLinear` | 输出维按 rank 切 | Q/K/V、gate/up(每个 rank 算一部分输出) |
| `MergedColumnParallelLinear` | 两个 Column 拼成一个权重, 按输出段切 | gate_up_proj |
| `QKVParallelLinear` | 特殊版 Column: Q/K/V 各按头切 | qkv_proj |
| `RowParallelLinear` | 输入维按 rank 切, 输出全量, 末尾 all_reduce | o_proj、down_proj |

**`narrow` 不常见⑩**: `tensor.narrow(dim, start, length)` 返回一个**视图**(不拷贝数据)切片, 等价于 `tensor[start:start+length]`。weight_loader 用它把完整权重"切出自己 rank 的那一段"再 `copy_` 进参数。为什么不直接切片? 语义更明确 + 支持任意维度。

**RowParallelLinear 的两个细节**:
```python
y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
if self.tp_size > 1:
    dist.all_reduce(y)     # 把各 rank 的部分和加起来
```
- **bias 只在 rank 0 加一次**(否则 bias 会被加 N 次)
- **all_reduce 求和**: 因为输入被切成 N 份、各 rank 算了部分输出, 必须求和才是完整结果

### 7.3 QKVParallelLinear 的头级切分

```python
def weight_loader(self, param, loaded_weight, loaded_shard_id):
    if loaded_shard_id == "q":   shard_size, shard_offset = num_heads*head_dim, 0
    elif loaded_shard_id == "k": shard_size, shard_offset = num_kv_heads*head_dim, num_heads*head_dim
    else:                        shard_size, shard_offset = num_kv_heads*head_dim, (num_heads+num_kv_heads)*head_dim
    param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
    loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
    param_data.copy_(loaded_weight)
```

Q/K/V 在权重里的排布: `[Q(16头) | K(4头) | V(4头)]`, 每部分内部再按 rank 切。`loaded_shard_id` 是 loader 根据权重名字("q_proj"/"k_proj"/"v_proj")传来的, 决定切哪一段。这里"按头切"保证了每个 rank 拿到**完整连续的若干个头**(而不是把每个头切成碎片), 因为注意力是按头并行的。

## 8. embed_head.py —— 词表并行

```python
class VocabParallelEmbedding(nn.Module):
    # 权重只装 vocab_size/tp_size 行
    def forward(self, x):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)   # 不在本 rank 词表范围的置 0
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y               # 置 0 的行 embedding 也为 0
            dist.all_reduce(y)                      # 各 rank 的部分和 = 完整 embedding
        return y
```

**词表切分的技巧**: 每张卡只存 1/tp_size 的词向量。token id 不在自己范围内的, 通过 `mask * (x - start)` 映射成 0(0 号词向量), 算完 embedding 再乘 mask 清零, 最后 all_reduce 求和——不在本卡的 token 贡献 0, 在的贡献真实值。**优雅的"稀疏求和"**, 避免条件分支。

```python
class ParallelLMHead(VocabParallelEmbedding):
    def forward(self, x):
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1   # 每条序列的最后一个位置
            x = x[last_indices].contiguous()              # (N_seqs, 1024)
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if rank==0 else None
            dist.gather(logits, all_logits, 0)            # rank 0 收齐所有分片
            logits = torch.cat(all_logits, -1) if rank==0 else None
        return logits
```

**prefill 取末位**: prefill 时每条序列只关心**最后一个位置的 logits**(用它采样下一个 token)。`cu_seqlens_q[1:] - 1` 直接算出每条序列末位的行号, 一步取齐——这也是为什么第 5 层要精心构造 cu_seqlens。`contiguous()` 确保索引后内存连续(后续 F.linear 要求)。

**gather vs all_reduce 不常见⑪**: `all_reduce` = 所有人拿到**相同**的完整结果(求和); `gather` = 只有 rank 0 拿到**拼接**的完整结果, 其他人拿到 None。LM Head 只需要 rank 0 输出 logits 采样, 用 gather 省了广播。

## 9. attention.py —— flash-attn 的两个入口

```python
class Attention(nn.Module):
    def forward(self, q, k, v):
        context = get_context()
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)  # 先存 KV
        if context.is_prefill:
            if context.block_tables is not None:    # 前缀缓存: 让 q 看到整个缓存
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                     max_seqlen_q=..., cu_seqlens_q=..., cu_seqlens_k=...,
                     softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                     cache_seqlens=context.context_lens, block_table=context.block_tables,
                     softmax_scale=self.scale, causal=True)
        return o
```

- **store_kvcache**: 用 Triton kernel(见下)把本轮新算的 K/V 按 `slot_mapping` 写进大缓存张量。slot_mapping = "每个 token 的 KV 应该存到缓存的哪个槽位"(由 block_table 换算, 第 5 层 prepare_* 里构造)。
- **prefill 用 `flash_attn_varlen_func`**: 支持变长序列(cu_seqlens 表达边界), 一次算完整个批次。前缀缓存时直接把 k,v 换成整个缓存(k_cache/v_cache) + block_table, q 就能"看到"缓存的全部历史。
- **decode 用 `flash_attn_with_kvcache`**: 输入只有新 token 的 q(unsqueeze(1) 补序列维), K/V 直接从缓存读。causal=True 保证只看过去。
- 两个函数都是 **flash-attention 库**的融合 kernel: 不显式构造 (N,N) 注意力矩阵, 分块计算, 省显存还快。这是 vLLM 速度的核心之一。

## 10. store_kvcache 的 Triton kernel —— 第一个也是唯一一个手写 kernel

```python
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                         k_cache_ptr, v_cache_ptr, slot_mapping_ptr, D: tl.constexpr):
    idx = tl.program_id(0)                    # 当前处理第几个 token (grid 维度)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return                     # -1 表示该 token 不需要存(如被 mask 掉)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

**Triton 入门(不常见⑫)**: Triton 是 OpenAI 的"用 Python 写 GPU kernel"框架:
- `@triton.jit` 装饰器把函数编译成 GPU kernel
- `tl.program_id(0)`: 当前线程块编号(启动时 `kernel[(N,)](...)` 里的 N 个块)
- `tl.arange(0, D)`: 生成 0..D 的向量索引, 配合指针做向量化读写
- `tl.load`/`tl.store`: 显式内存读写(没有自动魔法, 全部手动)
- `D: tl.constexpr`: 编译期常量——D 在编译时就确定, 可以生成专门的快速代码

这个 kernel 干的事很简单: 把每个 token 的 K/V 从临时张量搬到缓存张量的正确槽位。为什么值得手写? 因为"按 slot_mapping 散列搬运"是 flash-attn 没有提供的, 而用 PyTorch 索引又太慢(每次都要内核启动 + 内存分配)。

## 11. sampler.py —— 三行采样(第 0 层讲过, 复习)

```python
@torch.compile
def forward(self, logits, temperatures):
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))     # ① 除以温度
    probs = torch.softmax(logits, dim=-1)                           # ② 概率
    sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)  # ③ Gumbel
    return sample_tokens
```

`temperatures` 是每序列一个的向量(第 5 层 prepare_sample), `unsqueeze(dim=1)` 把它变成列向量广播。第 ③ 行的 Gumbel-max 技巧(概率 ÷ 指数噪声 → argmax)在 demo_layer0.py 里验证过等价性。

## 12. loader.py —— 权重怎么进模型

```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),        # 权重名映射: 磁盘上叫 q_proj
    "k_proj": ("qkv_proj", "k"),        # 模型里叫 qkv_proj 的 q 段
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}

def load_model(model, path):
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)   # 普通权重直接拷
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
```

- **`glob` 不常见⑬**: 通配符找文件(`*.safetensors`)。模型目录里通常有多个分片文件。
- **`safe_open(file, "pt", "cpu")`**: safetensors 格式(比 pickle 安全、更快), 指定用 torch 张量格式、加载到 CPU(再让各 rank 自己拷)。
- **`get_parameter(name)` 不常见⑭**: `nn.Module` 的按名字查参数方法——因为模型里权重名是嵌套的(`model.layers.0.self_attn.qkv_proj.weight`), 运行时用字符串动态取参。
- **`for...else`**: 内层循环如果被 `break`(命中了 packed 映射)就不执行 else; 没命中(普通权重)才走 else 的普通拷贝。和 scheduler 的 `while...else` 同一招。
- **权重名映射**: 磁盘上 Qwen 的权重叫 `q_proj.weight`, 模型里被"打包"成 `qkv_proj.weight` 的一段——映射表负责改名 + 告诉 weight_loader 这是哪一段(q/k/v 或 gate/up 的 0/1)。

## 13. 自测题

1. 为什么 `qkv_proj` 用一个大线性层而不是三个? GQA 在这里怎么体现?
2. QK-Norm(q_norm/k_norm) 为什么只加在 Q 和 K 上, 不加在 V 上?
3. `register_buffer(persistent=False)` 的 persistent 控制什么?
4. `narrow` 和普通切片 `tensor[a:b]` 有什么不同?
5. RowParallelLinear 为什么最后要 all_reduce? bias 为什么只在 rank 0 加?
6. prefill 时 LM Head 为什么只取每条序列的最后一个位置? 是丢掉信息吗?
7. `for...else` 在 loader 里扮演什么角色?

<details>
<summary>答案</summary>

1. 拼成一个大矩阵乘: 减少 kernel 启动、充分利用带宽(推理引擎标准做法)。GQA 体现在 K/V 部分只有 4 头(256 维)而 Q 是 16 头(1024 维)。
2. Q 和 K 的"匹配质量"决定注意力分数, norm 它们能稳定分数尺度; V 只是被加权求和的内容, 不需要。
3. persistent=True 会进 state_dict(可保存/加载); False 则只当运行时 buffer 用。cos/sin 表是纯函数预计算, 不必存。
4. 都是视图(不拷贝)。narrow 的优势: 语义明确 + 可作用于任意维度(dim 参数), 适合在 weight_loader 这种"通用代码"里用。
5. 输入被切到各 rank, 每个 rank 只算了部分和, all_reduce 求和才是完整输出。bias 只加一次避免重复。
6. 不是丢信息: prefill 的目的是算出整条序列的 KV 并采样第一个新 token, 只有最后位置的 logits 用于采样。中间的 logits 没人要(除非做训练/打分)。
7. 命中 packed 映射就 break 走专用 weight_loader; 没命中(普通权重)走 else 的默认拷贝——一个循环处理两类权重。
</details>

## 14. 练习

- 跑 `python practice/demo_shapes.py`, 对照本文的每个形状。
- 把 `demo_shapes.py` 的 `SEQ_LENS` 改成 `[100, 200, 300]`, 重算一遍, 体会 N 只影响批次维度、不影响 1024 主干。
- 在模型下载后运行 `python -c "from transformers import AutoConfig; ..."` 核对本文的配置数字。
- 读完 `qwen3.py` 后, 用手在纸上画出 `qkv_proj.weight` 的 shape 和 q/k/v 段的切分(提示: 看 QKVParallelLinear 的 shard_offset 计算)。
