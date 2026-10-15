# 第 5 层讲义：ModelRunner —— 推理引擎的性能内核

> 阅读: `engine/model_runner.py`(257 行, 全项目最硬核的文件)
> 前置: 第 3 层(调度) + 第 4 层(模型)。本层把两者在 GPU 上焊起来。
> 本层内容全部依赖 GPU/CUDA, 无法本机演示 —— 用"读代码 + 第 4 层维度计算器"理解。

---

## 0. ModelRunner 的职责全景

```
构造时(一次):
  init_process_group(NCCL) → 建模型 → 加载权重 → warmup(空跑一次)
  → 按剩余显存分配 KV Cache → 捕获 CUDA Graph

每次 step(循环):
  prepare_prefill / prepare_decode: 把 Python 列表拼成 GPU 张量
  run_model: 前向( eager 或 CUDA Graph 回放)
  sampler: 采样出新 token
  reset_context: 清空全局上下文
```

对应 `llm_engine.step()` 里的那一行 `self.model_runner.call("run", seqs, is_prefill)`。

## 1. 构造: 初始化 GPU 环境

```python
dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
torch.cuda.set_device(rank)                     # 本进程用第 rank 块卡
default_dtype = torch.get_default_dtype()
torch.set_default_dtype(hf_config.dtype)        # 默认 dtype 设为模型 dtype (bf16)
torch.set_default_device("cuda")                # ★ 之后所有 torch.empty() 默认放 GPU
self.model = Qwen3ForCausalLM(hf_config)        # 建模型(此时所有参数落在 GPU 上)
load_model(self.model, config.model)            # 从 safetensors 拷权重
self.sampler = Sampler()
self.warmup_model()
self.allocate_kv_cache()
if not self.enforce_eager:
    self.capture_cudagraph()                    # 捕获 CUDA Graph
torch.set_default_device("cpu")                 # 用完恢复全局默认
torch.set_default_dtype(default_dtype)
```

**不常见①: `torch.set_default_dtype` / `torch.set_default_device`**
全局默认值, 影响"没显式指定的所有张量创建"。这里在建模型前把默认设备设成 cuda、dtype 设成 bf16, 于是 `nn.Parameter(torch.empty(...))` 直接落在 GPU 上、还是 bf16——不用逐个 `.to("cuda")`。用完立刻恢复, 避免污染其他代码(比如 tokenizer 的 CPU 张量)。

**`torch.cuda.set_device(rank)`**: 多卡时每个进程必须显式选卡, 否则都用卡 0(会 OOM 或算错)。

**NCCL 初始化**: `dist.init_process_group("nccl", ...)` 第 6 层细讲, 先知道"所有 rank 必须同时调用, 才能互相认识"。

## 2. warmup_model —— 先空跑一次

```python
def warmup_model(self):
    torch.cuda.empty_cache()                    # 清碎片
    torch.cuda.reset_peak_memory_stats()        # 归零峰值统计
    seq_len = min(max_num_batched_tokens, max_model_len)
    num_seqs = min(max_num_batched_tokens // seq_len, self.max_num_seqs)
    seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]   # 假序列
    for seq in seqs:
        seq.num_scheduled_tokens = seq_len
    self.run(seqs, True)                        # 真跑一次前向!
    torch.cuda.empty_cache()
```

为什么必须 warmup:
1. **触发 lazy 分配**: PyTorch 的显存分配是"用时才要"——跑一次才知道模型实际占多少。
2. **拿到峰值**: `reset_peak_memory_stats` 后跑一次, `memory_stats()["allocated_bytes.all.peak"]` 就是模型 + 激活的真实峰值——下一步分配 KV Cache 要用它算"还剩多少"。
3. 注意这里绕过了调度器(直接构造 Sequence、直接 `run`), 因为 warmup 不需要真实请求。

## 3. allocate_kv_cache —— 用"剩下的显存"决定缓存多大

```python
free, total = torch.cuda.mem_get_info()
used = total - free
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
num_kv_heads = hf_config.num_key_value_heads // self.world_size
head_dim = getattr(hf_config, "head_dim", hidden_size // num_attention_heads)
block_bytes = 2 * num_hidden_layers * block_size * num_kv_heads * head_dim * dtype.itemsize
config.num_kvcache_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
self.kv_cache = torch.empty(2, layers, num_blocks, block_size, num_kv_heads, head_dim)
```

**不常见②: `torch.cuda.mem_get_info()`** → (free, total)。**不常见③: `memory_stats()["allocated_bytes.all.peak"]`** → PyTorch 分配器的内部统计, 可拿到"历史最高占用水位"。

**公式解读**: KV 预算 = 总显存 × 利用率 − 已用 − 峰值 + 当前。
直觉: 假设推理时峰值占用会重演, 所以把 peak 预留出来; 已用里有一部分不在 peak 里(当前分配), 加回去。剩下的空间 ÷ 每块字节数 = 块数。

**`block_bytes` 公式**(第 4 层 demo_shapes 第 6 节验证过): 每块 = K 和 V 两份 × 层数 × 块大小 × KV 头数 × head_dim × 字节。Qwen3-0.6B: 7.0MB/块。

**把缓存注入每个 Attention 层**:
```python
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]   # 视图, 不拷贝
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```
`nn.Module.modules()` 深度遍历所有子模块; 用 `hasattr` 找到 Attention 层, 把大张量**按层切片**(视图)塞进去。不用改模型构造函数——这是"运行时注入"的灵活做法。

## 4. prepare_prefill —— 把一批序列拼成 GPU 输入(核心!)

```python
for seq in seqs:
    start = seq.num_cached_tokens
    seqlen_q = seq.num_scheduled_tokens
    end = start + seqlen_q
    seqlen_k = end
    input_ids.extend(seq[start:end])            # 所有序列的 token 拼成一维
    positions.extend(range(start, end))         # 每个 token 的绝对位置
    cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)   # 前缀和 → 序列边界
    cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
    max_seqlen_q = max(seqlen_q, max_seqlen_q)
    ...
    for i in range(start_block, end_block):     # slot_mapping: token → 缓存槽位
        slot_start = seq.block_table[i] * self.block_size
        ...
        slot_mapping.extend(range(slot_start, slot_end))
```

**① 变长批次的打包方式**: 不 pad 到统一长度(省显存), 而是把所有 token **拼成一维** `(N_total,)`, 用 `cu_seqlens`(前缀和)告诉 flash-attn"每条序列从哪到哪"。这是 varlen attention 的标准输入格式(第 4 层讲过)。

**② `cu_seqlens_q` 和 `cu_seqlens_k` 为什么分开**: 前缀缓存时, K 能看到的历史比 Q 长(q 只看自己这条的已算部分, k 是缓存里的全部)——两者边界不同, 所以各存一份。

**③ slot_mapping**: 每个 token 的 KV 要写到缓存张量的哪个槽位。规则: 块 id × 块大小 + 块内偏移。首尾块可能不满, 要处理"首块从 start%block_size 开始、末块到 end 为止"的边界。**这段循环是整个 prepare 里最容易写错的地方**, 值得逐行读。

**④ `pin_memory=True` + `.cuda(non_blocking=True)` 不常见④**:
```python
input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
```
- `pin_memory=True`: 把 CPU 张量钉在**页锁定内存**, DMA 才能直接搬运
- `non_blocking=True`: 拷贝异步进行, 不阻塞 CPU —— CPU 可以继续准备下一批数据, 与 GPU 计算重叠
这是高性能推理/训练的标配双件套。

**⑤ block_tables 何时需要**: `if cu_seqlens_k[-1] > cu_seqlens_q[-1]` —— 有前缀缓存(要读历史)时才构造, 并传给 flash-attn 的 `block_table` 参数(第 4 层 attention.py)。

## 5. prepare_decode —— 每序列只喂 1 个 token

```python
for seq in seqs:
    input_ids.append(seq.last_token)                          # 输入: 最后 1 个 token
    positions.append(len(seq) - 1)                            # 位置: 末尾
    context_lens.append(len(seq))                             # 已缓存长度
    slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
block_tables = self.prepare_block_tables(seqs)                # 补齐到等长, -1 填充
```

**`prepare_block_tables`**: 各序列块数不同, pad 成 (N_seqs, max_blocks) 的稠密张量, 空位填 -1(flash-attn 约定 -1 = 无效块)。decode 是"每序列独立查自己的块表", 所以必须稠密化。

**slot 计算**: 新 token 的槽位 = 最后一块的起始 + 块内已有 token 数。`last_block_num_tokens` 是第 2 层讲过的 property。

## 6. run_model —— eager 还是 CUDA Graph?

```python
@torch.inference_mode()
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))   # 普通前向
    else:
        bs = input_ids.size(0)
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]       # 选 ≥bs 的最小图
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids          # 拷进静态缓冲
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)              # 先清空再填前 bs 行
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()                                    # ★ 回放!
        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

**不常见⑤: `torch.inference_mode()`** —— 比 `no_grad()` 更彻底: 连 autograd 的图追踪机制都关掉, 推理专用, 更快。

**为什么 decode 用 CUDA Graph、prefill 不用?**
- decode 每轮只算几十~几百个 token, **瓶颈是 Python/kernel 启动开销**(launch-bound): 每轮要启动上百个 kernel, 每个都有 CPU 开销
- prefill 一次算几千 token, 计算量大, 启动开销占比小
- 所以: **prefill 走 eager, decode 走 graph 回放**

**CUDA Graph 原理(不常见⑥)**: 第一次跑前向时, 把每个 kernel 的"启动指令"全部录下来(捕获), 之后每次 `graph.replay()` 直接重放整串 kernel —— Python 层一次调用, GPU 上连续执行, 启动开销几乎归零。

**静态缓冲(static buffer)**: 重放时输入不能变张量对象, 只能改**内容**——所以预先分配好最大尺寸的 `graph_vars` 缓冲, 每次把新数据 `[:bs]` 拷进去再回放。`fill_(-1)`/`zero_()` 是"清空超出部分"防止读到脏数据。

**`next(x for x in self.graph_bs if x >= bs)` 不常见⑦**: 生成器表达式取第一个满足条件的值——选"≥ 当前 batch 的最小已捕获图"。

## 7. capture_cudagraph —— 捕获过程

```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))   # 按批大小分档捕获
for bs in reversed(self.graph_bs):
    graph = torch.cuda.CUDAGraph()
    set_context(False, slot_mapping=..., context_lens=..., block_tables=...)
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # ① warmup
    with torch.cuda.graph(graph, self.graph_pool):               # ② 捕获
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
    if self.graph_pool is None:
        self.graph_pool = graph.pool()                           # ③ 共享内存池
    self.graphs[bs] = graph
    torch.cuda.synchronize()
```

- 捕获必须**先 warmup 再 capture**(同一批 kernel, 让分配器稳定)
- `graph.pool()`: 捕获期间分配的内存池, 后续捕获复用, 省显存
- 分档(1,2,4,8,16,32,...)捕获: 运行时不精确匹配, 取"≥bs 的最小档", 用不满的槽位填充(所以 graph_vars 要 fill_/zero_)
- **为什么需要全局 context(utils/context.py)**: 捕获要求"每次调用传一模一样的张量对象"。如果 cu_seqlens/slot_mapping 通过函数参数传, 每次都是新张量 → 无法捕获。所以 `set_context(...)` 把它们塞进模块级全局变量, Attention/LMHead 内部 `get_context()` 读取 —— 捕获和回放用的是同一个对象。这是"用全局变量换 CUDA Graph 兼容性"的典型权衡(单线程推理环境, 全局变量无并发问题)。

## 8. run —— 组装

```python
def run(self, seqs, is_prefill):
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None   # 只有 rank0 用
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()                       # 每步结束清空全局上下文
    return token_ids
```

- `prepare_sample`: 把每序列的 temperature 收成张量(第 2 层讲过: 反序列化后的 worker 序列没有 temperature 属性, 但这里只在 rank 0 执行——原序列)
- `reset_context()`: 全局上下文是"每步一换"的临时状态, 用完必须清, 否则下步读到脏数据
- `.tolist()`: GPU 张量 → Python list(调度器 postprocess 用)

## 9. 自测题

1. 为什么必须 warmup 之后才能准确分配 KV Cache?(提示: 峰值统计)
2. `pin_memory=True` + `non_blocking=True` 解决了什么问题?
3. prefill 为什么走 eager、decode 为什么走 CUDA Graph?
4. CUDA Graph 为什么要求"静态缓冲 + 全局 context"? 不这样会怎样?
5. `block_tables` 在 prefill 什么时候需要构造? decode 呢?
6. `reset_context()` 为什么放在 run() 末尾?

<details>
<summary>答案</summary>

1. PyTorch 显存是 lazy 分配的, 不跑一次不知道模型真实占用; 峰值统计也要先归零再跑才有意义。
2. pin_memory 让 DMA 直接搬运(不用先拷到页锁定中转区), non_blocking 让拷贝与 CPU 后续工作/GPU 计算重叠。
3. decode 是 launch-bound(每轮 token 少、kernel 启动开销占比大), 回放省掉 Python 层开销; prefill 计算量大, 启动开销可忽略, 且 prefill 形状多变不适合捕获。
4. 捕获的就是"这组张量对象上的 kernel 序列"; 换张量对象就失效。静态缓冲 + 全局 context 保证捕获与回放用同一批对象。
5. prefill 有前缀缓存(要读历史)时; decode 永远需要(查块表)。prefill 无缓存时 k 就是本批, 不需要块表。
6. 全局上下文是每步的临时状态(如 slot_mapping), 不清会污染下一步。
</details>

## 10. 练习

- 读 `prepare_prefill` 的 slot_mapping 循环, 手算一个例子: block_size=4, 序列 token 数=9, block_table=[0,1,2], start=0, end=9 → slot_mapping 应该是什么?(提示: 块0全量、块2只到偏移1)
- 读 `capture_cudagraph`, 说出 `fill_(-1)` 和 `zero_()` 分别保护了哪些张量的哪些区域。
- 环境就绪后: `enforce_eager=True` vs `False` 各跑一遍 example.py, 对比生成速度(这是本层练习的唯一 GPU 实验)。
