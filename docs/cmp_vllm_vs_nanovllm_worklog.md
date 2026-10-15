# vLLM 对比压测工作日志（逐轮记录）

> 本文档按对话轮次记录整段工作：出发点、做了什么、发现什么、遇到什么问题、怎么解决、有什么收获。
> 体例与 `docs/sm89_worklog.md` 一致。
> **主题：用同一份数据、同一块卡、同一套口径，把 nano-vllm-plus 和 vLLM 放一起压，看差在哪。**
> 环境：WSL2 Ubuntu / conda `ml` / **vLLM 0.31.0** / torch 2.13.0+cu130 / transformers 5.17.0 /
> 模型 `/home/adam/models/Qwen3-0.6B` / RTX 4060 Laptop (sm_89, 8 GB)。

---

## 0. 一句话总结

要压的是"TTFT 和总吞吐量，在线和离线都要"。最后压出来了，**但答案是反直觉的**：

**我们的引擎本身比 vLLM 快，慢的是我们套在它外面的服务层。**

- **离线（只比吞吐）**：基本打平。N=32 我们高 4.5%，N=128 差 0.4%。
- **在线 TTFT**：打平甚至我们更好（c=1 −14%、c=8 −9%、c=32 持平、c=48 −7%）。
- **在线 TPOT（解码）**：c≥32 时我们慢 **22–23%**，这**完全**解释了吞吐上 17% 的差距。
- **决定性对照**：同样的 batch 32、同样的形状，**离线我们 15.54 ms/step，比 vLLM 的 16.39 还快**；
  一过 gRPC 服务层，我们涨到 20.95 ms（**+34.8%**），vLLM 只从 16.39 涨到 17.22（**+5.1%**）。
  ⇒ 差距 **100% 产生在服务层**，与 kernel 无关。根因是我们把引擎放在和 gRPC servicer
  共享 GIL 的同一个进程里，而 vLLM 的 `EngineCore` 是独立进程。

这段工作里我先后犯了 **5 个错**，每一个都是被后续测量或用户追问推翻的（见 §3 错误账本）。
其中最有价值的一个不是我发现的——是用户问的："**离线是不是没有 TTFT？**"

---

## 第 0 轮：读 worklog，然后压测

### 用户说什么
> 读一下 sm89_worklog.md，看看这个仓库经历了什么。然后，用同样的输入 prompt 长度和输出长度，构造一组数据
> （注意前缀不要相同，以防前缀复用），压测一下 wsl 上 conda ml 环境中的 vllm 的运行效率，以及咱们自己项目的
> 运行效率（TTFT 和总吞吐量）；在线和离线方式都压一下；prompt 采用你认为对 qwen3 0.6B 合适的长度。

### 我做了什么

**1）读文档**：`docs/sm89_worklog.md`（739 行，实际不在仓库根目录）+ 三份配套文档。
worklog 的要点：为 RTX 4060 写的 Triton paged decode attention **做成了、正确、已贴 DRAM 墙，但收益 ~1.00x**；
真正的 3.4× 来自服务层一个跟 GPU 无关的配置值 `max_workers=10`；作者先后**推翻了自己 6 个结论**。

**2）探环境**（WSL `conda ml`）：

| | 版本 |
|---|---|
| vLLM | **0.31.0** |
| torch | 2.13.0+cu130 |
| transformers | 5.17.0 |
| 模型 | `/home/adam/models/Qwen3-0.6B` |
| GPU | RTX 4060 Laptop，8188 MiB |

**第一个硬约束**：桌面程序共用 GPU。vLLM 报 `Free memory on device (6.93/8.0 GiB) on startup is less than
desired GPU memory utilization (0.9, 7.2 GiB)` 直接拒绝启动。⇒ 两边统一 **`gpu_memory_utilization=0.80`**。

**3）工具链**（全部新增在 `benchmarks/`）：`cmp_common.py`、`cmp_offline_nanovllm.py`、`cmp_offline_vllm.py`、
`cmp_online_client.py`、`cmp_serve_nanovllm.py`、`cmp_report.py`。

**4）数据集设计**（用户两条要求都直接决定了设计）：

- **等长**：prompt = **512 token**，output = **128 token**，`ignore_eos=True` 保证输出长度精确。
  选 512 的理由：Qwen3-0.6B 是 28 层 / hidden 1024 / 8 KV head / head_dim 128；
  按 worklog 的交叉点 `B* = W/L`，ctx 512 时已进入 KV 主导区，TTFT 和吞吐同时有信号，
  而不是其中一个退化成常数。
- **前缀不能相同**：每条请求用**全随机 token id**（安全区间 `[1000, vocab-1000)`），
  **从第 0 个 token 就互不相同**。这比"首块不同"更强，在任何 block 粒度下都杜绝复现。
- **每次测量换数据集**：这条来自一个关键发现——

> **`BlockManager.hash_to_block_id` 从不清空。** 请求结束后 block 回收到 free list，但
> `token_ids` 和 hash 条目还在，`find_cached_prefix_blocks` 照样能命中。所以**重跑同一批 prompt
> 会变成前缀缓存命中**，静默污染第二次测量。⇒ 数据集 seed 由 `(engine, mode, concurrency, rep)` 决定，
> 任何两次运行都不可能共享 token。

**5）配置对齐的第一次发现**：vLLM 的默认值和我们差得很远。

| | nano-vllm 默认 | vLLM 默认 |
|---|---|---|
| `max_num_batched_tokens` | 16384 | **2048** |
| `max_num_seqs` | 512 | **128** |
| `max_model_len` | 4096 | 模型自身 **40960** |
| block_size | 256（assert 强制） | **16** |

512-token prompt 下 vLLM 的 2048 意味着每步只能 prefill 4 条 ⇒ 决定对齐到我们的默认值
（对齐方向是"把 vLLM 拉到 nano-vllm 的默认配置"）。

**6）验证无前缀复用**（每次都做）：我们侧 `prefix_cache_hits = 0 / 216 queries`；
vLLM 侧 `vllm:prefix_cache_hits_total = 0`（queries 110592）。全部 0 命中 ✅

### 收获
工具链和数据集站住了，但""TTFT"这个指标本身的定义我还没想清楚——这埋下了后面第 4 轮的大更正。

---

## 第 1 轮：第一轮完整压测（9 个点）

### 我做了什么
跑完全部 9 个点。当时的初步结论：

| 引擎 | 方式 | 并发 | TTFT mean | wall | 生成 tok/s |
|---|---|---|---|---|---|
| nano-vllm | 离线 | 32 | 677.0 | 2.784s | 1471.2 |
| vLLM | 离线 | 32 | 677.9 | 2.919s | 1403.0 |
| nano-vllm | 在线 | 1 | 30.6 | 61.8s | 132.5 |
| vLLM | 在线 | 1 | 34.8 | 66.8s | 122.7 |
| nano-vllm | 在线 | 8 | 160.5 | 12.45s | 658.1 |
| vLLM | 在线 | 8 | 174.7 | 12.25s | 668.6 |
| nano-vllm | 在线 | 32 | 698.0 | 7.03s | 1164.7 |
| vLLM | 在线 | 32 | 638.7 | 6.07s | 1350.3 |
| vLLM（stock 2048） | 在线 | 32 | 394.1 | 6.18s | 1326.1 |

**我当时报的结论**："离线打平、低并发我们略优、c=32 vLLM 领先 16%"。

**同时埋了两个错**：
1. 我把离线那两行的 TTFT 当指标报了（第 4 轮被用户推翻）。
2. 我用"离线 decode-only 1928.7 tok/s vs 在线 decode-only 1282.9 tok/s"推了个"**复现了 worklog B4 的 33%
   服务层损耗**"。这个推算后来被判无效（拿离线退化标量减在线分布均值，两边不是同一个量），
   但**方向恰好是对的**——第 10 轮用干净的方法重新做了一遍。

---

## 第 2 轮：离线 n=8 的异常 → 挖出**冷启动**这个真缺陷

### 我做了什么
为了做离线 batch 曲线跑了 n=8/16/32，结果 n=8 荒谬：

```
ours offline n=8 : TTFT 1169.18 ms, wall 2.322s, gen 441.0 tok/s
ours offline n=16: TTFT  398.15 ms, wall 1.885s
ours offline n=32: TTFT  676.96 ms, wall 2.784s
```

**n=8 的 TTFT 比 n=32 还差，wall 比 n=16 还长** —— 这不可能是 batch 效应。

写 `diag_n8.py` 在同一进程里连续测，直接量出前几步：

| 第几次真实 prefill | 耗时 |
|---|---|
| 第 1 次（n=4） | **2627.4 ms** |
| 第 2 次（n=8） | **1167.8 ms** |
| 第 3 次 | **167.0 ms** |
| 第 4 次 | 167.8 ms |
| n=16 | 340.5 ms |
| n=32 | 675.8 ms |
| n=8（在大 batch 之后） | 168.3 ms |

**根因（代码级确认）**：`ModelRunner.__init__` 里顺序是

```python
self.warmup_model()          # ← 先
self.allocate_kv_cache()     # ← 后
```

而 `Attention.forward` 是

```python
if k_cache.numel() and v_cache.numel():
    store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
```

预热时 KV cache 还没分配，`k_cache` 是 `torch.tensor([])` ⇒ `numel()==0` ⇒ **`store_kvcache` 整条路被跳过**；
而且 `prepare_prefill` 里有 `if not seq.block_table: continue`，prefill 时的 slot_mapping 是空 tensor。
**所以引擎自带的 warmup 从来没走过真实的 paged prefill 路径**，第一次真实请求要付 ~2.6 s 的一次性代价。

**修复**：预热带改成"按测量形状跑 2 轮、用不同数据集"。n=8 从 1169 → 167.5 ms，和诊断出的干净值 167.0 完全吻合。

### 收获
一个真实的、可复现的产品缺陷：**预热不覆盖真实路径**。对一个服务来说这值 2.6 s 的首请求延迟。
（第 6 轮用户决定不把它作为指标记录，但这条发现保留在文档里。）

---

## 第 3 轮：改完预热后，vLLM 离线的 TTFT 崩了

### 我做了什么
按新预热协议重跑，vLLM 离线的"TTFT"全部变成 **等于 wall**：

| n | vLLM 离线 "TTFT" | wall |
|---|---|---|
| 8 | 1496.4 ms | 1.497 s |
| 16 | 1949.3 ms | 1.950 s |
| 32 | 2932.4 ms | 2.933 s |

wall 和上一轮一致（吞吐数据没问题），只有 TTFT 检测坏了。我开始查：怀疑是预热和测量共用了同一个
`SamplingParams` 对象，`llm.generate()` 把它改成了 `FINAL_ONLY` 之类。

我写了个诊断脚本逐 step 打印输出节奏……**用户在这里叫停并问了进度**。

### 收获
没有收获，因为我查错了方向。真相是第 4 轮用户一句话点破的：**这个量本来就不该存在**。

---

## 第 4 轮：**用户推翻了我的口径** —— "离线是不是没有 ttft"

### 用户说什么
> 我的指令似乎犯了个错误，离线是不是没有 ttft

### 我做了什么（这是整段工作最重要的更正）

**用户是对的。** TTFT 的定义是"从请求提交到第一个 token 到达客户端"。离线/批处理里：

- 所有请求在 t0 一次性提交，调用者只在 `generate()` 返回时拿到**完整**结果；
- 中间**没有任何"首个 token"事件可观测**；
- 离线模式下的 TTFT 只是引擎内部的一个量。

**而且我自己的数据早就暴露了这一点**（我当时没看出来）：

```
nano-vllm 离线 n=32 : TTFT mean 677.0  p50 677.0  p99 677.0   ← 三个数一模一样
vLLM      离线 n=32 : TTFT mean 677.9  p50 697.9  p99 697.9
```

**一个"分布"退化成单点，说明 32 条请求全部在同一个 step 拿到首个 token** —— 这不是 per-request
延迟分布，而是**整个 batch 的 prefill 完成时间**，一个标量。我把它当 TTFT 报、还给了 p99，是错的。

**两个独立旁证，说明这是概念问题而非实现问题**：
1. 行业惯例：vLLM 自己的 `benchmark_throughput.py`（离线）**只报吞吐**，`benchmark_serving.py`（在线）才报 TTFT。
2. 实测 vLLM 离线路径的 `RequestOutput.metrics` 就是 **`None`** —— **vLLM 自己就不为离线提供 TTFT**。
   我为了凑出这个数，才去手写 `LLMEngine.step()` 循环打时间戳——这正是下面第 5 轮的问题。

### 更正后的口径

| | 该报什么 | 不该报什么 |
|---|---|---|
| **离线** | wall、生成吞吐、总吞吐、prefill 吞吐 | per-request TTFT / p50 / p99 |
| **在线** | **TTFT mean/p50/p99** + TPOT + 吞吐 | — |

### 收获
**指标的可观测性要先于可测量性。** "我能从内部抠出一个数"不等于"这个数对用户存在"。
一个退化成单点的"分布"（`p50 == mean == p99`）就是口径错掉的信号，我当时看见了这个数字却没读懂它。

---

## 第 5 轮：**用户追问 vLLM 侧的口径** —— 是原生还是魔改

### 用户说什么
> 现在，配置可以你自己决定，但是用法必须是官方用法，然后在线的比较 ttft 和吞吐量，离线的只比较吞吐量，你认为是否合理

（在这之前用户还问了："vllm 方面的口径是什么，是官方原生实现，还是你魔改的？"）

### 我做了什么——**先把老底交代清楚**

| | 引擎代码 | 调用方式 | 是否官方 |
|---|---|---|---|
| vLLM 在线 | 未改 | `vllm serve` + OpenAI `/v1/completions` streaming | ✅ 完全原生 |
| **vLLM 离线** | 未改 | **脚本里直接戳 `llm.llm_engine.add_request()/step()`** | ❌ **不是**，官方是 `llm.generate()` |
| nano-vllm 在线 | 未改 | `serve()` + gRPC 流式 `Infer` | ✅ 原生 |
| nano-vllm 离线 | 未改 | `add_request()+step()` | ✅ 原生（它自己的 `generate()` 就这么干的） |

**结论：两边源码一行都没改**；但 vLLM 的**配置**是我改过的（4 个默认值，见 §4.4），
而且**离线那一侧我用了非官方用法**——那个魔改唯一的产出就是第 4 轮被判死的 TTFT。

### 我对新协议的意见（**没有全盘同意，补了 5 个洞**）

用户的协议（配置我定 / 用法必须官方 / 在线报 TTFT+吞吐 / 离线只报吞吐）**大方向我同意**，而且它正好和
vLLM 自己的 benchmarking 惯例一致。但我说了 5 个必须补的洞：

1. **在线只报一个 TTFT 数字会被配置支配**。证据就是当时手里的 stock 对照点：
   **同一个 vLLM，只换 `max_num_batched_tokens`，TTFT 从 682 ms 变 388 ms（−43%），吞吐几乎不变。**
   ⇒ 必须加 **TPOT = (e2e − ttft)/(out_len − 1)**，而且它正好能接回 worklog 的 roofline
   （每 token 112 KiB ÷ 250 GB/s ≈ 5.07 ms/token 地板）。
2. **每点必须重复、且两引擎交替跑**。这不是洁癖——`vLLM 在线 c=1` 有一次跑出 wall 230.7s / 35.5 tok/s，
   而同样配置的正常值是 66.8s / 122.7 tok/s（**差 3.5 倍**），桌面共用 GPU 的干扰。
3. **"官方用法"的边界要写清楚**：nano-vllm 的 CLI 没暴露 `gpu_memory_utilization`，我得走它的官方
   `serve()` 编程入口透传 Config 字段。这和"戳内部 step()"性质不同，但必须披露。
4. **配置对齐的代价要披露**：KV 池不等大、碎片差异。
5. **"用满资源"和"公平对比"可能冲突**（这条在第 9 轮变成了现实，见下）。

### 收获
在"用户提协议"和"照做"之间，我选择了补洞而不是点头。第 1 条（TPOT）和第 2 条（重复）后来都真的救了场。

---

## 第 6 轮：定案 —— 冷启动不进表、KV 池的差异到底是什么

### 用户说什么
> 1. 我认为不用记录冷启动，直接记录稳定态 2. 你说的 kv 池不等大是什么意思？

### 我做了什么

**冷启动**：同意去掉。但保留两点：
- 两边都做**同样的显式预热**（按测量形状 2 轮），否则稳态数字本身就是假的；
- 在方法一节写**一句话**披露"nano-vllm 自带 warmup 不覆盖真实 prefill 路径"，作为协议说明而非指标。

**KV 池不等大**——两个引擎都被告知"用 0.80 的显存"，但算出来的池子不一样：

| | block_size | block 数 | 池容量 (token) | 池容量 (GiB) |
|---|---|---|---|---|
| nano-vllm | **256** | 133 | 133×256 = **34,048** | 3.63 |
| vLLM | **16** | 2,393 | 2,393×16 = **38,288** | 4.09 |

（每 token KV = 112 KiB：28 层 × 2 × 8 KV head × 128 × 2 B。）

**两个原因，性质不同**：
1. **记账公式不同** → vLLM 多拿 **12.6% 字节**。我们是 `total×util − 已用 − 峰值激活 + 当前`
   （"已用"含桌面占用），vLLM 是自己 profile 出权重/激活/CUDA graph 后再算。
2. **block 粒度造成的碎片** → 序列长 640 token 时：我们占 3×256 = **768 槽（浪费 20%）**，
   vLLM 占 40×16 = **640 槽（0 浪费）**。

按**有效并发**算：我们 `133/3 = 44 条`，vLLM `2393/40 = 60 条`（vLLM +34%）。

**为什么它不影响第一轮结论**：c=32 时两边需求都远没到顶（32 < 44 < 60）。
但**它解释了第 9 轮 c=48 的不公平**。

### 收获
把"KV 池不等大"拆成了**两个独立机制**（字节记账 vs 碎片），这决定了后面"对齐什么"的选择。

---

## 第 7 轮：**用户问 block size 能不能对齐**

### 用户说什么
> vllm 的 block size 不能对齐吗

### 我做了什么
技术上行（vLLM 有 `--block-size`，256 是 16 的倍数，需实测 backend 是否接受）。**但我建议不动，理由有量化证据**：

**对齐 block size 改不了池容量**（容量是字节决定的）：

| | KV 字节 | block_size | block 数 | 容量 (token) |
|---|---|---|---|---|
| nano-vllm | 3.63 GiB | 256 | 133 | 34,048 |
| vLLM | **4.09 GiB** | 16 → **256** | 2393 → **149** | 38,288 → **38,144** |

**而且方向是偏袒我们的**：

| 方案 | vLLM 有效并发 | 我们 | vLLM 优势 |
|---|---|---|---|
| 各用默认 | **59.8** | 44.3 | **+35%** |
| vLLM 也对齐 256 | **49.7** | 44.3 | **+12%** |

把 vLLM 从它自己调优的 16 拽到 256，等于**把它的优势从 +35% 削到 +12%**。这不是中性对齐，是给自己加分。
反方向也走不通（我们 `assert % 256 == 0` 锁死，改成 16 要动源码）。

**更根本的一点**："同样的显存预算能换出多少可用 KV"本身就是引擎效率的一部分，拉平它等于把要比较的东西藏起来。

我给的替代方案：主表各用默认 + 披露容量表；可选加 `--num-gpu-blocks-override` 对齐**容量**这个真正的变量，
或加一个高并发点让约束真的咬住。

### 收获
"能对齐"和"该对齐"是两件事。一个技术上可行、方向上偏袒自己的操作，即使做了也要标出来。

---

## 第 8 轮：**用户问 `assert kvcache_block_size % 256 == 0` 合不合理**

### 用户说什么
> 咱们项目源码干啥限制 kvcache block size 256，你认为合理吗

### 我做了什么（查源码，不凭记忆）

**根因不在 nano-vllm，在 flash-attn。** `nanovllm/config.py:22` 的 assert 是为了满足上游约束：

```
flash_attn_interface.py:1531（flash_attn_with_kvcache 的 docstring）
    page_block_size must be a multiple of 256.
```

而我们的默认解码路径正是调它（`attention.py:105` `flash_attn_with_kvcache(..., block_table=...)`）。
**理由正当**。

**但强制方式我认为不合理，三个具体问题**：

**1. 范围过宽**：256 只来自**一个**后端。项目有第二个解码后端 `NANOVLLM_DECODE_ATTN=triton`，
它自己的 kernel **明确处理非整除的 block size** —— `decode_attention.py:152` 的注释：
> `token_offs < block_size` keeps BLOCK_N from spilling into the next physical block when block_size is not a multiple of it.

但 assert 在 `Config` 里是无条件的，Triton 后端照样被管死。
（诚实标注：prefill 命中前缀缓存时也走 flash-attn 的 `block_table`，而 `flash_attn_varlen_func`
的 docstring **完全没提**这个约束——它签名接受 `block_table` 但文档没写限制，所以"换 Triton 就彻底自由"
我没法断言，需要单独验证。）

**2. 它只允许最浪费的那一端**：`% 256 == 0` 允许 256/512/768…，**恰好禁止了 16/32/64/128**。代价可量化：

| | 每条 640-token 序列占用 | 浪费 | 池子能装 |
|---|---|---|---|
| block 256（现状） | 3 块 = **768 槽** | **128 槽 (20%)** | 133 ÷ 3 = **44** |
| 若 block 16/32… | 640 槽 | 0 | 34,048 ÷ 640 = **53** |

**有效并发被碎片吃掉 17%。** 而 worklog 自己的结论是"短上下文下 batch 是唯一能提吞吐的手段"
⇒ **这是一个自己给自己设的 −17% 并发天花板**。

**3. 更糟：它允许的配置并没有被端到端支持**。256 在三处**独立硬编码**，与 config 无关：

| 位置 | 内容 |
|---|---|
| `sequence.py:16` | `block_size = 256`（类默认，被 `LLMEngine` 覆盖，无害） |
| `gateway.py:22 / 111` | `block_size: int = 256` |
| `redis_router.py:24 / 117` | `compute_prefix_hash(..., block_size: int = 256)` |

**gateway 是独立进程、看不到模型和 Config**，而 `__main__.py` 调 `serve_gateway()` 时**根本没传 `block_size`**
⇒ 网关永远用 256。但引擎侧 `EngineWorker._block_size` 取自 `scheduler.block_size`（config 值）。
**所以 assert 允许你把 `kvcache_block_size` 设成 512，而网关会按 256 算前缀哈希、引擎按 512 算，
前缀路由静默失配。** 一个 `% 256 == 0` 的 assert 允许 512、而另外三个模块假设 256，
比直接写 `== 256` 更糟——它**宣称了一个并不存在的自由度**。

**建议**：要么 `assert == 256`（诚实），要么按后端加约束 + 验证 varlen paged 路径是否也需要 256
（这是能不能降到 64/128 的真正卡点）+ 把 block size 收成单一事实来源。
用户决定：**先写进报告，不改代码。**

### 收获
这条把一个"看起来像随手写的 assert"追到了上游 kernel 文档 + 三处硬编码的一致性缺口，
并且**它和第 6 轮的碎片、第 9 轮 c=48 的不公平是同一个根因**。

---

## 第 9 轮：显存对齐 —— 以及"用满显卡"把我自己坑了

### 用户说什么
> 两边对齐同样大的显存，好实现吗？
> …主表用精确对齐的 kv 吧，让 vllm 对齐我们的项目，你觉得咋样
> 可以，开跑，配置要在对齐的情况下，**尽可能地把这张显卡利用率用上**

### 我做了什么

**1）分析三种"显存对齐"**：
- **总显存对齐**：不好做（激活峰值、CUDA graph 池都没法直接指定），而且**没必要**——约束并发的是 KV，不是总显存。
- **KV 字节对齐**：**好做**。vLLM 有精确字节开关 `--kv-cache-memory`；我们需要 2 行改动。
- **且能做到 token 容量精确相等**：因为 `256 % 16 == 0`，只要字节取 **28 MiB（我们一个 block）的整数倍**，
  两边容量就完全相等。

**2）那个"未生效的字段"**：`config.num_kvcache_blocks` 本来就有（默认 -1），但 `allocate_kv_cache` 里
**无条件覆盖它**：

```python
config.num_kvcache_blocks = int(total * util - used - peak + current) // block_bytes
```

`used` 是**桌面程序当时占了多少显存** ⇒ **池子大小每次都不同、不可复现**。
所以让它尊重正数覆盖不是加 hack，是**补完一个已有但未生效的开关**：

```python
if config.num_kvcache_blocks <= 0:
    config.num_kvcache_blocks = int(total * util - used - peak + current) // block_bytes
```

**3）按用户要求探测"能顶到多大"**：

| 候选 | nano-vllm | vLLM |
|---|---|---|
| 144 blocks (4032 MiB, 36,864 tok) | OK，剩 2203 MiB | OK（util 0.85），剩 1648 MiB |
| 152 blocks (4256 MiB, 38,912 tok) | OK，剩 2011 MiB | OK（util 0.85），剩 1362 MiB |
| 168 blocks (4704 MiB, 43,008 tok) | OK，剩 1539 MiB | OK（util 0.86/0.80），剩 979 MiB |
| 176 blocks (4928 MiB, 45,056 tok) | — | OK（util 0.86），剩 755 MiB |

顺带确认：**给了 `--kv-cache-memory` 之后 `gpu-memory-utilization` 不再决定 KV 大小**（168 blocks 在 util 0.80
和 0.86 下都是 43,008 token），它只用于启动时的 free-memory 前置检查。

我选了 **168 blocks = 43,008 tokens**（比原来的 34,048 高 26%），自以为是在"用满显卡"。

**4）结果：vLLM 崩了。** c=32 时 vLLM 的 TTFT 从（上一轮的）682 ms 变成 **1839.8 / 2012.8 ms**，
吞吐从 1354.8 掉到 **996.3 tok/s**。但——**TPOT 反而正常（17.82 ms，比我们的 20.99 还好）**。

这个签名很干净：**只有 prefill 崩、decode 正常** ⇒ 是 prefill 要的大块**激活缓冲**被 KV 挤掉了。
旁证：GPU sampler 记录的峰值显存 **7859/8188 MiB，只剩 329 MiB**。

**5）验证假设**（同 vLLM、只改 KV）：

| vLLM KV | TTFT (c=32) | TPOT | tok/s | 峰值显存 |
|---|---|---|---|---|
| 168 blocks | **1839.8 ms** | 17.82 | **996.3** | 7859 MiB |
| **144 blocks** | **611.3 ms** | 17.33 | **1452.3** | 7430 MiB |
| 136 blocks | 644.9 ms | 17.09 | 1450.9 | 7408 MiB |

**TTFT 提升 3 倍，TPOT 几乎不动** —— 假设完全证实。

**结论**：vLLM 用 profiling **主动给激活留量**，nano-vllm 没有这个机制（它的公式用 load 时的 peak，
不含 serving 期峰值）。所以**强行字节对齐时必须停在两边都吃得下的值**。⇒ 定案 **144 blocks = 36,864 token**。

### 收获
**"用满资源"和"公平对比"在这里是冲突的**，而且冲突的方向出人意料：把显存顶满**惩罚了 vLLM**。
我原本以为顶满是"中性的极限测试"，实际上它变成了一个"对某个引擎不利的配置"。
教训：任何"把资源推到极限"的操作，都要先问"两个系统对这个预算的内部分配策略是不是一样"。

---

## 第 10 轮：定案重跑 —— 以及"为啥我们的在线比 vLLM 慢"

### 用户说什么
> 别跑了，现在进度咋样
> 不跑任何数据，纯在对话框里分析，你觉得为啥我们的在线比 vllm 的在线慢？

### 我做了什么

**1）停止并清点**：KV=144 的重跑跑到 14/19 个文件时被叫停（我们侧全齐；vLLM 缺在线 4 个 rep1 和 stock 对照点）。
残留进程清理干净，GPU 空闲。**踩了一个工具坑**：`pkill -f 'run_cmp3.sh'` 的模式匹配到了**它自己所在 bash 的
命令行**，把自己杀了（无输出、exit 1）。改用 `'run_cmp[3].sh'` 方括号写法规避。

**2）已拿到的数据（KV 精确对齐 36,864 token）**：

*离线（只比吞吐）*

| 引擎 | N | 生成 tok/s（rep0 / rep1） |
|---|---|---|
| nano-vllm | 32 | 1567.3 / 1565.9 |
| vLLM | 32 | 1503.4 / 1495.0 |
| nano-vllm | 128 | 1548.6 / 1545.3 |
| vLLM | 128 | 1554.7 / 1552.4 |

*在线*

| 引擎 | 并发 | TTFT mean | p50 | p99 | **TPOT** | 生成 tok/s |
|---|---|---|---|---|---|---|
| nano-vllm | 1 | 29.5 / 28.6 | 28.1 / 27.4 | 51.2 / 38.7 | 6.85 | 142.3 / 142.5 |
| vLLM | 1 | 34.1 | 31.0 | 52.4 | 7.30 | 133.1 |
| nano-vllm | 8 | 146.4 / 145.0 | 168.3 / 168.4 | 177.4 / 173.9 | 10.32 | 702.8 / 701.5 |
| vLLM | 8 | 159.9 | 171.7 | 190.4 | 10.05 | 712.7 |
| nano-vllm | 32 | 640.4 / 642.6 | 665.0 / 660.7 | 684.0 / 691.0 | **20.95** | 1238.2 / 1234.8 |
| vLLM | 32 | 643.1 | 670.4 | 689.6 | **17.22** | **1443.3** |
| nano-vllm | 48 | 665.0 / 652.4 | 698.0 / 704.0 | 998.4 / 988.5 | **25.11** | 1247.7 / 1245.4 |
| vLLM | 48 | 708.9 | 743.1 | 1055.7 | **20.40** | **1455.9** |

重复间离散度：我们 ±0.1%、vLLM ±0.3%（离线）；在线也很好。数据质量没问题。

**3）而"为什么"的答案，是把问题提法反过来**：

**TPOT 的比值随并发单调变化** —— 这是定位的关键：

| 并发 | 我们 TPOT | vLLM TPOT | 我们 / vLLM |
|---|---|---|---|
| 1 | 6.85 ms | 7.30 ms | **0.94（我们更快）** |
| 8 | 10.33 | 10.05 | 1.03 |
| 32 | 20.98 | 17.22 | **1.22** |
| 48 | 25.18 | 20.40 | **1.23** |

**如果是 GPU kernel 缺陷，这个比值应该是平的。** 比值从 0.94 单调爬到 1.23，
说明成本**随并发线性增长**，与 batch、与 attention/GEMM 无关。

**决定性对照**（同 batch 32、同形状，唯一差别是"有没有经过服务层"）：

| | 每 decode step | |
|---|---|---|
| 我们 离线 n=32 | **15.54 ms** | wall 2.613s − prefill 0.640s = 1.973s ÷ 127 步 |
| 我们 在线 c=32 | **20.95 ms** | TPOT |
| | **+34.8%** | ← 上网关的税 |
| vLLM 离线 n=32 | **16.39 ms** | wall 2.725s − prefill 0.643s = 2.082s ÷ 127 步 |
| vLLM 在线 c=32 | **17.22 ms** | TPOT |
| | **+5.1%** | ← 它的税 |

**读法**：**离线我们 15.54 ms/step，比 vLLM 的 16.39 还快 5%** ⇒ 我们的 decode 引擎实现没问题。
一过服务层，我们多交 **35%**，vLLM 只交 **5%** ⇒ **差距 100% 产生在服务层。**

**4）根因：架构差别，不是代码细节。**

| | 进程模型 |
|---|---|
| **vLLM** | `EngineCore` 在**独立进程**里跑模型循环（每次启动日志都有 `EngineCore pid=...`），API 前端是另一个进程 |
| **我们** | 引擎是 gRPC 服务进程里的**一个线程**（`EngineWorker._run`），和 `max_workers` 个 servicer 线程**抢同一个 GIL** |

所以 vLLM 的 HTTP/SSE 序列化**物理上碰不到引擎的 GIL**，我们的会。并发越高，servicer 线程越多，抢得越凶
—— 这正好解释了比值随并发单调上升。

**每步我们的引擎线程要和这些争 GIL**（都在读过的代码里）：
1. **每个 token 一次 protobuf 构造 + socket 写**：`grpc_server.Infer` 的流式循环，32 请求 = 每步 32 条消息。
2. **`EngineWorker._run` 每步 `set(self.llm.get_cached_prefix_hashes())`** —— 它是
   `list(self.hash_to_block_id.keys())`，**O(已缓存 block 数) 的 Python 拷贝，每步一次**（worklog §7 已点出，未修）。
3. `_publish_incremental()` 每步遍历所有 running 序列并持锁。
4. `postprocess → block_manager.hash_blocks()` 每步在 Python 里做 xxhash（链式哈希）。
5. `prepare_decode` 每步新建 5 个 pinned tensor + 若干 list。
6. `.tolist()`、3 个 Prometheus gauge 每步 `.set()`。

单看每项都小，但**引擎必须先把 Python 干完才能发下一个 kernel**。worklog 自己测到
"引擎忙占比 99.8%、host ≈ GPU、**不是空转，是 step 本身变慢**"，方向完全一致。

**5）这独立复现并锚定了 worklog 的 B4。**
worklog 说：负载下 step 从 6.35 → 8.5 ms（**+33%**），"怀疑 servicer 线程与引擎抢 GIL"，
状态 **⚠️ 未证实、未修**，修复建议 P5 就是"把引擎放独立进程"。
我在 **batch 32**（不是它当年的 batch 8）测到 **+34.8%**，并且拿到了它当年没有的**对照组**：
**vLLM 在完全相同条件下只交 5%**。

**6）一个必须说清的 caveat**：**c=48 对我们不公平，别用它下结论。**
我们 48 × 3 block = **144 block，恰好等于池子 144（100%）**，vLLM 只要 1920/2304 = 83%。
所以 c=48 我们贴着 KV 上限、可能有 preempt。**干净的比较点是 c=32**（我们 96/144 = 67%，
vLLM 1280/2304 = 56%，都不受限）—— 上面整段论证都用 c=32。
**这是 block_size=256 的碎片代价第二次显形。**

### 收获
**"谁快"之前先问"差在哪个量上、这个量随什么变化"。** 如果我只报"c≥32 vLLM 快 17%"，
结论会是"我们 kernel 不行"——完全错。是"离线 vs 在线"这个对照 + "TPOT 比值随并发"这个形状，
把答案锁死在服务层架构上。

---

## §3 错误账本（我先后更正过的 5 件事）

| # | 错误说法 | 实际 | 错因 | 何时更正 |
|---|---|---|---|---|
| 1 | 离线也报 TTFT（含 p50/p99） | **离线没有可观测 TTFT**；那个数是整个 batch 的 prefill 完成时间，一个标量 | 把"内部能抠出的数"当成"用户存在的量"；`p50==mean==p99` 已经提示退化了，我没读懂 | 第 4 轮，**用户指出** |
| 2 | 用 `llm.llm_engine.add_request()/step()` 给 vLLM 抠离线 TTFT | **不是官方用法**；改完预热后直接退化成 `TTFT == wall` | 为了让两边的表长得一样，绕过了公开 API | 第 5 轮，用户追问口径后删除 |
| 3 | 预热按"小 batch、短输出"跑就够了 | 我们离线 n=8 的 TTFT 被污染 **7 倍**（1169 vs 干净 167 ms） | 引擎自带 warmup 不覆盖真实 paged prefill 路径；预热必须**按测量形状** | 第 2 轮，靠 n=8 的反常数字查出来 |
| 4 | vLLM 默认 `max_num_batched_tokens=2048` 会拖垮它的 TTFT | **推翻**：2048 反而 TTFT 好 43%（388 vs 682 ms），吞吐只差 2% | 忽略了 chunked prefill 会让首批 token 提前出来 | 第 1 轮实测 |
| 5 | "尽可能用上显卡" = 把 KV 顶到物理极限（168 blocks） | **反而惩罚了 vLLM**：它的 prefill 慢 3 倍（TTFT 1839 ms、996 tok/s），decode 正常 | 没考虑两个引擎对"预算内 KV vs 激活"的分配策略不同；vLLM 靠 profiling 主动留激活余量 | 第 9 轮，验证 144 vs 168 后定案 |

另有两个**工具坑**（影响复现，不算结论错误）：
- `pkill -f 'run_cmp3.sh'` **匹配到自身命令行**把脚本自己杀了（无输出、exit 1）⇒ 改用 `'run_cmp[3].sh'`。
- PowerShell 内联引号会把 Python 代码转义搞坏（worklog 也踩过）⇒ **一律写成脚本文件再 `wsl.exe -- bash <file>`**。

---

## §4 关键数据总表（备查）

### 4.1 定案口径

| | 配置 | 用法（官方） | 报什么 |
|---|---|---|---|
| **离线** | KV 对齐 + 见 4.3 | vLLM `llm.generate()`；nano-vllm `LLM.generate()` | wall、生成吞吐、总吞吐（**无 TTFT**） |
| **在线** | 同上 | 官方 `vllm serve` + OpenAI API；nano-vllm 官方 `serve()` + gRPC | **TTFT**(mean/p50/p99)、**TPOT**、吞吐 |
| 规模 | 离线 N=32/128；在线 N=64，并发 1/8/32/48 | | |
| 重复 | 每点 2 次，报离散度 | | |
| 工作负载 | prompt 512 / output 128，`ignore_eos`，temp 1.0，**raw token id** | | |
| 数据 | 每条请求全随机 token、**每次测量换 seed** ⇒ 前缀复用不可能 | | 实测 **0 命中** |

### 4.2 KV 精确对齐（本次对比的核心配置）

取 **28 MiB（我们一个 block）的整数倍**，因为 `256 % 16 == 0`，字节相等 ⇒ token 容量精确相等：

| | 设置 | 结果 |
|---|---|---|
| nano-vllm | `num_kvcache_blocks=144` | 144 × 256 = **36,864 token** |
| vLLM | `--kv-cache-memory 4227858432` | 2304 × 16 = **36,864 token** ✅ 日志实测确认 |

### 4.3 定案配置对照（默认 vs 实际使用）

| 项 | nano-vllm 默认 | 实际 | vLLM 默认 | 实际 |
|---|---|---|---|---|
| `gpu_memory_utilization` | 0.9 | **0.80** | 0.92 | **0.80** |
| KV 池 | 由公式算（随桌面漂移） | **pin 144 blocks** | 由 profiling 算 | **`--kv-cache-memory` 4227858432** |
| `max_num_batched_tokens` | 16384 | 16384 | **2048** | **16384** |
| `max_num_seqs` | 512 | 512 | **128** | **512** |
| `max_model_len` | 4096 | 4096 | 模型自身 40960 | **4096** |
| block_size | 256（assert） | 256 | 16 | 16（**故意不动**，见第 7 轮） |
| CUDA graph | on | on | on | on |
| prefix caching | 内建 | 内建 | 默认 on | on |
| dtype / attn backend | bf16 / flash-attn | 同 | bf16 / 自选 FLASH_ATTN | 同 |

### 4.4 KV 池容量与碎片

| | block_size | block 数（定案） | 容量 (token) | 池字节 |
|---|---|---|---|---|
| nano-vllm | 256 | 144 | 36,864 | 3.94 GiB |
| vLLM | 16 | 2304 | 36,864 | 3.94 GiB |

**容量已精确对齐，但碎片没对齐**（640-token 序列）：

| | 每条占用 | 有效并发上限 |
|---|---|---|
| nano-vllm | 3 × 256 = **768 槽**（浪费 20%） | 144 ÷ 3 = **48** |
| vLLM | 40 × 16 = **640 槽**（0 浪费） | 2304 ÷ 40 = **57.6** |

⇒ **这正是 c=48 对我们不公平的原因**（我们 100% 占用，vLLM 83%）。

### 4.5 vLLM KV 顶满的代价（第 9 轮，c=32 实测）

| vLLM KV | TTFT | TPOT | 生成 tok/s | 峰值显存 |
|---|---|---|---|---|
| 168 blocks (43,008 tok) | **1839.8 ms** | 17.82 | **996.3** | 7859 MiB |
| **144 blocks (36,864 tok)** | **611.3 ms** | 17.33 | **1452.3** | 7430 MiB |
| 136 blocks (34,816 tok) | 644.9 ms | 17.09 | 1450.9 | 7408 MiB |

### 4.6 冷启动（已定案不进表，仅存档）

| 第几次真实 prefill | 耗时 |
|---|---|
| 第 1 次 | **2627.4 ms** |
| 第 2 次 | **1167.8 ms** |
| 第 3 次起 | **167 ms**（稳态） |

根因：`warmup_model()` 跑在 `allocate_kv_cache()` 之前 ⇒ 预热时 `k_cache.numel()==0` ⇒ `store_kvcache` 被跳过。

### 4.7 服务层税（第 10 轮的核心论证）

| | 离线 ms/step | 在线 ms/step | 税 |
|---|---|---|---|
| nano-vllm（batch 32） | **15.54** | **20.95** | **+34.8%** |
| vLLM（batch 32） | **16.39** | **17.22** | **+5.1%** |

### 4.8 硬件与 roofline 常数（沿用 worklog）

| 量 | 值 |
|---|---|
| 每 token KV（每序列） | 112 KiB |
| 实测读带宽 | 238 GB/s（`torch.sum`）；ncu 反推理论峰值 263.7 GB/s |
| 本对比采用的带宽口径 | ~250 GB/s |
| 机器平衡点 | 122–123 FLOP/byte |
| GPU 利用率（本次全程采样） | mean **72.5%**，max 100%；显存 mean 6401 / max 7859 MiB |

---

## §5 代码现状（接手时看这里）

### 新增文件（全部在 `benchmarks/`）

| 文件 | 作用 |
|---|---|
| `cmp_common.py` | 数据集构造（等长 + 前缀不可能相同 + 按 `(engine,mode,conc,rep)` 定 seed）、统计汇总、结果落盘 |
| `cmp_offline_nanovllm.py` | 离线吞吐，**官方 `LLM.generate()`**，支持多 N + 重复 |
| `cmp_offline_vllm.py` | 离线吞吐，**官方 `llm.generate()`**，支持多 N + 重复 |
| `cmp_online_client.py` | 在线客户端（gRPC / OpenAI SSE 双引擎），报 **TTFT / TPOT / 吞吐**，带 `--rep` |
| `cmp_serve_nanovllm.py` | 用官方 `serve()` 入口启动，透传 `num_kvcache_blocks` 等 Config 字段 |
| `cmp_report.py` | 结果聚合出表 |

### 修改文件

| 文件 | 改动 |
|---|---|
| `nanovllm/engine/model_runner.py` | **2 行**：`allocate_kv_cache` 尊重正数 `num_kvcache_blocks` 覆盖（补完一个已存在但未生效的字段），使 KV 池可复现、可与 vLLM 字节对齐 |

**除此之外，两个引擎的源码一行未改。** vLLM 是 pip 装的原样 0.31.0。

### `experiments/` 下的复现脚本（关键几个）

`sync_to_wsl.sh`、`probe_kv_size.py` / `probe_kv_max.sh`（探我们 KV 上限）、
`probe_vllm_kv_max.sh` / `probe_vllm_kv_push.sh`（探 vLLM KV 上限）、
`verify_kv_pressure.sh`（验证 168 vs 144 的显存压力假设）、
`run_cmp3.sh`（定案压测主脚本：KV 对齐 + 重复 + GPU 采样）、
`probe_kv_size.py`、`diag_n8.py`（冷启动诊断）、`stop_and_report.sh`。

### 测试/回归状态
`nanovllm/engine/model_runner.py` 的 2 行改动是**条件分支**（默认 `-1` 走原路径），
不影响既有行为；但**尚未跑过 `python -m unittest discover -s tests`**。⚠️ 这是接手第一件该做的事。

---

## §6 未决问题与起点

| # | 事项 | 说明 |
|---|---|---|
| **A** | 补完 vLLM 在线 **rep1 × 4** 和 **stock-2048 对照点** | 停止时缺这 5 项，约 7 分钟 |
| **B** | **跑一遍全仓测试** | 确认 `model_runner.py` 那 2 行没有破坏回归（60/60） |
| **C** | **正式证实服务层税**：用 `benchmarks/profile_serving_step.py` 在真实负载下记录每步 host_ms / gpu_ms / 真实 batch。预期：gpu_ms 稳在 ~15.5 ms，host_ms 涨到 ~21 ms | §10 的论证目前是"离线 vs 在线对照 + 比值形状"，还差一条直接证据 |
| **D** | 写 `docs/cmp_vllm_vs_nanovllm.md`（正式报告） | 口径、方法、结果、KV/碎片披露、block-size 256 的发现、§10 的架构结论 |
| **E** | block_size 的三处独立硬编码（`gateway.py` / `redis_router.py`）与 `assert % 256 == 0` 的不一致 | 用户已决定**先写进报告，暂不改代码** |
| **F** | 验证 `flash_attn_varlen_func` 的 paged prefill 路径是否也真的需要 256 | 文档没写；这是能不能把 block 降到 64/128 的真正卡点 |
| **G** | c=48 的不公平需要在对齐口径下重做 | 需要先把碎片差异处理掉（对齐容量 or 降 block） |

### 明确**不要**做的
- ❌ 不要再把 KV 顶到物理极限去"用满显卡"（第 9 轮已证明它会单方面惩罚 vLLM）
- ❌ 不要对齐 block size（技术上可行，但方向偏袒我们，见第 7 轮）
- ❌ 不要在离线模式报任何 per-request 延迟指标（第 4 轮）

### 战略层面的表述（第 10 轮后）
```
离线（引擎本身）：我们 ≈ vLLM，甚至略快（batch 32：15.54 vs 16.39 ms/step）
在线低并发：      我们 ≈ 或略优（c=1 TTFT −14%、TPOT −6%）
在线高并发：      我们慢 17%，且**全部**来自服务层税（+34.8% vs +5.1%）
                  ⇒ 杠杆是**进程架构**（引擎独立进程），不是 kernel
```

---

## §7 环境与复现备忘

### 执行环境
```bash
# 代码在 Windows 仓库写，同步到 WSL 执行
bash experiments/shared/sync_to_wsl.sh                        # rsync，不带 --delete
wsl.exe -d Ubuntu -- bash /mnt/c/.../experiments/<script>.sh  # ← 一律用脚本文件
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ml
```

### 版本
```
vllm 0.31.0    torch 2.13.0+cu130    transformers 5.17.0
flash_attn 2.8.3.post1  (attn backend: FLASH_ATTN，vLLM 自选)
模型 /home/adam/models/Qwen3-0.6B    RTX 4060 Laptop 8188 MiB
```

### WSL / 沙箱注意事项
- **`wsl.exe` 走命名管道**，在受限沙箱下报 `WSL_ACCESSDENIED` / EPERM ⇒ 需要放宽到 full-access。
- **PowerShell 内联引号会把 Python 代码转义搞坏**（反复踩）⇒ 写脚本文件。
- **`pkill -f '<pattern>'` 会匹配到自身命令行** ⇒ 用 `'run_cmp[3].sh'` 这类方括号写法。
- **`/tmp` 在两次 `wsl.exe` 调用之间可能被清空** ⇒ 中间产物写仓库目录（worklog 已记）。
- **桌面程序共用 GPU**，桌面占用在 **400 MiB ~ 1.07 GiB** 之间波动；已实测到一次
  `vLLM 在线 c=1` 慢 3.5 倍（230.7s vs 正常 66.8s）⇒ 必须重复 + 用 GPU 采样交叉核对。

### vLLM 0.31.0 的几个反直觉点（下次直接查这里）
- `--disable-log-requests` **不存在**；现在是 `--no-enable-log-requests`（还有 `--disable-uvicorn-access-log`）。
- 官方离线 `llm.generate()` **不填** `RequestOutput.metrics` ⇒ **拿不到离线 TTFT**（也不该拿）。
- `--kv-cache-memory` **不能绕过**启动时的 `free >= util*total` 前置检查 ⇒ 必须同时给一个能过的
  `--gpu-memory-utilization`（本例 ≤0.86）；但给了 `--kv-cache-memory` 之后，util **不再决定 KV 大小**。
- `llm.generate()` 不收 `prompt_token_ids=` 关键字；token id 要按 `[{"prompt_token_ids": ids}, ...]` 传。
- 用 `vllm.LLM` 必须包在 `if __name__ == "__main__":` 里（V1 引擎核心是 spawn 子进程）。
- 它自己的默认值：`max_num_batched_tokens=2048`、`max_num_seqs=128`、`gpu_memory_utilization=0.92`、
  `block_size=16`、prefix caching on、chunked prefill on。

---

## §8 最有价值的四条方法论收获

1. **指标的可观测性要先于可测量性。** 离线没有 TTFT——不是"测不到"，是**它对用户不存在**。
   我当时从引擎内部硬抠出一个数，还给了 p99；而 `p50 == mean == p99` 这个退化成单点的"分布"
   早就在喊口径错了。**是用户一句话把它推翻的，不是我自己发现的。**

2. **用法必须是官方用法，否则你测的是你的 harness。** 那个魔改的离线 TTFT 不但错，还**脆弱**——
   一改预热协议就退化成 `TTFT == wall`，我差点去查一个根本不存在的 bug。

3. **永远先问"这个差异随哪个变量变化"。** 如果只报"c≥32 vLLM 快 17%"，结论会是"我们 kernel 不行"。
   是两件事把答案锁死的：**"离线 vs 在线"的对照**（离线我们更快），
   和 **"TPOT 比值随并发单调上升"的形状**（kernel 缺陷不会这样）。两者一起指向服务层架构。

4. **"把资源推满"不是中性的极限测试。** 我按用户要求把 KV 顶到 168 blocks（+26%），
   结果**单方面惩罚了 vLLM**（prefill 慢 3 倍，decode 正常）——
   因为两个引擎对"预算内怎么分配 KV 与激活"的策略不同。任何"推满资源"的操作，
   都要先问一句：**两个系统对这个预算的内部分配策略是不是一样的？**

> 附注：这次工作的结论和 worklog 的主线是**同一个教训的第二次出现**——
> worklog 里值 3.4× 的是服务层一个配置值（`max_workers=10`），而不是 GPU 上的任何东西；
> 这次差距 100% 来自服务层的**进程架构**（引擎与 servicer 共享 GIL），同样不是 GPU。
> **两次都把"性能问题"的归因从 kernel 拉回到了服务层。**
