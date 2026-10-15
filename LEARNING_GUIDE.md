# nano-vllm-plus 学习指南（六层递进路线）

> 配套讲解会话的持久化笔记。随教学进度更新。
> 项目本体约 1200 行 Python，是 vLLM 的迷你实现。
>
> **📌 讲解基线**：以**原始上游 nano-vllm**（commit `ab5450b^`，即用户添加 serving 层之前的
> 最后一个提交）为准。用户自加的 `nanovllm/serving/`（gRPC/Redis 路由）及对
> `block_manager.py`（哈希查询 API）、`llm_engine.py`（get_cached_prefix_hashes 等）、
> `scheduler.py`（num_waiting/abort 等）、`sequence.py`（ABORTED 状态）的增量修改
> **不纳入讲解**。layers/、models/、utils/、config.py 等其余文件两版本完全一致。

---

## 0. 进度追踪

| 层 | 内容 | 状态 |
|---|---|---|
| 第 0 层 | LLM 推理核心概念（token/logits/temperature/prefill-decode/KV Cache） | ✅ 已在对话中讲完 |
| 第 0 层配套 | 无 GPU 演示脚本 `practice/demo_layer0.py` | ✅ 已写好并验证可运行 |
| 第 1 层 | example.py + LLMEngine 全局流程 | 📖 讲义已备好 `lessons/layer1_llm_engine.md`，待对话讲解 |
| 第 2 层 | Sequence / SamplingParams / Config 数据结构 | 📖 讲义已备好 `lessons/layer2_data_structures.md` + 演示 `practice/demo_pickle.py`，待对话讲解 |
| 第 3 层 | Scheduler + BlockManager（分页 KV Cache / 前缀缓存 / 抢占） | 📖 讲义已备好 `lessons/layer3_scheduler.md` + 模拟器 `practice/demo_kvcache.py`，待对话讲解 |
| 第 4 层 | Qwen3 模型结构 + layers 算子 | 📖 讲义已备好 `lessons/layer4_model_layers.md` + 维度计算器 `practice/demo_shapes.py`，待对话讲解 |
| 第 5 层 | ModelRunner + 性能优化（CUDA Graph / torch.compile / flash-attn / Triton） | 📖 讲义已备好 `lessons/layer5_model_runner.md` + RPC 演示 `practice/demo_rpc.py`，待对话讲解（需 GPU 环境） |
| 第 6 层 | 张量并行 + 多进程通信 | 📖 讲义已备好 `lessons/layer6_tensor_parallel.md`，待对话讲解 |
| 环境 | WSL2 + CUDA + 依赖 + 模型下载 + 跑通 example.py | ⏳ 待用户操作（见第 3 节） |

---

## 1. 项目地图

```
LLM (nanovllm/llm.py)                    ← 用户入口，就一行：class LLM(LLMEngine): pass
 └─ LLMEngine (engine/llm_engine.py)     ← 总指挥：收请求、循环调度、返回结果
     ├─ Scheduler (engine/scheduler.py)         ← 调度器：谁该跑了？prefill 还是 decode？
     ├─ BlockManager (engine/block_manager.py)  ← 显存管家：KV 缓存按块分配/复用/回收
     ├─ ModelRunner (engine/model_runner.py)    ← 执行者：真正在 GPU 上跑模型
     │    └─ Qwen3ForCausalLM (models/qwen3.py) ← 模型本体（Qwen3 结构）
     │         └─ layers/                       ← 算子：attention / linear / rmsnorm / rope / sampler
     └─ utils/                                  ← 工具：权重加载 (loader.py)、全局上下文 (context.py)
```

**一次 `generate()` 的旅程**：
1. prompt 字符串 → `AutoTokenizer` → token id 列表（第 0 层的"token"概念）
2. 包成 `Sequence` 对象，进入 `Scheduler.waiting` 队列
3. `Scheduler.step()` 循环：选一批序列 → `ModelRunner.run()` 前向 → 采样出新 token → 追加回去
4. 直到所有序列 FINISHED → tokenizer 解码回文本

**两个阶段**（第 0 层讲过）：
- **prefill**：整个 prompt 一次算完，产出第一个新 token，KV 写入缓存
- **decode**：之后每轮每序列只喂 1 个新 token，靠 KV Cache 记住历史

---

## 2. 第 0 层笔记（速查）

### 概念 1：Token
- 模型不认识文本，只认识整数 id。Tokenizer 把文本切成 token 碎片并映射成 id。
- 对应代码：`example.py` 的 `AutoTokenizer`；`llm_engine.py` 的 `self.tokenizer.encode(prompt)`。

### 概念 2：Logits
- 模型最后一层输出长度 = 词表大小（Qwen3-0.6B ≈ 151,936）的向量，每个位置是该 token 的"原始分数"。
- 对应代码：`models/qwen3.py` 的 `compute_logits`（lm_head）。

### 概念 3：Temperature 采样
- 三步：logits ÷ temperature → softmax → 按概率随机抽一个。
- 对应代码：`layers/sampler.py` 三行（第三行是 **Gumbel-max 技巧**：概率除以 Exp(1) 噪声再 argmax，等价于按概率采样，但全张量运算便于编译优化）。
- 效果：温度越小越机械（分布尖），越大越随机（分布平）。`SamplingParams(temperature=0.6)`。

### 概念 4：自回归 + Prefill/Decode + KV Cache
- 自回归：第 N+1 个 token 依赖前 N 个，必须逐个生成（串行瓶颈）。
- Prefill 一次并行算完 prompt；Decode 每轮只生成 1 个 token。
- KV Cache：把历史 token 的 K/V 向量缓存起来避免重算；**是推理显存最大头**，所以 vLLM 才精心管理它。

### 第 0 层演示脚本
```bash
python practice/demo_layer0.py
```
纯标准库实现，演示：词表 / softmax / 温度对分布的影响 / 朴素采样 vs Gumbel 采样（2 万次对比验证等价性）/ 玩具自回归生成。**不需要 GPU，现在就能跑。**

---

## 3. 环境搭建（WSL2 + CUDA 完整跑路线）

> ⚠️ 已检测到本机 WSL 环境：WSL 2.7.12 + Ubuntu 26.04 已安装并注册（WSL2、用户 uid=1000）。
> 但**自动化 shell 处于受限沙箱，无法访问 WSL 服务**（`wsl -l -v` / `ubuntu.exe` 均报
> E_ACCESSDENIED / 0x80070005）——这需要你在**自己的终端**里操作，然后把输出贴回来。
> 另外 Windows 上的 Python 3.14 不适合装 ML 库，请在 WSL 里用 Python 3.11。

### 3.1 Windows 侧（管理员 PowerShell）
```powershell
wsl --install -d Ubuntu-22.04   # 装 WSL2 + Ubuntu（可能要求重启）
nvidia-smi                      # 重启后确认显卡驱动正常
```
- NVIDIA 显卡驱动装在 Windows 侧即可，WSL 里通过 CUDA passthrough 直接用。
- 装完重启后 `wsl` 进入 Ubuntu。

### 3.2 WSL 内部
```bash
# Python 3.11（flash-attn 对 3.10~3.12 支持最好，别用 3.13+）
sudo apt update && sudo apt install -y python3.11 python3.11-venv build-essential
python3.11 -m venv ~/venv && source ~/venv/bin/activate

# PyTorch（CUDA 版；cu 版本按 https://pytorch.org/get-started/locally/ 选，一般 cu124/cu126）
pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # 验证

# 其他依赖
pip install transformers safetensors xxhash tqdm

# flash-attn（最容易卡的一步：会现场编译，需要 nvcc）
#   方案 A：直接装（若报错缺 nvcc，先装 CUDA toolkit，版本要匹配 torch.version.cuda）
pip install flash-attn
#   方案 B：找预编译 wheel（搜索 "flash-attention prebuilt wheels torch 2.x cu12x"）
# 验证：
python -c "import flash_attn; print(flash_attn.__version__)"
```

### 3.3 下载模型 + 跑通
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ --local-dir-use-symlinks False

# 模型下载好后，先打印官方配置验证结构数字（第 4 层要用）：
python -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('~/huggingface/Qwen3-0.6B/')
print('vocab:', c.vocab_size, '| layers:', c.num_hidden_layers,
      '| heads:', c.num_attention_heads, '| kv_heads:', c.num_key_value_heads,
      '| hidden:', c.hidden_size, '| max_pos:', c.max_position_embeddings)
"
# 预期（Qwen3-0.6B 公开配置）：vocab 151936, layers 28, heads 16, kv_heads 4,
# hidden 1024, max_pos 32768

# 跑示例
cd /mnt/c/Users/Adam/PycharmProjects/nano-vllm-plus   # 或把仓库 git clone 进 WSL 更快
python example.py
```

### 3.4 常见坑
| 坑 | 解决 |
|---|---|
| flash-attn 编译报错缺 nvcc | 装与 `torch.version.cuda` 匹配的 CUDA toolkit（含 nvcc），或用预编译 wheel |
| 从 /mnt/c 跑代码慢 | 文件 IO 慢是 WSL 挂载盘的通病；把仓库复制/克隆到 WSL 家目录 |
| 显存不够（bench.py 256 条 × 最长 1024 token） | README 用的是 8GB 卡；显存更小就调小 `bench.py` 里的 `num_seqs` / 长度上限 |
| `tcp://localhost:2333` 端口占用 | `model_runner.py` 里 NCCL 初始化用的固定地址端口，被占用时改端口 |
| 千万别在 Windows 原生 Python 里装本项目 | NCCL / flash-attn / CUDA Graph 在 Windows 上不可用 |

---

## 4. "普通开发不常用"方法速查表（讲解时会逐个展开）

| 类别 | 具体内容 | 出处 |
|---|---|---|
| Python 语法 | `dataclass(slots=True)`、`__post_init__`、`dataclasses.fields()` | config.py / sampling_params.py |
| | `Enum` + `auto()` | sequence.py |
| | `@property`、`@classmethod` | sequence.py / block_manager.py |
| | `str \| list[int]` 类型联合、`for...else` / `while...else` | llm_engine.py / scheduler.py |
| 标准库 | `itertools.count`（无限计数器） | sequence.py |
| | `collections.deque`（`popleft`/`appendleft`/`extendleft`） | scheduler.py / block_manager.py |
| | `functools.lru_cache` | rotary_embedding.py |
| | `atexit.register`（进程退出钩子） | llm_engine.py |
| | **pickle 协议钩子** `__getstate__`/`__setstate__`（跨进程传对象时自定义序列化） | sequence.py |
| | `multiprocessing`：`get_context("spawn")`、`Event`、`SharedMemory` | llm_engine.py / model_runner.py |
| | `glob`、`perf_counter`、`int.from_bytes` | loader.py / llm_engine.py / model_runner.py |
| PyTorch | `nn.Parameter`、`register_buffer(persistent=False)`、`nn.ModuleList` | layers/* |
| | `narrow`（视图切片不拷贝）、`chunk`、`einsum` | linear.py / rope / qwen3.py |
| | `pin_memory=True` + `.cuda(non_blocking=True)`（异步拷贝） | model_runner.py |
| | `torch.compile`、`torch.inference_mode` | layers / model_runner.py |
| | **CUDA Graph**：`torch.cuda.CUDAGraph`、`graph.pool()`、`graph.replay()` | model_runner.py |
| | `set_default_dtype/set_default_device`、`mem_get_info`、`memory_stats()["allocated_bytes.all.peak"]` | model_runner.py |
| | 原地算子 `mul_/add_/div_`、`rsqrt`、`clamp_min_` | layernorm.py / sampler.py |
| 分布式 | `dist.init_process_group("nccl", ...)`、`all_reduce`、`gather`、`barrier`、`destroy_process_group`、`set_device`、`synchronize` | model_runner.py / linear.py / embed_head.py |
| 第三方 | `xxhash.xxh64().update().intdigest()`、`numpy.tobytes()` | block_manager.py |
| | `safetensors.safe_open` | loader.py |
| | `triton.jit`（`tl.program_id/load/store/constexpr`） | attention.py |
| | `flash_attn_varlen_func` / `flash_attn_with_kvcache` | attention.py |
| | `tqdm.set_postfix`（进度条附加指标）、`apply_chat_template` | llm_engine.py / example.py |

---

## 5. 练习题集（随层数增加）

- [x] 第 0 层：`python practice/demo_layer0.py` —— 观察温度对生成的影响（改 `temperature` 再跑）
- [ ] 第 0 层附加：把 `demo_layer0.py` 的玩具模型换成"只输出你好世界"的确定性模型，体会 temperature→0 的极限
- [x] 第 2 层：`python practice/demo_pickle.py` —— 观察 prefill/decode 两阶段 pickle 体积差（省 79% 通信量）与 __setstate__ 还原结果
- [ ] 第 2 层附加：给 demo_pickle.py 的 Sequence 加个不放进出参的属性，序列化后访问它看 AttributeError
- [x] 第 3 层前置：`python practice/demo_kvcache.py` —— 观察块分配/引用计数/前缀缓存命中/抢占全过程
- [ ] 第 3 层附加：把 `demo_kvcache.py` 的 `NUM_BLOCKS` 改成 6 或 9，观察抢占时机和频率如何变化
- [x] 第 4 层前置：`python practice/demo_shapes.py` —— Qwen3-0.6B 全模型张量形状推演 + 参数量 + KV 块显存公式
- [ ] 第 4 层附加：改 `demo_shapes.py` 的 `SEQ_LENS` 为 `[100, 200, 300]` 重跑，体会批次维度变化
- [x] 第 6 层前置：`python practice/demo_rpc.py` —— 主从进程共享内存 RPC 协议（write_shm/read_shm/loop）
- [ ] 第 6 层附加：把 `demo_rpc.py` 的 worker 数量改成 2，观察 write_shm 广播
- [ ] 第 1 层：把 `max_num_batched_tokens` 改小，观察吞吐量变化
- [ ] 第 2 层：写脚本创建几个 Sequence，手动模拟"追加 token → 块满 → 换新块"
- [ ] 第 3 层：构造两个共享前缀的 prompt，打印日志看缓存命中
- [ ] 第 4 层：`print(model)` 打印模型结构，手动算一遍各层维度（对照 demo_shapes）
- [ ] 第 5 层：`enforce_eager=True` vs `False` 各跑一遍，对比速度（需 GPU 环境）
- [ ] 第 6 层：`tensor_parallel_size=2` 跑一遍（需双卡或 ≥2 卡）
- [ ] 毕业练习：不看代码画出 `generate()` 完整流程图，标注每环节对应的层

---

## 6. 讲义文档（对话讲解的持久化版本）

- [x] 第 1 层：`lessons/layer1_llm_engine.md` —— example.py + LLMEngine 逐段精读，含数据流图、5 道自测题（附答案）
- [x] 第 2 层：`lessons/layer2_data_structures.md` —— SamplingParams/Config/Sequence 逐段精读（dataclass slots / Enum / count / property / pickle 协议钩子），5 道自测题（附答案）
- [x] 第 3 层：`lessons/layer3_scheduler.md` —— Scheduler/BlockManager 逐段精读（deque / while...else / 抢占 / chunked prefill / 哈希前缀链），5 道自测题（附答案）
- [x] 第 4 层：`lessons/layer4_model_layers.md` —— Qwen3 模型 + 全部 8 个 layers 算子逐段精读（GQA / QK-Norm / RoPE / 并行线性层 / Triton / gather），7 道自测题（附答案）
- [x] 第 5 层：`lessons/layer5_model_runner.md` —— ModelRunner 逐段精读（warmup / KV 分配 / prepare / CUDA Graph / 静态缓冲 / 全局 context），6 道自测题（附答案）
- [x] 第 6 层：`lessons/layer6_tensor_parallel.md` —— TP 架构 / NCCL / 共享内存 RPC / gather 合并 / 退出协议，7 道自测题（附答案）

---

## 7. 十分钟总览（学完六层后的速查）

**一句话**: nano-vllm 是一个迷你 vLLM —— 用"连续批处理 + 分页 KV 缓存"把 LLM 推理做到又省显存又高吞吐。

**一次 generate() 的旅程**（六层串联）:
```
prompt 字符串
  → [第0层] tokenizer 切词 → token id
  → [第2层] 包成 Sequence(状态机 WAITING→RUNNING→FINISHED, seq_id 全局唯一)
  → [第3层] Scheduler 选人: prefill 整段算 / decode 每轮 1 token; 显存不够就抢占
  → [第5层] ModelRunner 拼 GPU 张量(pin_memory+non_blocking) → 前向 → 采样
  → [第4层] Qwen3: embedding → 28×(RMSNorm+Attention(GQA+QK-Norm+RoPE)+SwiGLU) → LMHead
  → [第6层] TP>1 时: 权重切到多卡, 共享内存广播任务, all_reduce 合并, rank0 采样
  → 回到 [第3层] postprocess: 追加 token、判 EOS → FINISHED
  → [第1层] generate() 循环直到全部完成 → tokenizer 解码回文本
```

**六层的"一句话"**:
| 层 | 一句话 |
|---|---|
| 0 | token/logits/temperature/prefill-decode/KV Cache 是推理的地基 |
| 1 | LLMEngine 是总指挥: 收请求、循环 step、返回结果 |
| 2 | Sequence 是请求的"病历本": 状态、计数、块表; pickle 瘦身只传需要的 |
| 3 | Scheduler+BlockManager 是灵魂: 分页、前缀缓存、抢占、chunked prefill |
| 4 | Qwen3+layers 是身体: GQA、QK-Norm、RoPE、SwiGLU、并行线性层 |
| 5 | ModelRunner 是引擎: warmup、KV 分配、CUDA Graph、静态缓冲 |
| 6 | TP+多进程是协作: spawn、NCCL、共享内存 RPC、all_reduce/gather |

**速查数字** (Qwen3-0.6B): 0.737B 参数 | bf16 权重 1.47GB | KV 块 7.0MB/256token | 8GB 卡约 877 块 | vocab 151936 | 28 层 | GQA 4:1

---

*更新：目标 goal-8fe08d4a。六层讲义 + 5 个纯 Python 演示已全部备好并验证。剩余工作：用户在 WSL2+CUDA 环境实操（装依赖、下载模型、跑通 example.py/bench.py）+ 互动答疑。*
