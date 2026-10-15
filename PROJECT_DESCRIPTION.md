# 项目描述：nano-vllm-plus

> 本文把仓库里散落的 PRD、三份 worklog、六层讲义、benchmarks 与测试汇总成一份可直接用于
> README / 简历 / 面试自述的项目描述。所有数字都来自本仓库已落盘的实测记录，不采用计划值。
>
> **原创边界（重要，面试会被问）**：引擎本体来自上游 `nano-vllm`（约 1200 行，MIT）；
> 本仓库的增量全部由作者完成——`nanovllm/serving/`（服务层）、`nanovllm/layers/decode_attention.py`
> （Triton 算子）、`tests/`、`benchmarks/`、`experiments/`、`docs/`、`lessons/`、`practice/`。
> 仓库 fork 自 `ADAM00o00/nano-vllm-plus`。

---

## 1. 一句话

**在 1200 行的教学级 vLLM 复刻品（nano-vllm）之上，自己补齐一个能扛压测的在线推理服务层，
然后用 nsys / ncu 把"性能瓶颈到底在哪"这件事从猜测做成证据——最后发现最大的两个瓶颈
都不在 GPU 上（一个是 gRPC 线程池默认值，一个是进程架构）。**

三条主线，一条比一条更"反直觉"：

1. **服务化**：把阻塞式 `generate()` 循环改造成后台 worker，接上 gRPC 流式、Redis 前缀路由、
   双重背压、级联取消、Prometheus 服务层 → 从"只能离线跑的 1200 行 demo"变成"能演示、
   能压测、能观测的推理服务"。
2. **性能归因**：为 RTX 4060（sm_89）手写 Triton paged decode attention 算子，**做成了、正确、
   贴着显存带宽墙，但收益 1.00x**——然后顺着"为什么没收益"查出真正的瓶颈：
   `max_workers=10` 一个配置值值 3.4×。
3. **横向对标**：用同一块卡、同一份数据、同一套口径把自研引擎和 **vLLM 0.31.0** 放一起压，
   结论是"**引擎本身比 vLLM 略快，慢的是我们自己套在它外面的服务层**"，并定位到根因是
   **引擎与 gRPC servicer 共享 GIL**（vLLM 的 EngineCore 是独立进程）。

---

## 2. 项目定位与来龙去脉

| 项 | 内容 |
|---|---|
| 性质 | 个人项目 / 单人全栈（引擎改造 + 服务层 + 性能工程 + 全部文档） |
| 起点 | 上游 nano-vllm（教学向 vLLM 复刻：连续批处理 + 分页 KV cache + 前缀缓存 + 张量并行，约 1200 行） |
| 目标 | 不走"再写一个玩具"的路，而是补齐**真实服务化**能力，并把**性能问题归因**做成一门可复用的方法论 |
| 硬件 | RTX 4060 Laptop（sm_89，24 SM，8 GB，实测 bf16 28.9 TFLOP/s、读带宽 238 GB/s、ncu 反推理论峰值 263.7 GB/s） |
| 模型 | Qwen3-0.6B（28 层、hidden 1024、16 Q heads / 8 KV heads、head_dim 128、vocab 151936、bf16 权重约 1.19 GB） |
| 运行环境 | Windows 侧写代码 / WSL2 Ubuntu 26.04 + conda `ml` 执行（torch 2.13.0+cu130、triton 3.7.1、flash-attn 2.8.3.post1） |
| 对比对象 | 原版 pip 安装的 **vLLM 0.31.0**（除我们自己的 2 行 KV 覆盖外，两边源码均未魔改） |
| 规模 | 引擎本体约 1200 行；自研 serving 层 1374 行；测试 835 行 / 37 项；压测与剖析脚本 2002 行 / 12 个；实验脚本 70 个；文档与讲义 3050 行 |
| 产出 | 可一键演示的 GPU-free mock 全链路、可复现的压测工具链、三份逐轮 worklog + 实验脚本归档、六层源码讲义 + 5 个纯 Python 演示脚本 |

**为什么值得做**：nano-vllm 这类教学实现通常止步于"能生成文本"。真正区分工程能力的是三件事——
并发与流式、过载下的稳定性、以及**瓶颈到底在哪**。这个项目把这三件事都做了一遍，并把过程与错误
一并留档。

---

## 3. 仓库地图

```
nanovllm/
├── engine/          # 上游引擎：LLMEngine / Scheduler / BlockManager / ModelRunner
│   └──             #   分页 KV cache、前缀缓存、抢占、chunked prefill、连续批处理
├── layers/          # 上游算子 + 【自研】decode_attention.py（Triton paged decode attention）
├── models/          # Qwen3 结构
├── serving/         # 【自研】服务层（1374 行）
│   ├── engine_worker.py    后台单线程 step 循环 + 命令队列 + 请求句柄（把非线程安全的引擎收敛到单线程）
│   ├── grpc_server.py      gRPC 流式 Infer / Abort / Health + 级联取消
│   ├── gateway.py          多实例网关（流式调用占满整条流的线程，与 worker 同一个坑）
│   ├── redis_router.py     Redis 前缀缓存目录 / 实例心跳 / 负载退化路由
│   ├── admission.py        QPS 令牌桶 + KV 显存水位滞回（双背压）
│   ├── metrics.py          Prometheus 指标
│   └── mock_engine.py      GPU-free mock 引擎（让全链路在没有显卡时也能演示/测试）
tests/               # 37 项：serving 核心 8 + decode attention 29
benchmarks/          # 12 个脚本：gRPC 压测、step 剖析、kernel 对比、vLLM 横向对标
docs/                # PRD + 三份 worklog（sm89 / 瓶颈分析 / vLLM 对标）+ Phase 1 报告
lessons/             # 六层源码讲义（逐段精读 + 自测题）
practice/            # 5 个无 GPU 演示脚本（采样 / pickle / KV cache / 形状 / RPC）
experiments/         # 73 个实验脚本（shared / sm89 / serving / vllm 分组）+ 6 份原始日志（可复现配方）
```

---

## 4. 四条主线：做了什么、怎么做的

### 主线 A｜服务化：把阻塞循环变成可并发、可取消、可观测的服务

**难点**：`LLMEngine.generate()` 是阻塞的 `while not is_finished(): step()`；而
Scheduler / BlockManager / Sequence **全部非线程安全**。直接多线程调用必然竞态。

**做法**：

| 机制 | 实现 |
|---|---|
| 并发 | `EngineWorker` 持有引擎，在**单一后台线程**里跑 step 循环；外部线程只通过 `queue.Queue` 投递 `add` / `abort` / `stop` 命令，引擎内部状态只被这一个线程触碰 |
| 流式 | 每个请求一个 `RequestHandle`（输出队列 + token 游标），每步用游标算增量 token 推给 gRPC 流 |
| 取消 | 客户端断开 → gRPC `context.is_active()` 变 False → 提交 `abort` 命令 → `Scheduler.abort` → `BlockManager.deallocate` 回收 KV 显存；取消幂等，abort 延迟 = 一个 step |
| 背压 1（QPS） | 令牌桶（`max_qps` / `burst`），超限返回 `RESOURCE_EXHAUSTED` |
| 背压 2（显存） | KV cache 水位高/低水位**滞回**（默认 0.90 / 0.70），避免在阈值附近抖动；水位由 worker 每步更新 |
| 路由 | Redis 前缀目录：请求前缀 hash → 实例 → 未命中/失联则退化为"负载最轻实例"；目录定位为"尽力而为"，靠 TTL + 心跳兜底 |
| 可观测 | Prometheus：请求数、在途、队列深度、KV 水位、前缀命中/未命中、背压拒绝（按原因 label）、abort 数、回收耗时直方图 |
| 可测试性 | mock 引擎让整条链路**不需要 GPU/模型**就能跑：`python -m nanovllm.serving --mock` 一条命令起演示 |

**这一段的工程价值**：把"引擎非线程安全"这个约束翻译成了明确的所有权设计（单线程 + 命令队列），
而不是靠加锁硬扛——这也是后来 `abort` 语义能保持幂等、取消路径能干净收敛的原因。

### 主线 B｜性能归因：手写 Triton 算子，然后诚实地报告它没用

按计划为 sm_89 写 paged decode attention（online softmax + GQA 映射 + block_table 分页寻址 +
split-KV 路径 + shape-only 自动调参），29 项测试覆盖位置集合、末 token、分页寻址、GQA 映射、
NaN 尾块、padding 等**精确结构探针**。

结果：**正确（最差误差 1.95e-3，判据 1e-2），性能 1.00x（0.92–1.14x），显存不涨，端到端无退化。**
再用 ncu 拿到硬证据：自研 kernel `dram__throughput` 88.3% vs flash-attn 95.1%，
**`dram__bytes_read` 两者完全相同（67.35 vs 67.38 MB，且等于逻辑最小值）**，耗时只差 3%
（292.6 vs 269.5 µs）⇒ 该形状下的全部剩余天花板约 13 µs / 4.9%。
三条"看起来能赚"的路被实测证伪：split-KV（combine 固定开销 8–10 µs，净亏，默认关闭）、
num_stages 调优（1–4 无差异）、GQA head 配对（DRAM 字节完全相同，省不到带宽）。

**真正的收益来自排查过程本身：**

| 阶段 | 发现 | 效果 |
|---|---|---|
| 步效率直测 | 用 CUDA events 直接量 step：实测 73–89%（此前从 gRPC p50 反推得到的"50%"是错的） | 把 28 层/步的 attention 占比从"<3%"更正为 ctx11 4.4% / ctx2048 68% / ctx4096 85%（**此前错了 28 倍**） |
| 引擎 batch 直方图 | 32 个请求在飞，引擎只看到 batch 10 —— 根因是 `grpc.server(ThreadPoolExecutor(max_workers=10))`：**流式调用会占满一个工作线程整个流的生命周期，所以 max_workers 不是线程池大小，而是并发请求数的硬上限，也就是引擎 batch 的上限** | 改 3 处默认值 → 同并发 64 从 857 → 2930 tok/s（**3.4×**），p99 从 2366 → 691 ms |
| 继续推并发 | c128 达 3803 tok/s；c256 撞上"KV 池约 162 并发序列"的物理硬顶（背压 `memory` 生效，不 OOM） | 累计 ~4.4× |
| 逐 kernel 分解（nsys） | 高 batch 下 GPU 只跑到 25–33% 带宽、50% SM ⇒ **延迟/occupancy/host 开销主导** | 推翻了"优先做权重量化"的优先级判断 |

### 主线 C｜横向对标 vLLM：同一块卡、同一份数据、同一套口径

**方法学部分比数字更值钱**：等长数据（prompt 512 / output 128 / `ignore_eos`）、每条请求
全随机 token id 从第 0 个 token 就不相同、**每次测量换 seed**（因为发现
`BlockManager.hash_to_block_id` **从不清空**，重跑同一批 prompt 会静默变成前缀缓存命中）、
两边 KV 池**按字节精确对齐**（144×256 = 2304×16 = 36,864 token）。

**结论**：

| 场景 | 结果 |
|---|---|
| 离线吞吐（引擎本身） | 基本打平，batch 32 时我们 **15.54 ms/step vs vLLM 16.39**（我们还略快） |
| 在线 TTFT | 打平或略优（c=1 −14%、c=8 −9%、c=48 −7%） |
| 在线高并发吞吐 | c≥32 慢 17%，且 **100% 由服务层解释**：同 batch 32，离线→在线我们 +34.8%，vLLM 只 +5.1% |
| 根因 | 我们把引擎和 gRPC servicer 放在**同一个进程共享 GIL**；vLLM 的 EngineCore 是独立进程 |

**这个结论的价值**：如果只看"c≥32 vLLM 快 17%"，很容易误判成 kernel 不行，从而去优化 GPU。
是两件事把答案锁死的——**离线/在线的对照**（离线我们更快），以及**TPOT 比值随并发单调上升的形状**
（kernel 缺陷不会长成这样）。两次工作（B 和 C）都把性能归因从 kernel 拉回到了服务层。

### 主线 D｜可复现性与知识沉淀

- **37 项测试**（serving 核心 8 + decode attention 29），全仓可在无 GPU 环境跑大部分；
- **GPU-free mock 全链路**：mock gRPC 引擎一条命令演示流式、背压拒绝、级联取消三种行为；
- **12 个压测/剖析脚本**：gRPC 压测、驱动 `LLMEngine.step` 的步级剖析（host/gpu 时长 + 真实 batch
  直方图 + SIGTERM 落盘）、kernel 扫描（含 L2 flush 与实测带宽标定）、vLLM 对标客户端（TTFT/TPOT）；
- **三份逐轮 worklog**（sm89 / 瓶颈分析 / vLLM 对标），每份都带 **"错误账本"**：先后 6 个 + 5 个被
  后续测量推翻的结论，逐条记录错因与更正时机；
- **六层源码讲义 + 5 个纯 Python 演示**（采样等价性、pickle 瘦身 79%、KV 块分配/引用计数/前缀命中/抢占、
  张量形状推演、共享内存 RPC），把"读懂"变成了可复现的练习。

---

## 5. 关键数据（可写进简历的部分）

### 5.1 服务层吞吐与稳定性（RTX 4060 Laptop 8GB / Qwen3-0.6B）

| 阶段 | 并发 | 吞吐 | 备注 |
|---|---|---|---|
| 初始 | 8 / 64 | 775 / 857 tok/s | 引擎能力（单独驱动 batch 64 step）为 5689 tok/s |
| 修 `max_workers` 后 | 16 / 32 / 64 | 1290 / 1793 / **2930** | p99 2366 → 691 ms |
| 继续推并发 | **128** | **3803** | 累计约 **4.4×** |
| 并发上限 | 256 | 拒绝（`backpressure: memory`） | KV 池硬顶约 162 并发序列，过载不 OOM |

### 5.2 引擎 vs vLLM 0.31.0（同卡同数据同口径）

| 指标 | nano-vllm-plus | vLLM |
|---|---|---|
| 离线 ms/step（batch 32） | **15.54** | 16.39 |
| 在线 ms/step（batch 32） | 20.95 | 17.22 |
| 服务层税 | **+34.8%** | +5.1% |
| 在线 TTFT | c=1 优 14% / c=8 优 9% / c=48 优 7% | — |

### 5.3 自研 Triton decode attention（sm_89）

| 项 | 结果 |
|---|---|
| 正确性 | 最差误差 1.95e-3（判据 1e-2），29 项测试全过 |
| 延迟 | 均值 **1.00x**（0.92–1.14x）；batch≥4 场景 1.01–1.08x |
| 硬件计数器 | `dram__bytes_read` 与 flash-attn 完全相同（67.35 vs 67.38 MB）；DRAM 88.3% vs 95.1% |
| 剩余天花板 | 约 13 µs / 4.9%（该形状） |
| 显存 | 5767 vs 5764 MiB（无增加） |

### 5.4 环境与硬件常数（实测，非标称）

bf16 算力 28.9–29.3 TFLOP/s；读带宽 238 GB/s（`torch.sum`）～263.7 GB/s（ncu 反推理论峰值）；
寄存器 256 KiB/SM、L1+shared 100 KiB/SM、L2 24 MiB；权重 1136.8 MiB vs 全卡片上合计 32.3 MiB
（差 35.1×）；**机器平衡点 122–123 FLOP/byte**；每 token KV（每序列）112 KiB。

> 由此得到一个贯穿全项目的判据：**算术强度 ≈ batch**。低于平衡点时优化算力收益为零，只能提复用 /
> 减字节 / 分摊多卡；ctx ≥ ~512 后 KV 字节主导，batch 失效（只能 KV 量化）。
> 实测印证：batch 1→32 计算量翻 32 倍，耗时几乎不变（0.053 → 0.052 ms）。

---

## 6. 评测/验证体系

| 层次 | 手段 |
|---|---|
| 功能 | 37 项单元测试；mock 引擎全链路无 GPU 测试；真实 `LLM.generate` 输出 token 流与 flash-attn 后端逐 token 对齐（eager 5/5 一致，CUDA graph 4/5，且 flash-attn 自身 graph/eager 也会分叉，属 fp 噪声） |
| 性能 | CUDA events 步级剖析（host/gpu 时长 + 真实 batch 直方图）；nsys（`--cuda-graph-trace=node` + `cudaProfilerApi` 夹住稳态）；ncu 硬件计数器；`triton.testing.do_bench` + L2 flush |
| 对比 | 与 vLLM 0.31.0 的离线 + 在线（TTFT/TPOT/吞吐）对照，KV 池字节对齐、数据 seed 每次更换、每点重复报离散度 |
| 过载 | 双重背压触发路径（qps / memory）实测拒绝而非 OOM；取消路径（客户端断流 / 显式 Abort RPC）对比回收延迟 |

---

## 7. 技术栈

Python 3.10–3.12 · PyTorch · Triton（手写 kernel）· flash-attn · gRPC / protobuf · Redis ·
Prometheus · multiprocessing / 共享内存 / NCCL · CUDA Graph · torch.compile · nsys / ncu ·
WSL2 + CUDA passthrough · unittest

---

## 8. 难点与踩坑（面试可展开的素材）

**并发与取消**
1. 引擎非线程安全 ⇒ 单线程 + 命令队列的所有权设计；abort 只投命令，不直接碰 scheduler。
2. cancel 链路要幂等：对已完成/已中止的 sequence 返回 False 而非抛错。
3. 流式 gRPC 与线程池的交互陷阱（见下）。

**两个"跟 GPU 毫无关系"的性能陷阱**（同一个模式的两次出现：一个默认值悄悄锁死上限，只在压测时暴露）
4. `max_workers=10`：不是线程池大小，是并发请求数上限 = 引擎 batch 上限 → 值 3.4×。
5. `AdmissionController(max_qps=100, burst=100)`：令牌桶被 256 个同时到达的请求打爆，
   压测直接 `RESOURCE_EXHAUSTED`。

**测量口径的坑（每一个都曾导致错误结论）**
6. **粒度错位**：拿"一层 attention"除以"整步"地板 ⇒ attention 占比低估 28 倍。
7. **反推代替直测**：用 gRPC p50 反推步效率得 50%，直测是 73–89%。
8. **带宽口径**：按"加载单元发出的字节"算带宽，忽略 24 MB L2 吸收了 GQA 的 2× 重复读 ⇒ 算出 >100% 的假峰值。
9. **冷启动污染**：`warmup_model()` 跑在 `allocate_kv_cache()` **之前** ⇒ 预热时 `k_cache.numel()==0`
   ⇒ `store_kvcache` 整条路被跳过 ⇒ 首次真实 prefill 2627 ms vs 稳态 167 ms（离线 n=8 的 TTFT 被污染 7 倍）。
10. **缓存永不失效**：`hash_to_block_id` 从不清空 ⇒ 重跑同一批 prompt 静默变成前缀缓存命中。
11. **"把资源推满"不是中性测试**：把 KV 顶到物理极限（+26%）反而单方面惩罚 vLLM
    （prefill 慢 3 倍、TTFT 1839 ms），因为两个引擎对"预算内 KV vs 激活"的分配策略不同。
12. **工具坑**：nsys 默认把 CUDA graph 内核折叠成单节点（须 `--cuda-graph-trace=node`）；
    `nsys stats --format csv` 在表头前打印进度行会让 `csv.DictReader` 静默跳过所有行；
    WSL 性能计数器需 Windows 宿主机授权（root 也不行）；WSL `/tmp` 在两次调用之间可能被清空。

**已在文档中明确"不要做"的事（都是被测量证伪的）**
attention kernel 微优化 · GQA head 配对 · split-KV · num_stages 调优 · RoPE+KV store 融合 ·
把 KV 顶满 · 在不统一粒度的情况下比较 p99。

---

## 9. 当前状态与后续方向

**已完成**：服务层全链路（gRPC / Redis 路由 / 双背压 / 级联取消 / 指标）、
sm_89 Triton decode attention（Phase 1 交付 + 完整对比报告）、vLLM 横向对标（含口径定案）、
37 项测试、GPU-free 演示链路。

**未决（按预期收益排序）**：
1. **服务层进程架构**：把引擎拆到独立进程，消除与 servicer 共享 GIL 的 +34.8% 服务层税（预期最大杠杆）。
2. **验证一条线索**：高 batch + 极短 ctx 下 flash-attn 单次调用耗时超线性增长（batch 8→64 涨 19 倍），
   该形状只跑到约 7% 带宽——若成立，自研 kernel 在"服务真实工况"下可能反超。
3. **GEMM occupancy**（49% tensor / 16.6% occupancy，寄存器限制 2 blocks）——证据目前不足以给收益预期。
4. block_size 三处硬编码与 `assert % 256 == 0` 的不一致（已决定先写进报告、暂不改代码）。
5. `EngineWorker` 每步 `set(get_cached_prefix_hashes())` 是 O(已缓存 block 数) 的 Python 拷贝，长跑会变慢。

---

## 10. 多版本文案（可直接取用）

### 10.1 简历条目（详细版，4 条）

> **分布式 LLM 推理服务层 · nano-vllm-plus（个人项目）**
> 基于 nano-vllm（1200 行教学级 vLLM 复刻）自建完整在线推理服务，并用硬件级剖析完成瓶颈归因。
> - **服务化**：把阻塞式 `generate()` 改造为后台单线程 worker + 命令队列，接上 gRPC 流式推流、
>   Redis 前缀缓存路由、QPS 令牌桶 + KV 显存水位滞回双背压、gRPC 级联取消与显存回收、
>   Prometheus 指标，以及 GPU-free mock 全链路演示。
> - **性能**：压测发现 gRPC 线程池默认值 `max_workers=10` 把并发流锁死进而锁死引擎 batch，
>   在线吞吐 **857 → 3803 tok/s（约 4.4×）**，p99 **2366 → 691 ms**；过载时背压触发而非 OOM，
>   实测 KV 池硬顶约 162 并发序列。
> - **对标**：用同卡同数据同口径对比 **vLLM 0.31.0**，证明**引擎本体不输**（离线 batch32
>   15.54 vs 16.39 ms/step、在线 TTFT 低并发优 7–14%），并定位高并发 17% 差距
>   **100% 来自服务层**（服务层税 +34.8% vs +5.1%），根因是引擎与 servicer 共享 GIL。
> - **算子**：为 RTX 4060（sm_89）手写 Triton paged decode attention（online softmax / GQA / split-KV），
>   29 项测试、误差 1.95e-3，ncu 证明其已贴 DRAM 墙且与 flash-attn **DRAM 字节完全相同**，
>   据此把 3 条优化路线实测证伪，避免无效投入。

### 10.2 简历条目（精简版，3 行以内）

> 为教学级 vLLM 复刻品 nano-vllm 自建在线推理服务层（gRPC 流式 + Redis 前缀路由 +
> 双背压 + 级联取消 + Prometheus），定位并修复 `max_workers` 默认值导致的
> batch 锁死，在线吞吐提升约 4.4×（857→3803 tok/s）；手写 sm_89 Triton paged decode attention
> 并用 nsys/ncu 证明瓶颈不在 kernel，与 vLLM 0.31.0 同口径对标得出"差距 100% 来自服务层进程架构"。

### 10.3 项目一句话（用于项目列表）

> **nano-vllm-plus** — 给 1200 行的 vLLM 教学复刻品补上生产向服务层，并用硬件计数器把
> "性能瓶颈在哪"从猜测做成证据：最大的两个瓶颈分别是一个 gRPC 配置值和进程架构，都不在 GPU 上。

### 10.4 面试 STAR 骨架

- **S**：想练"真实推理服务"而不是再写一个 demo；目标是在 8GB 消费级卡上跑出可压测、可观测、
  可取消的在线服务，并说清性能到底受限于什么。
- **T**：① 引擎不线程安全但必须服务化；② 要能过载不崩；③ 要能给出可信的性能结论而不是"感觉快了"。
- **A**：单线程 worker + 命令队列的所有权设计；双背压（QPS + 显存滞回）+ 级联取消；
  GPU-free mock 让全链路可测；上 nsys/ncu 做证据链；与 vLLM 做同口径对标。
- **R**：吞吐 4.4×、p99 降到 691 ms、过载不 OOM；手写 kernel 正确且贴墙但收益 1.00x；
  **最重要的产出是把两个"不在 GPU 上"的瓶颈用证据钉死**，并留下 11 条自我更正的错误账本。

### 10.5 如果被问"这个项目的缺点"

老实说三点：① 服务层仍与引擎同进程（GIL 税 +34.8%）是已知未修的架构债；
② 单机单卡、多实例路由只在 mock 与逻辑层验证，没有真实多卡多实例的端到端证据；
③ block_size 受上游 `assert % 256 == 0` 限制，导致与 vLLM（16）对比时**内存碎片口径不可比**
（我们每条 640-token 序列占 768 槽 = 浪费 20%，vLLM 占 640 槽 = 0 浪费），这一点已在报告中显式披露。
