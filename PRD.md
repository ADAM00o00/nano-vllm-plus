# Nano-vLLM Plus PRD

> 项目定位：在 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 之上，构建一个**分布式 LLM 推理服务层**，用于简历工程实践。
> 三大工程点：
> 1. 基于 Redis 的前缀缓存路由（多实例）
> 2. QPS + 显存水位的双重背压防护
> 3. 基于 gRPC/Context cancellation 的分布式级联取消与显存回收
>
> 本文档描述架构设计、功能需求、分步实现方案与验收标准。实现语言：**Python（服务层）+ Java/Python（控制面可选）**。

---

## 1. 现状分析（基于当前仓库代码）

| 现有模块 | 能力 | 与本项目的关系 |
|---|---|---|
| `nanovllm/engine/block_manager.py` | 已有 prefix cache：`hash_to_block_id`，`can_allocate/allocate/deallocate/hash_blocks` | 路由和回收的基础，需补充对外查询接口 |
| `nanovllm/engine/scheduler.py` | `waiting/running` 队列、`preempt`、`postprocess` | 需补充 `abort` 和队列深度/水位查询 |
| `nanovllm/engine/sequence.py` | `SequenceStatus`（WAITING/RUNNING/FINISHED） | 需补充 `ABORTED` 状态 |
| `nanovllm/engine/llm_engine.py` | `add_request`、`step`、阻塞式 `generate` | 需拆出后台 step 循环，支持服务化调用 |
| `nanovllm/config.py` | 推理相关配置 | 需增加服务层配置（端口、Redis 地址等，放在 serving 层，不改核心） |
| `nanovllm/llm.py` | `LLM(LLMEngine)` 薄封装 | 保持兼容，不改 |

**关键约束**
- `generate()` 是阻塞循环（`while not is_finished(): step()`），不能直接服务化，必须引入后台 Engine Worker。
- 引擎内部（Scheduler/BlockManager/Sequence）非线程安全，所有引擎操作必须收敛到**单一线程**执行。
- 当前开发环境 Python 3.14 不满足项目要求（`>=3.10,<3.13`），需要准备 Python 3.10~3.12 环境。
- 依赖 `torch`、`flash-attn`，建议先跑通 `example.py` / `bench.py` 再开发。

---

## 2. 总体架构

```
Client
  │ gRPC streaming
  ▼
┌──────────────────────── 控制面（Gateway）────────────────────────┐
│ 1. 前缀路由：Redis 前缀目录 + 实例心跳/负载                        │
│ 2. 准入控制：QPS 令牌桶 + 显存水位高/低水位滞回                    │
│ 3. 级联取消：gRPC context cancelled → 下发 Abort                  │
└───────────────────────────┬──────────────────────────────────────┘
                            │ gRPC
                            ▼
┌──────────────────────── 服务层（新增 nanovllm/serving/）─────────┐
│ EngineWorker：后台 step 循环 + 命令队列（add/abort）               │
│ gRPC Server：Infer(streaming) / Abort / Health                    │
│ AdmissionController：QPS + 水位                                    │
│ Metrics：Prometheus                                                │
└───────────────────────────┬──────────────────────────────────────┘
                            │ 直接调用
                            ▼
┌──────────────────────── 引擎层（现有 nanovllm/engine/）──────────┐
│ LLMEngine → Scheduler → BlockManager（prefix cache）              │
└───────────────────────────────────────────────────────────────────┘
```

**设计原则**
1. 引擎核心逻辑（调度、缓存、模型运行）尽量不动，只开必要接口。
2. 所有新增服务化逻辑放在 `nanovllm/serving/`，与引擎解耦。
3. 先 Python 全链路跑通，Java 控制面作为可选二期（Java 需要解决 tokenizer 问题，见风险节）。

---

## 3. 功能需求

### FR-1 引擎基础改造
- `SequenceStatus` 增加 `ABORTED`。
- `BlockManager` 暴露显存水位和前缀缓存信息。
- `Scheduler` 支持按 `seq_id` 中止请求并回收显存。
- `LLMEngine.add_request` 返回 `seq_id`，新增 `abort_request`。

### FR-2 Engine Worker（服务化核心）
- 后台单线程运行 `step()` 循环。
- 线程安全命令队列：`add`、`abort`、`stop`。
- 流式输出：每个正在生成的序列增量吐出 completion tokens。

### FR-3 gRPC 服务与级联取消
- `Infer`：接收 prompt / token_ids，流式返回生成 token。
- `Abort`：按 `request_id` 中止请求。
- 客户端断开 → gRPC context cancelled → 触发引擎 abort → 释放 KV cache。

### FR-4 双重背压防护
- QPS 维度：令牌桶限流，超限拒绝（`RESOURCE_EXHAUSTED`）。
- 显存维度：KV cache 使用率超过高水位拒绝，低于低水位恢复，中间滞回。
- 队列深度：等待队列过长可视为辅助信号（v1 可选）。

### FR-5 Redis 前缀缓存路由
- 多实例注册与心跳。
- 请求前缀 hash → 实例路由。
- 路由失效与降级：目录 miss / 实例失联时退化为负载最轻路由。

### FR-6 可观测性
- Prometheus 指标：请求数、QPS、队列深度、显存水位、前缀命中率、abort 数、回收延迟。

### FR-7 压测与验收
- 对比实验，产出可写入简历的量化结果。

---

## 4. 分步实现方案

### Step 0：环境准备（预计 2~3 天）
1. 准备 Python 3.10~3.12（建议 3.10 或 3.11，flash-attn 兼容性最好）。
2. 安装依赖：`torch`、`flash-attn`、`transformers`、`xxhash`、`grpcio`、`grpcio-tools`、`prometheus-client`、`redis`。
3. 下载模型并跑通 `example.py` 与 `bench.py`，记录 baseline 吞吐。
4. 安装 Redis（建议 Docker：`docker run -p 6379:6379 redis:7`）。

**验收**：`bench.py` 正常出吞吐数字。

---

### Step 1：引擎基础改造（预计 2 天）

**1.1 `nanovllm/engine/sequence.py`**
- `SequenceStatus` 增加 `ABORTED = auto()`。
- 增加便捷属性 `is_aborted`。

**1.2 `nanovllm/engine/block_manager.py`**
- 新增 `num_free_blocks` 属性：`len(self.free_block_ids)`。
- 新增 `usage_ratio` 属性：`1 - num_free_blocks / len(self.blocks)`。
- 新增 `get_cached_prefix_hashes()`：返回 `list(self.hash_to_block_id.keys())`，供路由上报。
- 可选重构：把 `can_allocate` 里“探测前缀命中块数”的逻辑抽成 `find_cached_prefix_blocks(seq)`，方便后续做命中率统计。

**1.3 `nanovllm/engine/scheduler.py`**
- 新增 `cache_usage_ratio` 属性：委托给 `block_manager.usage_ratio`。
- 新增 `num_waiting` / `num_running` 属性。
- 新增 `abort(seq_id) -> bool`：
  1. 在 `waiting` / `running` 中查找 `seq_id`；
  2. 从队列移除；
  3. 若 `seq.block_table` 非空，调用 `block_manager.deallocate(seq)` 回收显存；
  4. 置 `SequenceStatus.ABORTED`；
  5. 返回是否成功。
- 注意：`deallocate` 已存在且会释放 block，直接复用即可。

**1.4 `nanovllm/engine/llm_engine.py`**
- `add_request` 返回 `seq`（或 `seq.seq_id`）。
- 新增 `abort_request(seq_id)`：委托 `scheduler.abort`。
- 新增 `cache_usage_ratio` / `num_waiting` / `num_running` 只读属性。
- 新增 `get_cached_prefix_hashes()`：委托 `scheduler.block_manager`。

**验收**：
```python
llm = LLM(model, ...)
seq_id = llm.add_request("hello", SamplingParams(max_tokens=16))
llm.step()
assert llm.abort_request(seq_id) is True
assert llm.cache_usage_ratio >= 0
```

---

### Step 2：Engine Worker（预计 3~4 天）

新增 `nanovllm/serving/__init__.py` 与 `nanovllm/serving/engine_worker.py`。

**设计**
- `EngineWorker` 持有 `LLMEngine`，启动一个 daemon 线程。
- 命令队列 `queue.Queue`，命令类型：
  - `("add", prompt, sampling_params, handle)` → 调用 `llm.add_request`，绑定 `handle.seq_id`；
  - `("abort", seq_id)` → 调用 `llm.abort_request`，通知对应 handle；
  - `("stop",)` → 退出循环。
- 主循环：
  1. 非阻塞/带超时地消费命令队列；
  2. 若 `not llm.is_finished()`，执行一次 `llm.step()`；
  3. 对 finished 的 seq 推送完整 completion 并标记完成；
  4. 对 running 的 seq 按游标增量推送新 completion tokens；
  5. 无任务时阻塞等待命令。
- `RequestHandle`：包含 `seq_id`、输出队列 `queue.Queue`、完成标志、abort 标志、已推送 token 游标。
- 所有 `llm`/`scheduler`/`seq` 的访问只发生在 worker 线程内。

**要点**
- `step()` 每次 decode 产生 1 个新 token，prefill 完成后产生 1 个新 token；用 `seq.completion_token_ids` 与游标计算增量。
- abort 的延迟 = 一个 step 的耗时，这是可接受的回收粒度。

**验收**：
- 多线程同时提交 N 个请求，worker 能持续 batching 输出。
- `abort` 后，对应 handle 收到 abort 通知，`block_manager` 的 free blocks 回升。

---

### Step 3：gRPC 服务（预计 2~3 天）

**3.1 新增 `nanovllm/serving/proto/nanovllm.proto`**

```proto
service NanoVLLMService {
  rpc Infer(InferRequest) returns (stream InferResponse);
  rpc Abort(AbortRequest) returns (AbortResponse);
  rpc Health(HealthRequest) returns (HealthResponse);
}
```
- `InferRequest`：`request_id`、`prompt`（可选）、`prompt_token_ids`（可选）、`temperature`、`max_tokens`、`ignore_eos`。
- `InferResponse`：`request_id`、`token_ids`（增量）、`finished`、`aborted`。

**3.2 生成代码**
- 用 `grpcio-tools` 生成 `nanovllm_pb2.py` / `nanovllm_pb2_grpc.py`。
- 提供一个 `nanovllm/serving/generate_proto.py` 脚本，一行生成。

**3.3 新增 `nanovllm/serving/grpc_server.py`**
- 启动时创建 `EngineWorker` + `AdmissionController` + Prometheus。
- `Infer` 处理器：
  1. 准入检查（Step 5）；
  2. 提交 `add` 命令；
  3. 循环读 handle 输出队列，写流式响应；
  4. 监听 `context.cancelled()` / `context.is_active()`，客户端断开则提交 `abort`。
- `Abort` 处理器：按 `request_id` 找 `seq_id`，提交 abort 命令。
- 维护 `request_id → seq_id` 映射（线程安全 dict + lock）。

**验收**：
- grpc client 正常流式收到 tokens。
- 客户端断开后，服务端日志显示 abort 被处理，显存回落。

---

### Step 4：级联取消（可与 Step 3 合并，预计 1~2 天）

**链路**：
```
Client cancel → gRPC context cancelled（网关/服务端自动感知）
  → grpc_server 提交 ("abort", seq_id)
  → EngineWorker 在下一步前执行 llm.abort_request(seq_id)
  → Scheduler.abort → BlockManager.deallocate → 显存回收
```

**要点**
- `Infer` 处理器内用 `while not handle.done:` 循环发送 token，同时在每次循环检查 `context.is_active()`。
- 取消是幂等的：`abort_request` 对已完成/已中止的 seq 返回 False，不报错。
- 记录指标：`abort_total`、`reclaim_duration_seconds`（abort 前后 `cache_usage_ratio` 变化时间）。

**验收**：
- 压测：N 个长请求，中途断开 50%，观察 free blocks 数量明显回升。

---

### Step 5：双重背压防护（预计 1~2 天）

新增 `nanovllm/serving/admission.py`。

**5.1 QPS 限流**
- 令牌桶：容量 = burst，速率 = max_qps，配置化。
- 取不到令牌直接拒绝。

**5.2 显存水位**
- 数据来源：worker 每次 `step()` 后更新 `latest_cache_usage_ratio`（原子变量或 lock）。
- 高水位 `high_watermark`（默认 0.90）：超过则拒绝新请求。
- 低水位 `low_watermark`（默认 0.70）：降到低水位以下才恢复接收。
- 滞回区间（0.70~0.90）内维持上一状态，避免抖动。

**5.3 拒绝语义**
- gRPC 返回 `RESOURCE_EXHAUSTED`，客户端可重试或降级。
- 指标 `admission_rejected_total` 记录拒绝原因（qps / memory）。

**验收**：
- 过载测试：持续高压下服务不 OOM，p99 受控，拒绝率稳定。

---

### Step 6：Redis 前缀缓存路由（预计 2~3 天）

新增 `nanovllm/serving/redis_router.py`（Python 版控制面，可被 Java 版替换）。

**6.1 Redis Key 设计**

| Key | 类型 | 内容 | TTL |
|---|---|---|---|
| `nano:instance:{instance_id}` | hash | `addr`、`cache_usage_ratio`、`prefix_count`、`last_seen` | 10s（心跳刷新） |
| `nano:instances` | set | 存活 instance_id | - |
| `nano:prefix:{prefix_hash}` | string | 最近缓存该前缀的 `instance_id` | 30s |

**6.2 实例侧**
- 启动时注册 `nano:instance:{id}`，每 5s 心跳。
- 心跳时上报 `cache_usage_ratio`、`get_cached_prefix_hashes()` 的数量（或抽样上报 prefix hash 列表）。

**6.3 路由侧**
1. 计算请求前缀 hash（取 prompt 前 `kvcache_block_size` 个 token，或完整 system prompt）。
2. `GET nano:prefix:{hash}`：
   - 命中且实例存活、负载未超上限 → 路由到该实例；
   - 否则从 `nano:instances` 中选 `cache_usage_ratio` 最低的实例。
3. 路由成功后 `SET nano:prefix:{hash} {instance_id} EX 30`（尽力而为）。

**注意**
- **tokenizer 问题**：计算前缀 hash 需要 tokenizer。Python 版控制面可直接加载 tokenizer；Java 版控制面需要客户端传 hash，或起一个 Python tokenizer 服务。
- 目录是“尽力而为”，引擎可能因内存压力逐出缓存，TTL 兜底。
- v1 先做单实例 + 一个 Python 路由客户端，验证命中率提升；v2 再做多实例。

**验收**：
- 构造共享 system prompt 的请求流，对比随机路由：prefix cache 命中率显著提升。

---

### Step 7：Prometheus 指标（预计 1 天）

新增 `nanovllm/serving/metrics.py`，使用 `prometheus_client`。

| 指标 | 类型 | 说明 |
|---|---|---|
| `nano_vllm_requests_total` | Counter | 总请求数 |
| `nano_vllm_inflight_requests` | Gauge | 当前在途请求 |
| `nano_vllm_queue_depth` | Gauge | waiting 队列深度 |
| `nano_vllm_cache_usage_ratio` | Gauge | KV cache 使用率 |
| `nano_vllm_prefix_cache_hit_total` | Counter | 前缀缓存命中请求数（可选） |
| `nano_vllm_prefix_cache_miss_total` | Counter | 前缀缓存未命中请求数（可选） |
| `nano_vllm_admission_rejected_total` | Counter | 背压拒绝数（按原因 label） |
| `nano_vllm_abort_total` | Counter | 取消/中止数 |
| `nano_vllm_reclaim_duration_seconds` | Histogram | abort → 显存回收耗时 |

Prometheus HTTP 端口单独开（如 `9100`），与 gRPC 端口分离。

**验收**：`curl localhost:9100/metrics` 能看到上述指标。

---

### Step 8：Benchmark 与验收（预计 2~3 天）

**8.1 前缀路由实验**
- 场景：256 个请求，80% 共享同一个长 system prompt。
- 对比：随机路由 vs 轮询 vs Redis 前缀路由。
- 指标：prefix cache 命中率、TTFT、p99 latency、吞吐。

**8.2 背压实验**
- 场景：持续高 QPS 压力 + 长输出请求填充显存。
- 对比：开/关背压。
- 指标：是否 OOM、拒绝率、p99。

**8.3 级联取消实验**
- 场景：长请求流式输出中随机断开 50% 客户端。
- 对比：开/关级联取消。
- 指标：显存水位回落时间、后续请求成功率。

**8.4 产出**
- `benchmarks/` 目录放脚本和结果图表。
- README 增加架构图、Quick Start、实验结果。

---

## 5. 里程碑与工时

| 里程碑 | 内容 | 预计天数（1h/天） |
|---|---|---|
| M0 | 环境跑通 baseline | 2~3 |
| M1 | 引擎改造 + abort + 水位指标 | 2 |
| M2 | Engine Worker 服务化 | 3~4 |
| M3 | gRPC + 级联取消 | 2~3 |
| M4 | 背压 + Prometheus | 2~3 |
| M5 | Redis 前缀路由 | 2~3 |
| M6 | 压测 + README + 简历数据 | 2~3 |
| **合计** | | **15~20 天** |

---

## 6. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| Python/GPU 环境不兼容 | 阻塞 M0 | 固定 Python 3.10/3.11 + Docker/conda 环境；先跑通原版 bench.py |
| `generate` 阻塞改造成 worker 后出现竞态 | M2 返工 | 所有引擎操作收敛到单线程命令队列；abort 只发命令，不直接操作 scheduler |
| flash-attn 在 Windows 安装失败 | 阻塞 M0 | 使用 WSL2 或 Linux 容器；或临时用 eager 模式跑通 |
| 前缀路由 tokenizer 在 Java 侧不可用 | M5 延期 | 一期用 Python 控制面；Java 版由客户端传 hash |
| Redis 目录 stale 导致路由到已逐出缓存的实例 | 命中率下降 | TTL + 心跳兜底，定位为“尽力而为” |
| 单机显存无法同时跑多实例 | M5 无法验证 | 降低 `gpu_memory_utilization` 跑 2 个小实例，或用 CPU 模拟 engine 做路由逻辑测试 |

---

## 7. 简历呈现（最终产出）

> **分布式 LLM 推理服务层（nano-vllm + gRPC + Redis）**
> - 为 nano-vllm 增加服务化能力：后台 step 循环 + gRPC streaming + 请求级 abort；
> - 设计 Redis 前缀缓存目录路由，多实例 prefix cache 命中率提升 x%；
> - 实现 QPS + KV-cache 显存水位双重背压，过载下零 OOM、p99 控制在 x ms；
> - 实现 gRPC 级联取消与显存即时回收，客户端断开后回收延迟 x ms。
