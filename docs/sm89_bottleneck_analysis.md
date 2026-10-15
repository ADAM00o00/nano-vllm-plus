# 实际运行瓶颈排查（RTX 4060 / Qwen3-0.6B / nano-vllm-plus）

> 工具：torch.profiler、CUDA events 自建 harness、**nsys 2026.1.3**、**ncu 2026.2.1**（本次安装）。
> 环境：WSL2 Ubuntu 26.04、conda `ml`、torch 2.13.0+cu130、triton 3.7.1、flash-attn 2.8.3.post1。
> 硬件实测：**28.9 TFLOP/s**（bf16 实测矩阵乘）、**238 GB/s**（实测显存读带宽）。
> 复现脚本：`benchmarks/profile_decode_step.py`、`benchmarks/profile_serving_step.py`、`benchmarks/benchmark_decode_attention.py`。

---

## 0. 先更正我之前说错的三件事

排查过程中我发现自己前几轮有三个错误结论，先列出来，**下面的内容以本文为准**：

| 之前的说法 | 实际 | 错因 |
|---|---|---|
| "步效率约 50%" | **73% ~ 89%** | 用 gRPC 的 p50 反推 step 时间，反推链条不成立 |
| "attention 占整步 <3%，kernel 优化天花板 ≤3%" | **ctx≈11 时 4.4%；ctx 2048 时 68%；ctx 4096 时 85%** | 拿"一层"的耗时除以"整步"的地板，**错了 28 倍**（一个 decode step 跑 28 层） |
| "attention 还有 ~18% 可赚（GQA 配对）" | **没有**。ncu 实测该 kernel 已在 **91% DRAM 峰值**，与 flash-attn 只差 3% | 之前的 18% 来自含分配开销的 benchmark；ncu 用硬件计数器否掉了 |

---

## 1. 结论：瓶颈排序

| # | 瓶颈 | 影响 | 状态 |
|---|---|---|---|
| **B1** | **gRPC `max_workers=10` 把并发流锁死在 10 → 引擎 batch 上不去** | 并发 64 时 **3.4×** 吞吐 | ✅ **已修复并验证** |
| **B2** | **decode 是纯显存带宽受限**（算术强度 ≈ batch，机器平衡点 122 FLOP/byte） | 决定一切优化方向 | ✅ 已用 ncu + 实测 GEMM 扫描证实 |
| **B3** | **decode GEMM 只有 49% tensor / 16.6% occupancy（寄存器受限）** | 短上下文路径的大头，**潜在 ~2×** | ⚠️ 已定位，未修 |
| **B4** | 负载下 step 从 6.35 ms 涨到 8.5 ms（+33%） | ~25% 吞吐 | ⚠️ 疑似 GIL，未证实 |
| **B5** | attention kernel | 长上下文占 68–85%，但**已在 91% DRAM 峰值** | ✅ 已到硬件墙，无需再优化 |

---

## 2. B1：服务层把 batch 锁死了（已修复，3.4×）

### 证据

给 `EngineWorker` 插桩，记录**每一步真实 batch 大小**（`benchmarks/profile_serving_step.py`）：

| 客户端并发 | 引擎实际 batch 直方图 | 吞吐 | 引擎忙占比 |
|---|---|---|---|
| 8（`max_workers=10`） | `{8: 139}` | 595 tok/s | 99.8% |
| 32（`max_workers=10`） | `{2:31, 8:15, 10:93}` ← **32 个请求在飞，引擎只看到 10 个** | 861 tok/s | 99.7% |
| 16（`max_workers=64`） | `{16: 139}` | 1290 tok/s | 99.7% |
| 64（`max_workers=64`） | `{64:31, 16:15}` | **2926 tok/s** | — |

引擎本身的能力（直接驱动 `LLMEngine`）：batch 32 = **3596 tok/s**，batch 64 = **5689 tok/s**。
所以原先的 ~800 tok/s 只有能力的 **1/4 ~ 1/7**。

### 原因

`grpc.server(futures.ThreadPoolExecutor(max_workers=10))`。**流式 `Infer` 会把一个工作线程占满整个流的生命周期**，所以 `max_workers` 不是"线程池大小"，而是**并发请求数的硬上限** → 也就是引擎能组成的 batch 上限。gRPC 的其余请求只能在队列里等。

### 修复

| 文件 | 改动 |
|---|---|
| `nanovllm/serving/grpc_server.py` | `serve(max_workers=10)` → `64`，并在 docstring 说明这是 batch 上限 |
| `nanovllm/serving/__main__.py` | `--max-workers` 默认 `10` → `64`，help 里写清后果 |
| `nanovllm/serving/gateway.py` | `serve_gateway(max_workers=10)` → `64`（同样的 bug：网关也占线程整个流） |

### 验证（走默认 CLI，不带任何 flag）

| 并发 | 修复前 | 修复后 | p99 前 → 后 |
|---|---|---|---|
| 8 | 595 | 572（无变化，8 ≤ 10） | 1108 → 1196 ms |
| 16 | 897 | **1290** | 681 → 427 ms |
| 32 | 861 | **1793** | 1360 → 509 ms |
| 64 | 857 | **2930（3.4×）** | 2366 → 691 ms |

`python -m unittest discover -s tests` → **60/60 通过**。

---

## 3. B2：decode 是纯带宽受限，"提复用"才是杠杆

用模型真实的 `qkv_proj` 形状（`[B,1024] × [1024,4096]^T`，权重 8.0 MiB）扫 batch：

| batch | ms | TFLOP/s | %算力峰值 | GB/s | %带宽峰值 | FLOP/byte | 谁到顶 |
|---|---|---|---|---|---|---|---|
| 1 | 0.053 | 0.16 | **0.5%** | 157.7 | 66% | 1.0 | DRAM |
| 4 | 0.055 | 0.61 | 2.1% | 152.4 | 64% | 4.0 | DRAM |
| 8 | 0.055 | 1.21 | 4.2% | 153.2 | 64% | 7.9 | DRAM |
| 16 | 0.055 | 2.43 | 8.4% | 154.7 | 65% | 15.7 | DRAM |
| 32 | 0.052 | 5.14 | 17.8% | 166.9 | 70% | 30.8 | DRAM |
| 64 | 0.056 | 9.53 | 33.0% | 160.6 | 68% | 59.4 | DRAM |
| 128 | 0.070 | 15.42 | 53.3% | 139.3 | 59% | 110.7 | DRAM |
| 256 | 0.096 | 22.31 | 77.1% | 114.4 | 48% | 195.0 | **compute** |
| 1024 | 0.292 | 29.43 | 102% | 64.7 | 27% | 455.1 | **compute** |

**batch 1 → 32，计算量翻 32 倍，耗时几乎不变（0.053 → 0.052 ms）。** 因为 `FLOP/byte = batch`，而机器平衡点是 `28.9e12/238e9 = 122`。低于 122 时算力白送，时间完全由"搬 1.19 GiB 权重"决定。

实测交叉点在 batch 128（DRAM）与 256（compute）之间，与预测 122 吻合。

**推论**：decode 场景下"优化 kernel 数学"收益为零；只有**提复用（加大 batch）**、**减字节（量化）**、**分摊到多卡**有效。B1 的修复正是"提复用"，所以它值 3.4×，而任何 attention kernel 优化最多值百分之几。

### 步效率（实测，非估计）

| batch | ctx | step 实测 | 带宽地板 | 步效率 |
|---|---|---|---|---|
| 1 | 11 | 5.98 ms | 5.01 ms | 83.8% |
| 8 | 11 | 6.35 ms | 5.05 ms | 79.6% |
| 16 | 11 | 6.97 ms | 5.09 ms | 73.1% |
| 32 | 11 | 8.90 ms | 5.18 ms | 58.2% |
| 64 | 11 | 11.25 ms | 5.35 ms | 47.5% |
| 8 | 2048 | 14.53 ms | 12.91 ms | 88.8% |

注意**步效率随 batch 增大反而下降**（79.6% → 47.5%）：因为越过平衡点后计算开始咬合，而 B3 说明 GEMM 本身效率不高。

---

## 4. B3：decode GEMM 只有 49% tensor / 16.6% occupancy（潜在 ~2×）

ncu 抓 decode 里的 cutlass GEMM（batch 8 与 64 结果一致）：

| 指标 | 值 | 含义 |
|---|---|---|
| `sm__throughput` | **48.7%** | SM 整体半忙 |
| `sm__pipe_tensor_cycles_active` | **48.7%** | 张量核心只有一半时间在干活 |
| `sm__warps_active` | **16.6%** | **occupancy 只有 1/6** |
| `launch__occupancy_limit_registers` | **2 blocks** | ← 寄存器把 occupancy 限制到 2 个 block |
| `launch__occupancy_limit_shared_mem` | 3 blocks | 显存限制更松，寄存器才是瓶颈 |
| `dram__throughput` | 14–17% ⚠️ | **这个数不可信**，见下 |

**解读**：occupancy 被寄存器限制到 2 blocks/SM（256/1536 线程），内存级并行度不够 → 既喂不满 DRAM，也喂不满张量核心 → 属于**延迟受限**。这是短/中上下文路径的**第二大杠杆**（ctx≈11 时 GEMM 占 kernel 时间约 85%）。

⚠️ **诚实标注**：`dram__throughput` 14–17% 与 step 级实测矛盾（batch 64 整步搬运 1.19 GiB / 11.25 ms = 106 GB/s = 44% 峰值）。若每个 GEMM 真的只有 17%，整步不可能 11.25 ms。怀疑 ncu replay 期间权重被 L2 命中（24 MiB L2 vs 单层权重 ~12 MB），所以 **DRAM% 这一项我不采信，occupancy/tensor 两项采信**。要证实需要单层隔离测量。

---

## 5. B5：attention kernel 已到硬件墙（**不要**再优化）

ncu 抓我们自己的 `_paged_decode_attention_kernel`（batch 8，ctx 2048）：

| 指标 | 值 |
|---|---|
| `dram__throughput` | **91.1% / 88.8%** ← 贴着显存峰值 |
| `sm__throughput` | 16.8% |
| `lts__throughput`（L2） | 29.8% |
| `sm__warps_active` | 31.7% |
| `gpu__time_duration` | 298 µs |

两条独立方法互证：单层逻辑 KV = 67.1 MB ÷ 298 µs = **225 GB/s = 实测 238 GB/s 的 94%**。
同形状 flash-attn = 290 µs → **我们与 flash-attn 只差 3%**。

### 由此否掉 GQA head 配对

我上一轮建议的"每个 program 处理一个 KV head 的两个 Q head，把 L2 流量和 load 指令砍半"：

- ncu 显示 **L2 只有 29.8%**、DRAM 已 91% → 瓶颈是 DRAM，不是 L2/指令发射
- GQA 重复读**早被 24 MiB L2 吸收了**（否则 DRAM 早就 182% 了，物理不可能）
- 配对**不减少任何 DRAM 字节** → 上限就是那剩下的 9% 余量

**结论：不做。** 同理已实测证伪的：split-KV（combine 开销 8–10 µs，净亏）、`num_stages`（1–4 无差异）、Phase 2 的 RoPE+KV store 融合（占比 ≤1.5%）。

### attention 占整步的真实比例（更正后）

| batch | ctx | attention / 整步 |
|---|---|---|
| 8 | 11（服务压测点） | **4.4%** |
| 8 | 512 | 36% |
| 8 | 2048 | **68%** |
| 8 | 4096 | **85%** |
| 16 | 2048 | 82% |

短上下文时 GEMM 占 ~85%；长上下文时 attention 占 68–85% 且已到 DRAM 墙。

---

## 6. B4：负载下 step 慢 33%（未解决）

- 单独驱动：batch 8 step = **6.35 ms**
- 真实 gRPC 负载下：同 batch 8 step = **8.5 ms**（+33%）
- 引擎忙占比 **99.8%**、host ≈ GPU → **不是空转，是 step 本身变慢**
- 怀疑：8~64 个 servicer 线程每个 token 都在做 protobuf 序列化/写 socket，与引擎线程抢 **GIL**；引擎每步的 Python 侧工作（scheduler / prepare_decode / `_publish_incremental` / `.tolist()`）被拉长到超过 6.35 ms 的 GPU 时间，于是 GPU 反而要等 CPU。
- **未证实**。验证方式：在后台跑 N 个纯 Python 线程（模拟 servicer 的序列化负载），看 step 时间是否随 N 上升。

---

## 7. 次要问题

`EngineWorker._run` 每步执行 `set(self.llm.get_cached_prefix_hashes())`，而它是 `list(self.hash_to_block_id.keys())` —— 每步一次 **O(已缓存 block 数)** 的 Python 拷贝。当前规模（几十个请求）影响很小，但长跑服务会随缓存增长线性变慢。建议改为增量维护或按需失效。

---

## 8. 优化建议（按预期收益排序）

| 优先级 | 建议 | 预期收益 | 成本/风险 |
|---|---|---|---|
| **P0** | ✅ **已完成**：`max_workers` → 64（含 gateway）+ 回归测试 | **3.4×**（并发 64） | 已修复，60/60 测试通过 |
| **P1** | **把 batch 顶到能力上限**：现在 conc 64 拿到 2930，而引擎能力 5689。检查 `max_num_seqs`/scheduler 是否放得下，测 conc 128/256；客户端 64 线程也可能成为瓶颈 | **+30% ~ +90%** | 低，纯参数/配置 |
| **P2** | **权重量化 fp8/int8**：每步权重从 1.19 GiB → 0.6 GiB，短上下文地板 5.05 → 2.6 ms | **~1.9×**（短上下文） | 中，需精度验证（Qwen3-0.6B 敏感） |
| **P3** | **修 GEMM occupancy**（B3）：换 cutlass tile 配置 / 试 fp8 GEMM（寄存器压力更低）；先复测确认寄存器限制 | **最多 ~2×**（短/中上下文 GEMM 路径） | 中高，需要逐 kernel 调 |
| **P4** | **KV 量化**：仅在长上下文有意义（B8/ctx2048 时 KV 1.75 GiB > 权重 1.19 GiB），地板 12.9 → 8.96 ms | 1.44×（长上下文）；ctx≈40 时 **0%** | 中，需精度验证 |
| **P5** | **消除 B4 的 33%**：若确认 GIL，把引擎放独立进程（PRD 已有 TP>0 子进程路径），或减少 servicer 每 token 的 Python 工作 | **~1.3×** | 中，架构改动 |
| **P6** | 修 §7 的 O(n) 前缀哈希拷贝 | 长跑稳定性 | 低 |

### 明确**不要**做的

- ❌ attention kernel 微优化（已在 91% DRAM 峰值，与 flash-attn 差 3%）
- ❌ GQA head 配对（不减 DRAM 字节，上限 9%）
- ❌ split-KV（实测净亏 8–10 µs）
- ❌ `num_stages` 调优（无差异）
- ❌ Phase 2 RoPE+KV store 融合（占比 ≤1.5%）

**一句话总结**：这个项目的性能瓶颈**从头到尾都不在 attention kernel**。真正的 P0 是服务层一个配置值（已修，3.4×），真正的结构性方向是**减少每步必须搬运的字节**（权重/KV 量化）与**提高每字节的复用**（batch），而不是把 kernel 写得更快。

---

## 9. 复现命令

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ml
cd ~/nano-vllm-plus

# step 时间 / 步效率 / 逐 kernel
python benchmarks/profile_decode_step.py --batch 8 --prompt-len 10 --steps 24 --torch-profiler

# 负载下引擎的真实 batch 与 step 时间
python benchmarks/profile_serving_step.py --model /home/adam/models/Qwen3-0.6B \
    --port 50051 --max-workers 64 --write /tmp/steps.json
python benchmarks/benchmark_grpc.py --target localhost:50051 --requests 64 --concurrency 64 --max-tokens 32

# nsys（注意：必须加 --cuda-graph-trace=node，否则 graph 内 kernel 会被折叠成一个节点）
nsys profile -o /tmp/t --trace=cuda --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    python benchmarks/profile_decode_step.py --batch 64 --prompt-len 10 --steps 12 --nsys-range
nsys stats --report cuda_gpu_kern_sum /tmp/t.nsys-rep

# ncu（需要 Windows 侧开启 GPU 性能计数器）
ncu --clock-control none -k regex:paged_decode_attention --metrics \
  gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed \
  python -c "from nanovllm.layers.decode_attention import *; ..."
```
