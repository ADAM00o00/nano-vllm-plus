# 第 6 层讲义：张量并行与多进程 —— 把模型拆到多张卡上

> 配套: `python practice/demo_rpc.py`(共享内存 RPC 协议演示, 已就绪)
> 阅读: `model_runner.py` 的分布式部分 + `linear.py` + `embed_head.py`
> 前置: 第 2 层(pickle 瘦身) + 第 4 层(并行线性层) + 第 5 层(prepare/run)

---

## 0. 先回答: 为什么要张量并行(TP)?

一张卡放不下模型权重(或放得下但算不快)时, 把**每一层的权重**按维度切开, 分给多张卡, 各卡算一部分, 再通信合并。

```
Data Parallel(数据并行):  每卡一份完整模型, 各算各的数据 → 只解决"吞吐", 不解决"放不下"
Tensor Parallel(张量并行): 每卡 1/tp_size 的权重, 协作算同一批数据 → 解决"放不下" + 提速
```

Qwen3-0.6B 权重 1.47GB(bf16)单卡没问题, 但 7B/72B 模型必须 TP。nano-vllm 把 TP 的实现完整写出来了——这是它最"像 vLLM"的地方。

## 1. 进程架构: rank 0 在主进程, 其他 rank 在子进程

```python
# LLMEngine.__init__ (第 1 层讲过)
ctx = mp.get_context("spawn")
for i in range(1, config.tensor_parallel_size):     # rank 1..n-1 开子进程
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
    self.ps.append(process); self.events.append(event)
self.model_runner = ModelRunner(config, 0, self.events)   # rank 0 就在主进程
```

```
┌─ 主进程 (rank 0) ─────────────┐     ┌─ 子进程 (rank 1) ─────────────┐
│ ModelRunner(config, 0, events) │     │ ModelRunner(config, 1, event)  │
│  - 建模型(权重第 0 片)          │     │  - 建模型(权重第 1 片)          │
│  - call("run", ...):           │     │  - loop():                     │
│    ① write_shm 广播任务         │────▶│    ① event.wait() 醒来         │
│    ② 自己跑模型                 │     │    ② read_shm 读任务           │
│    ③ 采样(只有 rank 0)          │     │    ③ 跑模型(自己的那一片)       │
└───────────────────────────────┘     └───────────────────────────────┘
```

**为什么 rank 0 不也 spawn?** 主进程本来就在, 省一次进程创建; 而且采样、调度器、tokenizer 都在主进程, rank 0 放主进程最自然。

**为什么 spawn 而不是 fork(复习第 1 层)**: fork 会复制已经初始化 CUDA 的进程, 子进程的 GPU 状态是坏的。spawn 重新 import 模块、干净启动——这是 PyTorch 多进程 + CUDA 的黄金法则。

## 2. 分布式初始化

```python
dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
torch.cuda.set_device(rank)        # 每进程钉在自己的卡上
```

- **backend="nccl"**: NVIDIA 的 GPU 通信库(英伟达集体通信), all_reduce/gather 都走它
- **`"tcp://localhost:2333"`**: 进程组用 TCP 做"集合点(rendezvous)"——所有 rank 连到这个地址互相认识。这里固定 localhost + 2333(单机多卡), 被占用就得改
- **必须所有 rank 同时调用**这一行, 否则互相等不到, 挂死
- `world_size=1` 也会初始化: NCCL 单进程模式, 正常工作(example.py 默认就是它)

## 3. 权重怎么切(第 4 层复习 + 加深)

四种并行线性层, 记住一张表:

| 层 | 切法 | 通信 | 对应模型部分 |
|---|---|---|---|
| `ColumnParallelLinear` | 输出维切成 tp_size 份 | 无(各算各的输出段) | QKV/gate/up 的投影 |
| `MergedColumnParallelLinear` | 两个 Column 拼一个权重再切 | 无 | gate_up_proj |
| `QKVParallelLinear` | Q/K/V 各自按"头"切 | 无 | qkv_proj |
| `RowParallelLinear` | 输入维切成 tp_size 份 | **all_reduce 求和** | o_proj / down_proj |
| `VocabParallelEmbedding` | 词表行切成 tp_size 份 | **all_reduce**(mask 清零后求和) | embed_tokens |
| `ParallelLMHead` | 词表行切成 tp_size 份 | **gather**(只 rank 0 收全) | lm_head |

**一次前向的通信轨迹**: embedding 后 all_reduce 一次 → 每个 attention 的 o_proj all_reduce 一次 → 每个 MLP 的 down_proj all_reduce 一次 → 28 层 × 2 次 = 56 次 all_reduce → 最后 LMHead gather 一次。通信都发生在"行并行"的归约点——这是 TP 的标准模式(计算尽量本地, 只在该合并时通信)。

**bias 只在 rank 0 加**(RowParallelLinear): 因为 all_reduce 会把各 rank 的 bias 加 tp_size 次。

**QKV 按头切的意义**: 每卡拿到**完整连续的若干个头**(Q: 16/tp 头, KV: 4/tp 头), 注意力按头并行不需要跨卡通信。这也是为什么 KV 缓存也要按头切: `num_kv_heads = total // world_size`, 每卡只存自己的那几头 → **KV 缓存直接除以 tp_size**(第 5 层 block_bytes 公式里就是这么算的)。

## 4. 任务广播: 共享内存 + 事件的手写 RPC

`demo_rpc.py` 已经把这个协议跑通了, 这里对照真实代码:

```python
def write_shm(self, method_name, *args):      # 只有 rank 0 调
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")   # 长度前缀(4字节小端)
    self.shm.buf[4:n+4] = data                     # 数据体
    for event in self.event:                       # 每个 worker 一个 event
        event.set()

def read_shm(self):                           # 只有 rank>0 调
    self.event.wait()                             # 阻塞等通知
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()                            # ★ 立即消费, 允许下一次 set
    return method_name, args

def loop(self):                               # worker 主循环
    while True:
        method_name, args = self.read_shm()
        self.call(method_name, *args)             # 执行任务
        if method_name == "exit":
            break

def call(self, method_name, *args):           # 主进程入口
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)        # ① 先广播
    method = getattr(self, method_name, None)     # ② 自己执行
    return method(*args)
```

**要点**:

1. **为什么用共享内存 + pickle, 不用 mp.Queue**: Queue 底层有锁和额外线程, 慢; 共享内存是"主进程写完、worker 直接读同一块内存", 极快, 且架构一目了然。
2. **长度前缀**: 共享内存是定长 buffer, 消息是变长的 → 前 4 字节存长度, 读写双方约定好。`int.to_bytes(4, "little")` / `int.from_bytes(..., "little")` 是"普通开发几乎不用、通信协议里天天见"的字节操作。
3. **`event.clear()` 必须在读完后立即执行**(时序保证): worker 清得早, 主进程下一次 `event.set()` 才不会丢。真实引擎里主进程写完任务还要自己跑模型(耗时), 天然给 worker 留了清除时间——demo_rpc.py 用 sleep 模拟了这个窗口。
4. **`getattr(self, method_name)` 动态分发**: 把"方法名字符串"变成调用, 主进程/worker 共用同一套 `call`, 只是入口不同。任务可以是 "run"、"exit" 或任何 ModelRunner 方法。
5. **pickle 的内容里就有 Sequence**(`call("run", seqs, ...)`)——第 2 层的 `__getstate__` 瘦身在这里真正派上用场: **decode 阶段跨进程只传 1 个 last_token, 而不是整串历史**(demo_rpc 任务 2 演示了瘦身后的样子)。

**SharedMemory 生命周期**(必须成对):
- rank 0: `SharedMemory(name="nanovllm", create=True, size=2**20)` 创建(1MB)
- 其他 rank: `dist.barrier()` 同步后按名字 attach
- 退出: 各 rank `shm.close()`(解除映射) → barrier → **只有 rank 0 `shm.unlink()`**(销毁)

**`dist.barrier()` 不常见①**: 集体同步点——所有 rank 到齐才放行。attach 前 barrier 保证共享内存已创建好; exit 时 barrier 保证没人还在读就销毁。

## 5. 输出怎么合并: 只有 rank 0 出结果

```python
# ParallelLMHead (第 4 层)
if self.tp_size > 1:
    all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if rank==0 else None
    dist.gather(logits, all_logits, 0)          # 各 rank 的词表分片 → rank 0 收齐
    logits = torch.cat(all_logits, -1) if rank==0 else None

# ModelRunner.run (第 5 层)
temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
```

- **gather vs all_reduce**: 采样只需要 rank 0 有完整 logits → gather(只有 0 拿结果); embedding 每个 token 都要完整向量继续往下算 → all_reduce(人人都有)。
- **worker 的 token_ids 是 None**: 它们把模型跑完就完事, 采样是 rank 0 的专属(这就是第 2 层说"worker 不需要 temperature"的原因)。
- 所以**整个引擎的输出管线只有 rank 0 在走**: scheduler → postprocess → decode, 全在主进程。

## 6. 退出协议

```python
def exit(self):
    if self.world_size > 1:
        self.shm.close()
        dist.barrier()                      # 等所有 rank 都解除映射
        if self.rank == 0:
            self.shm.unlink()               # 只有创建者能销毁
    if not self.enforce_eager:
        del self.graphs, self.graph_pool    # 释放 CUDA graph
    torch.cuda.synchronize()                # 等 GPU 上的活全部干完
    dist.destroy_process_group()            # 拆进程组(NCCL 资源释放)
```

主进程通过 `call("exit")` 广播 → worker 的 loop 收到 exit 跳出 → 各 rank 依次: 解除共享内存 → barrier → rank 0 unlink → 同步 GPU → 拆 NCCL。主进程再 `p.join()` 等所有子进程真正退出。**顺序错一步就可能挂死或泄漏 GPU 资源**——这也是 `atexit.register(self.exit)`(第 1 层)存在的意义: 保证退出路径一定执行。

## 7. 实践注意(WSL2 / 多卡)

| 事项 | 说明 |
|---|---|
| NCCL 需要 CUDA GPU | `tensor_parallel_size>1` 必须多卡; WSL2 里 NCCL 可用(驱动在 Windows 侧) |
| Windows 原生 Python 不行 | torch.distributed 的 NCCL 后端不支持 Windows —— 又一个必须 WSL 的理由 |
| `tcp://localhost:2333` 固定端口 | 被占用改 `model_runner.py` 里的地址 |
| `world_size=1` | 不 spawn、不建共享内存、不走 RPC, 一切退化为单进程(example.py 默认) |
| 显存 | KV 缓存按头切, 每卡只存 1/tp —— 多卡等于 KV 容量 × tp |
| 每卡权重 | 完整权重 ÷ tp, 但每层中间结果(如 Q 投影)每卡一份 —— 激活翻倍是 TP 的固有代价 |

## 8. 自测题

1. TP 和 Data Parallel 解决的核心问题有什么不同?
2. 为什么 rank 0 放主进程而不是也 spawn 一个?
3. embedding 用 all_reduce、LMHead 用 gather —— 为什么不同?
4. worker 收到的序列为什么是"瘦身版"的? 完整版会给谁?
5. `event.clear()` 为什么必须在 `read_shm` 里立即执行?
6. 为什么必须 `spawn` 不能 `fork`?
7. 一次 28 层前向, 每个 rank 要参与多少次 all_reduce?

<details>
<summary>答案</summary>

1. TP 解决"单卡放不下"(权重切分) + 提速; DP 只复制模型解决吞吐, 不解决放不下。
2. 省一次进程创建; 调度器/tokenizer/采样都在主进程, rank 0 放主进程最自然。
3. embedding 的输出人人要继续用(all_reduce 人人有); logits 只有 rank 0 采样用(gather 省广播)。
4. worker 跑模型只需要 last_token/计数/block_table(第 2 层); 完整序列(含温度/状态)只存在于主进程。
5. 清得早, 主进程下一次 set 才不会丢 —— 时序协议的核心约定。
6. fork 复制已初始化 CUDA 的进程, 子进程 GPU 状态损坏; spawn 干净启动。
7. 每层 2 次(o_proj + down_proj) × 28 层 = 56 次 all_reduce(embedding 还有 1 次)。
</details>

## 9. 练习

- 跑 `python practice/demo_rpc.py`, 对照第 4 节的真实代码逐行找对应。
- 把 `demo_rpc.py` 的 worker 数量改成 2(多开一个进程、多一个 event), 观察 write_shm 怎么广播。
- 环境就绪后(需 ≥2 卡): `tensor_parallel_size=2` 跑 example.py, 观察进程数、显存占用与 tp=1 的差异。
- **毕业练习**: 不看代码, 画出 `generate()` 从 prompt 到文本的完整流程图, 标注每个环节用到了哪一层的哪个机制。
