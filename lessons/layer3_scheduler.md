# 第 3 层讲义：Scheduler + BlockManager —— vLLM 的灵魂

> 配套: `python practice/demo_kvcache.py`(分页 KV 缓存可视化模拟器, 已就绪)
> 阅读顺序: `engine/scheduler.py`(92行) → `engine/block_manager.py`(120行)
> 建议: 先跑一遍模拟器, 再看本文, 最后对照真实代码。

---

## 0. 这一层解决什么问题

朴素推理(无调度器)的痛点:
- 每条请求独占 GPU, 排队等; 显存按"最长可能长度"预分配 → 浪费
- 没有缓存复用: 相同前缀的请求各算各的

vLLM 的答案(本层两个类):
- **Scheduler**: 每一轮决定"让谁上 GPU、算多少 token"(连续批处理 + 分块 prefill + 抢占)
- **BlockManager**: 显存按"块"精细管理(分页 KV 缓存 + 前缀缓存 + 引用计数)

## 1. Scheduler 的两个队列

```python
self.waiting: deque[Sequence] = deque()   # 排队等调度的
self.running: deque[Sequence] = deque()   # 正在生成中的
```

**不常见①: `collections.deque`** —— 双向队列, 两端都能 O(1) 增删:
- `popleft()` 取队首 / `appendleft()` 插队首 / `pop()` 取队尾 / `extendleft()` 批量插队首
- 比 list 在"两端操作"场景快得多(第 2 层的 `itertools.count`、本层的 deque 都是"普通开发不常用但引擎里到处是"的标准库)

一个请求的完整生命周期:
```
add_request → waiting ──schedule──▶ running ──postprocess──▶ FINISHED
                 ▲                     │
                 └────── preempt ──────┘
```

## 2. schedule() 阶段一: prefill

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]                  # 只"看"队首, 先不弹出
    remaining = self.max_num_batched_tokens - num_batched_tokens
    if remaining == 0:
        break                              # 本轮 token 预算用完了
    if not seq.block_table:                # 从未调度过 → 要分配块
        num_cached_blocks = self.block_manager.can_allocate(seq)
        if num_cached_blocks == -1:
            break                          # 显存不够 → 后面的人也别看了
        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
    else:                                  # 被抢占过、重新调度的 → 块还在
        num_tokens = seq.num_tokens - seq.num_cached_tokens
    if remaining < num_tokens and scheduled_seqs:
        break                              # ★ chunked prefill 的规则
    if not seq.block_table:
        self.block_manager.allocate(seq, num_cached_blocks)
    seq.num_scheduled_tokens = min(num_tokens, remaining)   # 截断到剩余预算
    num_batched_tokens += seq.num_scheduled_tokens
    if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
        seq.status = SequenceStatus.RUNNING     # 整个 prompt 都排进去了
        self.waiting.popleft()
        self.running.append(seq)
    scheduled_seqs.append(seq)
if scheduled_seqs:
    return scheduled_seqs, True            # True = 这是 prefill 轮
```

**要点逐条拆**:

1. **FIFO 顺序 + 数量上限**: 按入队顺序扫 waiting, 最多 `max_num_seqs`(512)个。

2. **token 预算**: 每轮最多算 `max_num_batched_tokens`(16384)个 token —— 这是吞吐上限, 也是"分批"的单位。

3. **`can_allocate` 返回 -1 = 显存不足**: 直接 break。注意它同时返回"能命中多少前缀缓存块"。

4. **chunked prefill(分块预填充)**: 如果队首序列的 prompt 太长(比如 20000 token > 16384 预算):
   - `remaining < num_tokens` 且还没调度任何序列 → 允许把队首序列**切成几块**, 本轮先算 `remaining` 个
   - `scheduled_seqs` 非空时再遇到超长序列 → break(保证只有第一个序列可以被切)
   - 被切分后 `num_cached + num_scheduled < num_tokens` → **不**移入 running, 留在 waiting, 下轮继续切剩下的
   - 好处: 一条超长请求不会饿死其他请求, GPU 也不会因为一个巨长 prompt 卡一整轮

5. **状态迁移**: 只有整个 prompt 都排完(`num_cached + num_scheduled == num_tokens`)才 WAITING→RUNNING。

## 3. schedule() 阶段二: decode

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()           # 取出队首
    while not self.block_manager.can_append(seq):   # 下一个 token 需要新块但不够?
        if self.running:
            self.preempt(self.running.pop())        # 踢队尾(还没处理的)
        else:
            self.preempt(seq)                       # 只有自己了, 踢自己
            break
    else:                                  # ★ while...else: 没 break 才执行
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)          # 块满则开新块
        scheduled_seqs.append(seq)
assert scheduled_seqs
self.running.extendleft(reversed(scheduled_seqs))   # 处理完的排回队首
return scheduled_seqs, False               # False = decode 轮
```

**要点**:

1. **decode 每序列每轮只算 1 个 token**: `num_scheduled_tokens = 1` —— 这就是第 0 层说的"decode 慢"在代码里的样子。

2. **抢占(Preemption)**: 显存不够时, 把**队尾**(还没轮到处理的)序列踢回 waiting, 释放它的块给当前序列用。这是 vLLM 的"内存压力阀"——总比整个引擎崩掉好。
   - `preempt(running.pop())`: 踢队尾
   - 只有自己在跑且块不够: `preempt(seq)` 踢自己, 等别人完成释放后再回来

3. **不常见②: `while...else`** —— Python 独有语法: 循环体**没有**被 `break` 中断时执行 else。这里 `else` 块 = "can_append 成功了, 正常调度"。用 if/flag 也能写, 但 while...else 更紧凑。

4. **`extendleft(reversed(scheduled_seqs))`**: 把本轮处理过的序列按原顺序插回队首 —— 下一个 decode 轮从同一个序列开始, 形成**轮转(round-robin)**, 公平且不会丢序列。

5. **`assert scheduled_seqs`**: decode 轮必须至少调度一个(否则死循环)。调试期断言, 生产可 `-O` 关掉。

## 4. preempt 与 postprocess

```python
def preempt(self, seq):
    seq.status = SequenceStatus.WAITING
    seq.is_prefill = True                  # 回来时要重新 prefill(重算 KV)
    self.block_manager.deallocate(seq)     # 释放所有块 → 别人能用
    self.waiting.appendleft(seq)           # ★ 插到队首: 比其他等待者优先

def postprocess(self, seqs, token_ids, is_prefill):
    for seq, token_id in zip(seqs, token_ids):
        self.block_manager.hash_blocks(seq)            # ① 新填满的块入前缀索引
        seq.num_cached_tokens += seq.num_scheduled_tokens
        seq.num_scheduled_tokens = 0
        if is_prefill and seq.num_cached_tokens < seq.num_tokens:
            continue                                   # ② chunked prefill 未完, 不追加 token
        seq.append_token(token_id)                     # ③ 把模型吐的新 token 接上
        if (not seq.ignore_eos and token_id == self.eos) \
                or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED       # ④ 结束判定
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

**① 顺序敏感**: `hash_blocks` 必须在 `num_cached_tokens += ...` **之前**调用——它内部用 `num_cached_tokens` 作为"本次新增了哪些块"的起点。

**② chunked prefill 的收尾**: prompt 没算完就不追加 token, 下一轮继续排剩余块。

**③ `append_token`**: 第 2 层讲过的, 把新 token 接进序列历史。

**④ 结束判定**: 碰到 EOS(且没 ignore_eos)或生成满 max_tokens → FINISHED → 释放块 → 移出 running。`running.remove(seq)` 是 O(n) 的, 但序列数不多, 无所谓。

**`preempt` 为什么 `appendleft` 插队首?** 被抢占的序列已经跑了很久(付出了 prefill 成本), 优先让它回来把活干完, 避免"总被打断的饿死"现象(类似操作系统的 aging)。

## 5. BlockManager —— 内存怎么"分页"

`demo_kvcache.py` 已经把核心逻辑可视化过了, 这里只强调真实代码里 3 个对照点:

| 方法 | 作用 | 模拟器里的对应 |
|---|---|---|
| `can_allocate(seq) -> num_cached_blocks 或 -1` | 沿哈希链找可复用的前缀块; -1=显存不够 | 第 1 轮 B "前缀缓存命中 2 块" |
| `allocate(seq, num_cached)` | 命中的块引用计数+1, 其余全新分配 | 块 0/1 的 r=2 |
| `deallocate(seq)` | 引用计数-1, 归零的块回空闲池 | B 被抢占时块 3/6 回收 |
| `can_append/may_append` | decode 需要新块时检查/分配 | "当前块写满, 分配新块" |
| `hash_blocks(seq)` | 新填满的块算哈希, 进前缀索引 | 供后续请求命中 |

**引用计数的两个层级**(真实代码的精细处):
- `Block.ref_count`: 多少个序列共享这块缓存(前缀相同)
- `used_block_ids` / `free_block_ids`: 这块显存当前被谁占用
- `allocate` 里有个细节: 命中缓存的块若**已在使用**(`block_id in self.used_block_ids`)→ 只 `ref_count += 1`; 若在空闲池 → 要 `free_block_ids.remove(block_id)` 把它"领回来"。看 `block_manager.py` 75-92 行。

**哈希前缀链**: `compute_hash(token_ids, prefix)` 把前一块的哈希也混进当前块 —— 这样"块 5 的哈希"隐含着"块 0~4 的内容", 只要两块哈希相等, 前面的链必然相同(内容都对比过)。这是前缀缓存能安全复用的数学基础。

## 6. 自测题

1. decode 阶段为什么用 `while...else` 而不是 `while...if`?
2. `preempt` 为什么 `appendleft` 而不是 `append`? 不这么做会怎样?
3. chunked prefill 时序列留在 waiting, 它的 `is_prefill` 是什么? 什么时候变 False?
4. `extendleft(reversed(scheduled_seqs))` 等价于什么更简单的操作? 为什么要这么写?
5. `can_allocate` 返回 -1 时, 为什么直接 break 而不是看后面的序列?(提示: 显存是全局的)

<details>
<summary>答案</summary>

1. else 块只在"没有 break"时执行, 正好表达"can_append 通过, 正常调度"的语义; 用 if 需要额外标志变量。
2. appendleft 让被抢占的序列插到队首优先回来, 避免它反复被抢占永远完不成(饿死)。
3. is_prefill 保持 True(它在 preempt 里被设回 True); 只有整个 prompt 都排完的那个时刻, decode 阶段才把它设成 False。
4. 等价于 `for s in reversed(scheduled): running.appendleft(s)`: 把本轮处理过的序列按原顺序放回队首, 下轮轮转。
5. 显存不足是全局状态: 队首都分不到, 后面的更分不到(它们只会要更多)。
</details>

## 7. 练习

- 跑 `python practice/demo_kvcache.py`, 把输出与本文 5 节的对照表一一对应。
- 把 `demo_kvcache.py` 的 `NUM_BLOCKS` 改成 6 → 抢占更早更频繁; 改成 9 → 抢占消失。体会"显存压力 → 抢占频率"的关系。
- 读 `scheduler.py` 原码, 说出 `num_tokens`(prefill 分支)在 `seq.block_table` 为空和不为空两种情况下分别代表什么。
