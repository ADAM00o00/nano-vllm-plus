# 第 1 层讲义：example.py + LLMEngine —— 看全局流程

> 配套: 先读 `example.py`(33 行) → `nanovllm/llm.py`(5 行) → `nanovllm/engine/llm_engine.py`(90 行)
> 本讲义是对话讲解的完整文字版。讲完这层, 你应该能画出"一次 generate() 的数据流图"。

---

## 1. 阅读顺序为什么是 example → llm → llm_engine

- `example.py`: 用户视角, 看"怎么用"
- `nanovllm/llm.py`: 只有 5 行, `class LLM(LLMEngine): pass` —— 一个空壳子类, 纯粹为了 API 名好看 (`LLM(...)` 而不是 `LLMEngine(...)`)
- `nanovllm/engine/llm_engine.py`: 真正的大脑, 90 行

## 2. example.py 逐段

```python
path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")   # ~ 展开成家目录
tokenizer = AutoTokenizer.from_pretrained(path)          # ① 加载分词器(文本↔token id)
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)  # ② 创建引擎(会加载模型!)

sampling_params = SamplingParams(temperature=0.6, max_tokens=256)  # ③ 采样参数
```

**④ 聊天模板** —— 这里有个"普通开发不常用"的东西:

```python
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],   # 把纯文本包成 OpenAI 风格的对话格式
        tokenize=False,                          # 只要文本, 先不转 token
        add_generation_prompt=True,              # 末尾加上模型开始作答的标记
    )
    for prompt in prompts
]
```

**为什么需要聊天模板?** 指令微调过的模型只有在特定格式下才听话。`"introduce yourself"` 会变成类似:
`<|im_start|>system\nYou are Qwen...<|im_end|>\n<|im_start|>user\nintroduce yourself<|im_end|>\n<|im_start|>assistant\n`
模型看到这个格式才知道"该我回答了"。**这是 chat 模型和 base 模型的本质区别**。

**⑤ `zip` 解包** —— `for prompt, output in zip(prompts, outputs)`:
`zip` 把两个列表逐对拼起来。这是 Python 里极其常用的写法, 如果还不熟, 现在记牢:
`zip([a,b], [1,2])` → `[(a,1), (b,2)]`。

## 3. llm_engine.py 逐段 —— 构造函数

```python
config_fields = {field.name for field in fields(Config)}
config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
config = Config(model, **config_kwargs)
```

**不常见方法①: `dataclasses.fields()`**
`fields(Config)` 返回 Config 所有字段的元数据列表, 这里取它们的名字做成集合。
**目的**: 用户 `LLM(path, enforce_eager=True, ...)` 传入的 kwargs 里, 只挑 Config 认识的参数, 其他的悄悄丢掉。
（vLLM 里这是用 `**kwargs` 透传 + Config 校验实现的, 这里用 dataclass 反射, 更轻量。）

```python
Sequence.block_size = config.kvcache_block_size   # 不常见: 给"类属性"赋值!
```
`block_size` 是 `Sequence` 的类属性(所有实例共享), 在这里统一设置, 之后所有 Sequence 实例都用它。
**类属性 vs 实例属性**: 类属性存在类上(`Sequence.block_size`), 实例属性存在对象上(`self.xxx`)。这里刻意用类属性, 因为"块大小"是全局统一的配置, 没必要每个对象存一份。

```python
self.ps = []      # 子进程列表 (张量并行 >1 时才有)
self.events = []
ctx = mp.get_context("spawn")     # 不常见②
for i in range(1, config.tensor_parallel_size):   # 只给 rank 1..n-1 开进程
    event = ctx.Event()                           # 跨进程"信号灯"
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
    self.ps.append(process)
    self.events.append(event)
self.model_runner = ModelRunner(config, 0, self.events)   # rank 0 就在主进程里
```

**不常见方法②: `mp.get_context("spawn")`**
Python 多进程有三种启动方式:
- `fork`: 子进程复制父进程内存(快, 但容易出幽灵问题) —— Linux 默认
- `spawn`: 子进程重新 import 模块再执行(慢, 但干净、跨平台安全)
- `forkserver`: 折中

这里用 **spawn**, 因为 ModelRunner 里要初始化 NCCL/GPU, fork 一个已经初始化过 CUDA 的进程会崩。这是 PyTorch 多进程的经典坑, vLLM 官方也踩过。

**张量并行 (TP) 架构预览**(第 6 层细讲):
- 每张卡一个 rank, rank 0 在主进程, rank 1..n-1 是子进程
- 模型权重按 rank 切分, 各算各的, 需要时用 NCCL 通信(`all_reduce` 等)
- `tensor_parallel_size=1` 时这个循环直接跳过, 只有主进程的 ModelRunner

```python
self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
config.eos = self.tokenizer.eos_token_id    # 从 tokenizer 拿结束符 id
self.scheduler = Scheduler(config)
atexit.register(self.exit)                  # 不常见③: 进程退出时自动调用 exit()
```

**不常见方法③: `atexit.register(self.exit)`**
注册一个"进程正常退出时自动执行"的钩子。这里用来保证 `exit()` 一定被调用——
`exit()` 负责让子进程退出(`p.join()`)和销毁 NCCL, 否则程序退出时 GPU 资源可能悬挂。

## 4. llm_engine.py 逐段 —— 三个小方法

```python
def add_request(self, prompt: str | list[int], sampling_params):
    if isinstance(prompt, str):          # 字符串 → 编码成 token id 列表
        prompt = self.tokenizer.encode(prompt)
    seq = Sequence(prompt, sampling_params)   # 包成 Sequence 对象
    self.scheduler.add(seq)                   # 丢进调度器的 waiting 队列
```

**不常见语法: `str | list[int]`** —— Python 3.10+ 的类型联合注解, 意思是"字符串或整数列表"。`isinstance` 做运行时判断。这里允许用户直接传 token id 列表(bench.py 就是直接传的, 跳过 tokenizer)。

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()          # ① 调度器选人
    num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
    token_ids = self.model_runner.call("run", seqs, is_prefill)  # ② 跑模型+采样
    self.scheduler.postprocess(seqs, token_ids, is_prefill)      # ③ 收尾: 追加token/判结束
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

**看懂 `num_tokens` 的符号约定**(这是工程上的小聪明):
- prefill: 正数 = 本轮一共算的 token 数
- decode: 负数 = -序列数(每序列只算 1 个 token)
- 后面 `generate()` 用正负号区分"这是 prefill 吞吐还是 decode 吞吐"

**`self.model_runner.call("run", seqs, is_prefill)`** —— 方法名字符串 + 参数, 动态调用 `ModelRunner.run(...)`。rank 0 会先把任务广播给其他 rank 的进程(第 6 层细讲), 自己是主进程时直接执行。

```python
def is_finished(self):
    return self.scheduler.is_finished()   # waiting 和 running 都空 = 全部完成
```

## 5. generate() —— 主循环(重点!)

```python
def generate(self, prompts, sampling_params, use_tqdm=True):
    pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
    if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len(prompts)   # 单个参数 → 复制成列表
    for prompt, sp in zip(prompts, sampling_params):
        self.add_request(prompt, sp)                          # ① 全部请求入队

    outputs = {}
    prefill_throughput = decode_throughput = 0.
    while not self.is_finished():                             # ② 循环调度直到全部完成
        t = perf_counter()                                    # 不常见④: 高精度计时
        output, num_tokens = self.step()
        if num_tokens > 0:
            prefill_throughput = num_tokens / (perf_counter() - t)
        else:
            decode_throughput = -num_tokens / (perf_counter() - t)
        pbar.set_postfix({                                    # 不常见⑤: 进度条附加指标
            "Prefill": f"{int(prefill_throughput)}tok/s",
            "Decode": f"{int(decode_throughput)}tok/s",
        })
        for seq_id, token_ids in output:                      # ③ 收集完成的序列
            outputs[seq_id] = token_ids
            pbar.update(1)

    pbar.close()
    outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]  # ④ 按 seq_id 排序
    outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
               for token_ids in outputs]                              # ⑤ 解码回文本
    return outputs
```

**不常见方法④: `perf_counter()`** —— `time.perf_counter()` 是 Python 里精度最高的计时器(测性能专用), 比 `time.time()` 更适合测"这一段代码跑了多久"。

**不常见方法⑤: `tqdm.set_postfix(...)`** —— tqdm 是进度条库, `set_postfix` 可以在进度条**右边**实时显示自定义指标。这里显示实时 prefill/decode 吞吐, 让你肉眼看到引擎在干活。

**核心循环的本质**:
```
while 还有没完成的请求:
    调度一批 → GPU 前向 → 采样出新 token → 检查是否结束
```
这就是"连续批处理(continuous batching)"的雏形——每次循环都重新决定"下一轮跑谁"。

**④ 为什么按 seq_id 排序?**
`Sequence.counter = count()` 从 0 开始递增分配 seq_id, 先入队的 id 小。
所以 `sorted(outputs.keys())` 恰好把结果**按请求的原始顺序**排好, 和你传入的 prompts 一一对应。
(小细节: `count()` 是无限计数器, 全局唯一, 不会重复 —— 这就是 `itertools.count` 的用处, 第 2 层细讲。)

## 6. 完整数据流图(背下来)

```
prompts(字符串列表)
  │ tokenizer.encode / apply_chat_template
  ▼
token id 列表 ──▶ Sequence 对象 ──▶ Scheduler.waiting 队列
                                        │ 每轮 step():
                                        ▼
                        Scheduler.schedule() → 选一批 seqs (prefill/decode)
                                        │
                                        ▼
                        ModelRunner.run(seqs, is_prefill)
                          ├─ prepare_prefill / prepare_decode (拼 GPU 张量)
                          ├─ Qwen3 前向 → logits
                          └─ Sampler → 每个 seq 一个新 token id
                                        │
                                        ▼
                        Scheduler.postprocess: 追加 token、判 EOS/max_tokens、
                                               更新缓存计数、标记 FINISHED
                                        │
                                        ▼
                        is_finished? ──否──▶ 下一轮
                           │是
                           ▼
                        tokenizer.decode → 文本列表返回
```

## 7. 自测题

1. `generate()` 的 while 循环里, 一次 `step()` 是只处理一个请求吗?(提示: 看 `schedule()` 返回的 `seqs`)
2. `num_tokens` 为什么 decode 时是负数? 这个符号约定解决了什么问题?
3. 如果不按 seq_id 排序直接返回 outputs, 会发生什么?
4. `enforce_eager=True` 和默认 `False` 在 example.py 里分别是什么效果?(答案在第 5 层)
5. 为什么张量并行必须用 `spawn` 而不是 `fork`?

<details>
<summary>答案</summary>

1. 不是。一次 `step()` 调度**一批**序列(最多 `max_num_seqs` 个), GPU 一次前向算完这一批 —— 这就是批处理提升吞吐的核心。
2. decode 每序列只生成 1 个 token, 用负数标记"这是 decode", 正数标记 prefill 的 token 总数, `generate()` 据此分别统计两种吞吐。
3. 结果顺序会和请求顺序错乱(dict 遍历顺序虽是插入序, 但完成时间不同, 用 seq_id 排序保证稳定)。
4. `enforce_eager=True` 禁用 CUDA Graph(第 5 层), 慢一些但兼容性好、易调试。
5. fork 会把已经初始化 CUDA 的进程复制一份, 子进程里的 GPU 状态是坏的; spawn 重新启动干净的进程。
</details>

## 8. 练习

- 把 `llm_engine.py` 里 `generate()` 的 while 循环抄一遍, 去掉 tqdm, 自己加打印, 观察每轮 `step()` 返回的 `(outputs, num_tokens)` 长什么样(纯读代码即可, 不用真跑)。
- 回答自测 5 题, 发给我检查。
