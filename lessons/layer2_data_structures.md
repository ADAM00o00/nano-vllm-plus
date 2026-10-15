# 第 2 层讲义：请求的数据结构 —— SamplingParams / Config / Sequence

> 配套: `python practice/demo_pickle.py`(演示 Sequence 的 pickle 瘦身)
> 阅读顺序: `sampling_params.py`(8行) → `config.py`(25行) → `engine/sequence.py`(83行)
> 本层是全项目"普通开发不常用语法"最密集的一层, 也是理解后面所有层的基础。

---

## 1. SamplingParams —— dataclass 一行流

```python
from dataclasses import dataclass

@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

**不常见①: `@dataclass`** —— 自动生成 `__init__`/`__repr__`/`__eq__` 等样板代码。你只写字段声明, Python 帮你写构造器。这是现代 Python 定义"纯数据类"的标准方式(替代手写 `__init__(self, temperature=1.0, ...)`)。

**不常见②: `slots=True`** —— dataclass 的进阶选项。默认情况下每个实例都有个 `__dict__` 字典装属性(占内存、访问慢); 开启 slots 后:
- 属性改成"描述符"直接存在对象内存里, **省内存 + 访问更快**
- 代价: 不能再给实例随意挂新属性(如 `obj.xxx = 1` 会报错)
- 本项目所有 dataclass 都用了 slots —— 因为推理引擎要创建几千上万个这样的对象

**不常见③: `__post_init__`** —— dataclass 在 `__init__` 末尾自动调用的钩子, 用来做**字段校验**。这里校验 temperature > 1e-10, 禁止贪心解码(temperature=0 会除零)。

**`ignore_eos`**: 生成到 EOS 也不停 —— bench.py 里用它让所有请求生成满 max_tokens, 便于公平测吞吐。

## 2. Config —— 引擎的全局配置

```python
@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384   # 一轮最多算多少 token (吞吐上限)
    max_num_seqs: int = 512               # 一轮最多并行多少序列
    max_model_len: int = 4096             # 上下文长度上限
    gpu_memory_utilization: float = 0.9   # KV 缓存最多占显存比例
    tensor_parallel_size: int = 1         # 用几张卡
    enforce_eager: bool = False           # True=禁用 CUDA Graph (第5层)
    hf_config: AutoConfig | None = None   # 模型的 HuggingFace 配置
    eos: int = -1                         # 结束符 id (构造后由 tokenizer 填)
    kvcache_block_size: int = 256         # KV 缓存块大小 (页大小)
    num_kvcache_blocks: int = -1          # 总块数 (构造后由显存算出)

    def __post_init__(self):
        assert os.path.isdir(self.model)                  # 模型目录必须存在
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)   # 读模型配置
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
```

**关键字段的直觉**:
- `max_num_batched_tokens=16384`: 一次 prefill 最多喂 16384 个 token。prompt 总长超过它就要 **chunked prefill**(第 3 层)。
- `gpu_memory_utilization=0.9`: 第 5 层分配 KV 缓存时, 按"总显存 × 0.9 − 模型占的"来算块数。
- `AutoConfig.from_pretrained`: 从模型目录的 config.json 读结构(层数、头数等), **不加载权重**, 很快。

**`str | None` 类型联合** 是 Python 3.10+ 语法, 等价于旧版的 `Optional[str]`。

## 3. Sequence —— 一个请求的全部状态

### 3.1 状态机与类属性

```python
class SequenceStatus(Enum):
    WAITING = auto()    # 排队等调度
    RUNNING = auto()    # 正在生成
    FINISHED = auto()   # 完成(EOS 或达到 max_tokens)

class Sequence:
    block_size = 256            # 类属性: 所有实例共享 (第1层讲过: 由引擎统一设置)
    counter = count()           # 类属性: 无限递增计数器

    def __init__(self, token_ids: list[int], sampling_params=SamplingParams()):
        self.seq_id = next(Sequence.counter)     # 全局唯一 id: 0,1,2,...
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)         # 浅拷贝, 防止外部篡改
        self.last_token = token_ids[-1]          # decode 时只喂这一个
        self.num_tokens = len(self.token_ids)    # 已生成的总 token 数
        self.num_prompt_tokens = len(token_ids)  # prompt 的 token 数
        self.num_cached_tokens = 0               # 已有多少 token 的 KV 在缓存里
        self.num_scheduled_tokens = 0            # 本轮要算多少个
        self.is_prefill = True                   # 下一轮是 prefill 还是 decode
        self.block_table = []                    # 分页: 这个序列占用了哪些块
```

**不常见④: `Enum` + `auto()`** —— 枚举。`auto()` 自动从 1 开始编号(WAITING=1, RUNNING=2, FINISHED=3), 不用手写数字。比用字符串常量更安全(拼错就报错)也更规范。

**不常见⑤: `itertools.count()`** —— 一个"无底洞"计数器: `next(counter)` 依次返回 0,1,2,... 永不重复。这里用来生成**全局唯一**的 seq_id。注意它是**类属性**, 所以所有 Sequence 实例共享同一个计数器 —— 这正是结果按 seq_id 排序后能对应回请求顺序的原因。

**`copy(token_ids)` 浅拷贝**: 只复制列表本身, 里面的元素(整数)是共享的。防止外部代码 `prompt.append(...)` 时意外改到序列内部状态。

### 3.2 一系列 property(把方法伪装成属性)

```python
@property
def is_finished(self):
    return self.status == SequenceStatus.FINISHED

@property
def num_completion_tokens(self):
    return self.num_tokens - self.num_prompt_tokens   # 已生成的(非 prompt 的)token 数

@property
def prompt_token_ids(self):
    return self.token_ids[:self.num_prompt_tokens]    # 切片: 前 N 个是 prompt

@property
def completion_token_ids(self):
    return self.token_ids[self.num_prompt_tokens:]    # 剩下的都是生成结果

@property
def num_blocks(self):
    return (self.num_tokens + self.block_size - 1) // self.block_size   # 向上取整除法

@property
def last_block_num_tokens(self):
    return self.num_tokens - (self.num_blocks - 1) * self.block_size    # 最后一块里有几个

def __len__(self):
    return self.num_tokens          # 注意! 不是 len(self.token_ids)

def __getitem__(self, key):
    return self.token_ids[key]      # seq[i] 等价于 seq.token_ids[i]

def append_token(self, token_id: int):
    self.token_ids.append(token_id)
    self.last_token = token_id
    self.num_tokens += 1
```

**不常见⑥: `@property`** —— 把方法变成属性访问。`seq.is_finished` 后面没有括号, 但背后是方法调用。好处: 字段变化时计算逻辑不用改调用方。

**`num_blocks` 的向上取整除法**: `(N + block_size - 1) // block_size` 是"ceil(N / block_size)"的经典写法, 比 `math.ceil` 快且不引入 import。9 个 token, 每块 4 → (9+3)//4 = 3 块。

**`__len__` 返回 `num_tokens` 而不是列表长度** —— 这很关键! 因为 pickle 瘦身后 decode 阶段的 `token_ids` 是空列表(见 3.3), 但序列"逻辑长度"必须保留。`len(seq)` 因此永远读计数, 不读列表。

### 3.3 `__getstate__` / `__setstate__` —— pickle 协议钩子(本层重点!)

**背景**: 张量并行时(第 6 层), 主进程要把 Sequence 序列化(pickle)后通过共享内存发给工作进程。

```python
def __getstate__(self):
    # prefill 阶段: 打包全部 token; decode 阶段: 只打包最后一个 token
    last_state = self.last_token if not self.is_prefill else self.token_ids
    return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
            self.num_scheduled_tokens, self.block_table, last_state)

def __setstate__(self, state):
    (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
     self.num_scheduled_tokens, self.block_table, last_state) = state
    if isinstance(last_state, list):        # prefill: 恢复整串 token
        self.token_ids = last_state
        self.last_token = self.token_ids[-1]
    else:                                   # decode: 只恢复最后一个 token
        self.token_ids = []
        self.last_token = last_state
```

**不常见⑦: pickle 协议钩子**
- 默认情况下, `pickle.dumps(obj)` 会序列化对象的 `__dict__`(所有属性)。
- 但类可以定义 `__getstate__`(打包成什么)和 `__setstate__`(怎么还原), 完全接管序列化过程。
- 这是"普通开发几乎用不到、但跨进程/跨机器传输对象时必须会"的机制。

**这里的设计意图**:
1. **decode 阶段只传 1 个 token**: 工作进程跑模型只需要 `last_token`(输入) + 各种计数和 block_table(KV 位置)。整串历史在工作进程侧毫无用处 —— 省通信量(演示脚本实测省 79%)。
2. **prefill 阶段传全部 token**: 工作进程要算整个 prompt, 必须拿到完整 token 列表。
3. **哪些属性被故意省略了**: `seq_id / status / is_prefill / temperature / max_tokens / ignore_eos` 都不在 state 里。为什么安全? 看 `model_runner.py`:
   - `prepare_sample` 读 `seq.temperature`, 但它只在 **rank 0**(主进程)执行(`if self.rank == 0`)
   - `status` 的流转只在主进程的 scheduler 里发生
   - 工作进程只读 `block_table / num_* 计数 / last_token / token_ids`
4. **`__len__` 必须读计数**: 因为还原后的 `token_ids=[]`, 若 `len` 读列表长度, decode 阶段一切都会崩 —— 所以 3.2 里 `__len__` 返回 `num_tokens` 是配套设计。

**is_prefill 判断的妙处**: `__getstate__` 里用 `is_prefill` 决定打包什么, 反序列化端用 `isinstance(last_state, list)` 判断是哪一种 —— 不需要单独传一个标志位。这种"用类型编码状态"的技巧值得记住。

## 4. 数据流串起来

```
LLMEngine.add_request(prompt, sp)
  ├─ prompt: str → tokenizer.encode → list[int]
  └─ Sequence(prompt, sp)   ← seq_id=next(counter), status=WAITING
       │
       ▼
scheduler.waiting ──schedule()──▶ 分配块(block_table) ──▶ status=RUNNING
       │
       ▼
model_runner.run(seqs, is_prefill)
  ├─ prefill: 读 seq.token_ids / num_cached_tokens / num_scheduled_tokens
  └─ decode : 读 seq.last_token / len(seq) / block_table
       │
       ▼
scheduler.postprocess: seq.append_token(新token) / num_cached_tokens += ...
  └─ EOS 或生成满 → status=FINISHED → deallocate 块
```

## 5. 自测题

1. `slots=True` 解决了什么问题? 代价是什么?
2. `Sequence.counter` 为什么必须是类属性而不是实例属性?
3. `num_blocks` 的 `(n + block_size - 1) // block_size` 在算什么? 举个数。
4. 为什么 `__len__` 返回 `num_tokens` 而不是 `len(token_ids)`?(提示: 想 pickle 瘦身)
5. decode 阶段 pickle 只传 last_token, 为什么不会丢信息? 工作进程还需要什么?

<details>
<summary>答案</summary>

1. 省内存 + 属性访问更快(不用 `__dict__`); 代价是不能动态挂新属性。
2. 所有实例必须共享同一个计数器, seq_id 才能全局唯一; 若放实例上, 每个新对象都从 0 开始。
3. 向上取整除法: 9 个 token、块大小 4 → 需要 3 块(最后一块只有 1 个)。
4. 因为 decode 阶段还原出来的 Sequence 的 token_ids 是空列表, 长度信息全靠 num_tokens 保留。
5. 工作进程跑 decode 只需要: last_token(输入)、num_tokens(位置)、num_cached_tokens/scheduled(调度)、block_table(KV 落点)。温度只在 rank 0 用, status 只在主进程 scheduler 用。
</details>

## 6. 练习

- 运行 `python practice/demo_pickle.py`, 观察 prefill/decode 两阶段的 pickle 体积差。
- 改一改: 在 `demo_pickle.py` 里给 Sequence 加一个 `temperature` 属性(不放进 `__getstate__`), 序列化后访问它, 看 AttributeError —— 体会"什么该传、什么不该传"。
- 把 `sampling_params.py` 的 `SamplingParams(temperature=0)` 跑一下, 看断言怎么拦下贪心解码。
