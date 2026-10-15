# -*- coding: utf-8 -*-
"""
第 2 层概念演示: Sequence 的 pickle 协议瘦身 (__getstate__ / __setstate__)
=======================================================================
纯 Python 标准库, 复刻 nanovllm/engine/sequence.py 的序列化设计。

为什么需要它?
  张量并行(TP>1)时, 主进程要把 Sequence 对象 pickled 后通过共享内存
  发给其他 rank 的进程 (model_runner.py 的 write_shm / read_shm)。
  但 decode 阶段主进程其实只需要"最后一个 token"就能让工作进程跑模型,
  把整串 token 都发过去纯属浪费。所以 Sequence 自定义了序列化:
    - prefill 阶段: 传全部 token_ids (工作进程要算整个 prompt)
    - decode 阶段: 只传最后一个 token (工作进程只算这一个位置)

用法:  python practice/demo_pickle.py
对照阅读: nanovllm/engine/sequence.py
"""

import copy
import pickle
import sys
from enum import Enum, auto
from itertools import count

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BLOCK_SIZE = 4  # 演示用小块; 真实 nano-vllm 是 256


class SequenceStatus(Enum):
    WAITING = auto()   # auto() 自动编号: WAITING=1, RUNNING=2, FINISHED=3
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    # ---- 类属性: 所有实例共享, 不占每实例的内存 ----
    block_size = BLOCK_SIZE
    counter = count()            # itertools.count: 无限递增计数器, 全局唯一 id

    def __init__(self, token_ids, is_prefill=True):
        self.seq_id = next(Sequence.counter)          # 0, 1, 2, ...
        self.status = SequenceStatus.WAITING
        self.token_ids = copy.copy(token_ids)         # 浅拷贝: 防止外部改原列表
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = is_prefill
        self.block_table = []                          # 分页: 本序列用了哪些块

    def __len__(self):
        return self.num_tokens                         # 注意: 不是 len(token_ids)!

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def block(self, i):
        return self.token_ids[i * self.block_size:(i + 1) * self.block_size]

    # ---- pickle 协议钩子 ----
    def __getstate__(self):
        # prefill 阶段: 打包全部 token; decode 阶段: 只打包最后一个 token
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
                self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
         self.num_scheduled_tokens, self.block_table, last_state) = state
        if isinstance(last_state, list):               # prefill: 恢复整个 token 列表
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:                                          # decode: token_ids 置空
            self.token_ids = []
            self.last_token = last_state


def main():
    print("=" * 66)
    print("演示 1: 类属性与 itertools.count")
    print("=" * 66)
    s1 = Sequence([1, 2, 3, 4])
    s2 = Sequence([5, 6, 7, 8])
    print(f"  Sequence.block_size 是类属性: {Sequence.block_size} "
          f"(所有实例共享同一份)")
    print(f"  seq_id 用 itertools.count 分配, 全局唯一不重复: "
          f"{s1.seq_id}, {s2.seq_id}")
    print(f"  len(s1) = {len(s1)}  (注意: __len__ 返回 num_tokens 而不是列表长度)")
    print()

    print("=" * 66)
    print("演示 2: pickle 体积对比 —— prefill 阶段 vs decode 阶段")
    print("=" * 66)
    prompt = list(range(1, 101))          # 100 个 token 的 prompt

    prefill_seq = Sequence(prompt, is_prefill=True)
    prefill_seq.num_cached_tokens = 100   # 假装 prefill 已算完
    prefill_bytes = pickle.dumps(prefill_seq)

    decode_seq = Sequence(prompt, is_prefill=False)
    decode_seq.num_cached_tokens = 100
    decode_bytes = pickle.dumps(decode_seq)

    print(f"  prefill 阶段序列: {len(prefill_bytes):>5} 字节  "
          f"(要传全部 {len(prompt)} 个 token, 因为工作进程要算整个 prompt)")
    print(f"  decode  阶段序列: {len(decode_bytes):>5} 字节  "
          f"(只传最后 1 个 token {decode_seq.last_token})")
    print(f"  节省了 {(1 - len(decode_bytes) / len(prefill_bytes)) * 100:.0f}% 的通信量")
    print()

    print("=" * 66)
    print("演示 3: 反序列化 (__setstate__) 还原了什么")
    print("=" * 66)
    restored = pickle.loads(decode_bytes)      # 模拟工作进程收到序列
    print(f"  num_tokens={restored.num_tokens}, num_prompt_tokens={restored.num_prompt_tokens}")
    print(f"  last_token={restored.last_token}")
    print(f"  token_ids={restored.token_ids}  (decode 阶段是空的, 省内存)")
    print(f"  num_blocks={restored.num_blocks}  (由 num_tokens 算出, 不需要 token 列表)")
    print(f"  block_table={restored.block_table}")
    print()
    print("  注意: 以下属性在反序列化后【不存在】:")
    print("    seq_id / status / is_prefill / temperature / max_tokens")
    print("  原因: 工作进程只负责跑模型(读 block_table/计数/last_token),")
    print("  这些调度用的属性只在主进程有意义 —— 序列化时被刻意省略了。")
    print("  想验证? 取消注释下一行会抛 AttributeError:")
    # print(restored.status)
    print()

    print("=" * 66)
    print("演示 4: Enum / copy / property 小抄")
    print("=" * 66)
    print(f"  SequenceStatus.RUNNING 的值: {SequenceStatus.RUNNING.value}  "
          f"(auto() 自动编号)")
    print(f"  copy.copy 是浅拷贝: 只复制外层列表, 元素仍是原对象")
    inner = [1, 2]
    a = [inner]
    b = copy.copy(a)
    print(f"    a is b -> {a is b}, a[0] is b[0] -> {a[0] is b[0]}  "
          f"(内层共享, 外层独立)")
    print()

    print("=" * 66)
    print("演示 5: 完整往返 —— 主进程打包 → 模拟跨进程 → 工作进程使用")
    print("=" * 66)
    seq = Sequence(prompt, is_prefill=False)
    seq.num_cached_tokens = 96
    seq.block_table = [0, 1, 2, 3]
    data = pickle.dumps(seq)                 # write_shm 干的事
    worker_seq = pickle.loads(data)          # read_shm 干的事
    # 工作进程 prepare_decode 里用到的字段:
    print(f"  input token      : {worker_seq.last_token}")
    print(f"  position         : {len(worker_seq) - 1}")
    print(f"  slot (块内位置)   : 块{worker_seq.block_table[-1]} 内偏移 "
          f"{worker_seq.num_tokens - (worker_seq.num_blocks - 1) * BLOCK_SIZE - 1}")
    print()
    print("结论: __getstate__/__setstate__ 让'跨进程传 Sequence'又快又省,")
    print("decode 阶段只传 1 个 token 而不是整串历史。")


if __name__ == "__main__":
    main()
