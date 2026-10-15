# -*- coding: utf-8 -*-
"""
第 3 层概念演示: 分页 KV 缓存 (Paged KV Cache) 模拟器
===================================================
纯 Python 标准库, 复刻 nanovllm/engine/block_manager.py + scheduler.py 的核心逻辑:
  - 块分配/释放 (block allocation / deallocation)
  - 引用计数 (ref counting, 多个序列共享同一块缓存)
  - 哈希前缀缓存 (hash-based prefix caching, 相同前缀复用 KV, 省显存省计算)
  - 抢占 (preemption, 显存不够时把队尾序列踢回等待队列, 释放它的块)

用法:  python practice/demo_kvcache.py

对照阅读: nanovllm/engine/block_manager.py + nanovllm/engine/scheduler.py
"""

import hashlib
import sys
from collections import deque

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BLOCK_SIZE = 4      # 每块装几个 token (真实 nano-vllm 是 256)
NUM_BLOCKS = 7      # 总显存块数 (故意调小, 才能看到"内存压力"和抢占)

# ---------------- 模拟 BlockManager (对照 block_manager.py) ----------------

def compute_hash(token_ids, prefix=-1):
    """哈希前缀链: 把'前一块的哈希'也混进来。
    nanovllm 用 xxhash; 这里用 hashlib 代替 (教学用, 逻辑一致)。"""
    h = hashlib.blake2b(digest_size=8)
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(bytes(token_ids))
    return int.from_bytes(h.digest(), "little")


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, h, token_ids):
        self.hash, self.token_ids = h, token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks):
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = {}              # 哈希 -> 块id (前缀缓存索引)
        self.free_block_ids = deque(range(num_blocks))
        self.used_block_ids = set()

    # -- 内部: 从空闲池取/还一块 --
    def _allocate_block(self):
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]      # 清理过期索引
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # -- 对外: 能不能分配? 有多少块能复用缓存? (对照 can_allocate) --
    def can_allocate(self, seq):
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):       # 最后一块留到 allocate 时新建
            token_ids = seq.block(i)
            h = compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break                              # 没命中/内容不同 -> 前缀到此为止
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1                # 已占用 -> 少建一块
        if len(self.free_block_ids) < num_new_blocks:
            return -1                              # 显存不够
        return num_cached_blocks

    # -- 对外: 正式分配 (对照 allocate) --
    def allocate(self, seq, num_cached_blocks):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):         # 命中缓存的块: 引用计数+1
            token_ids = seq.block(i)
            h = compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):   # 其余块全新分配
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * BLOCK_SIZE

    # -- 对外: 释放 (对照 deallocate) --
    def deallocate(self, seq):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # -- decode 阶段: 下一个 token 是否需要新块? 空闲块够吗? (对照 can_append) --
    def can_append(self, seq):
        # len(seq) % BLOCK_SIZE == 1 表示新 token 的 KV 要写入一个新块
        return len(self.free_block_ids) >= (len(seq) % BLOCK_SIZE == 1)

    def may_append(self, seq):
        if len(seq) % BLOCK_SIZE == 1:
            seq.block_table.append(self._allocate_block())

    # -- 给刚填满的块算哈希, 加入前缀缓存索引 (对照 hash_blocks) --
    def hash_blocks(self, seq):
        start = seq.num_cached_tokens // BLOCK_SIZE
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // BLOCK_SIZE
        if start == end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id


# ---------------- 模拟 Sequence (对照 sequence.py) ----------------

class Seq:
    counter = 0

    def __init__(self, name, prompt, max_tokens):
        Seq.counter += 1
        self.seq_id = Seq.counter
        self.name = name
        self.prompt = list(prompt)
        self.token_ids = list(prompt)
        self.max_tokens = max_tokens
        self.status = "WAITING"          # WAITING / RUNNING / FINISHED
        self.is_prefill = True
        self.block_table = []
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0

    def __len__(self):
        return len(self.token_ids)

    @property
    def num_blocks(self):
        return (len(self) + BLOCK_SIZE - 1) // BLOCK_SIZE

    def block(self, i):
        return self.token_ids[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]

    @property
    def num_completion_tokens(self):
        return len(self.token_ids) - len(self.prompt)

    @property
    def is_finished(self):
        return self.status == "FINISHED"


# ---------------- 可视化输出 ----------------

def block_display_content(bm, seqs, block_id):
    """从各序列的 block_table 反推出: 这块缓存里存了哪些 token 的 KV。
    (真实代码里 KV 在 GPU 张量中; 这里用 token 内容示意)"""
    parts = []
    for s in seqs:
        for i, bid in enumerate(s.block_table):
            if bid != block_id:
                continue
            start = i * BLOCK_SIZE
            end = min(start + BLOCK_SIZE, s.num_cached_tokens)
            if end > start:
                parts.append(",".join(str(t) for t in s.token_ids[start:end]))
    uniq = []                                   # 共享块内容去重 (共享由 r= 体现)
    for p in parts:
        if p not in uniq:
            uniq.append(p)
    return " ".join("[" + p + "]" for p in uniq) if uniq else "空"


def print_state(bm, seqs, note=""):
    print(f"  -- 显存 (共{NUM_BLOCKS}块, 每块{BLOCK_SIZE}token) -- {note}")
    for b in bm.blocks:
        mark = "空闲" if b.ref_count == 0 else f"r={b.ref_count}"
        print(f"  [块{b.block_id} {mark:4} {block_display_content(bm, seqs, b.block_id)}]")
    for s in seqs:
        bt = s.block_table if s.block_table else "-"
        print(f"  {s.name}: len={len(s):2} 生成={s.num_completion_tokens}/{s.max_tokens}  "
              f"状态={s.status:8} 块表={bt}  已缓存token={s.num_cached_tokens}")
    print()


def fake_model_next_token(seq):
    """模拟模型前向: 输出一个新 token (当前最后一个 token + 1, 确定性)。"""
    return seq.token_ids[-1] + 1


# ---------------- 主模拟 ----------------

def main():
    print("=" * 72)
    print("场景: 7 块显存(故意紧张), 块大小 4 token, 三条请求:")
    print("  A: prompt [1..9]     (9 token,  最多生成 6 个)")
    print("  B: prompt [1..12]    (12 token, 与 A 共享前缀 [1..8], 最多生成 6 个)")
    print("  C: prompt [100..105] (6 token,  最多生成 6 个)")
    print("=" * 72)
    bm = BlockManager(NUM_BLOCKS)
    A = Seq("A", range(1, 10), 6)
    B = Seq("B", range(1, 13), 6)
    C = Seq("C", range(100, 106), 6)
    waiting = deque([A, B, C])
    running = deque()

    def preempt(seq, reason):
        seq.status = "WAITING"
        seq.is_prefill = True
        bm.deallocate(seq)
        waiting.appendleft(seq)
        print(f"   [抢占] {reason}: 把 {seq.name} 踢回等待队列并释放其块")

    round_no = 0
    while waiting or running:
        round_no += 1
        assert round_no <= 60, "模拟超过 60 轮, 疑似死循环, 请检查场景参数"
        print("#" * 72)
        print(f"第 {round_no} 轮")

        # ---------- 阶段1: prefill (优先调度等待队列) ----------
        did_prefill = False
        if waiting:
            for s in list(waiting):
                num_cached = bm.can_allocate(s)
                if num_cached == -1:
                    print(f"   {s.name}: 显存不足, 继续等待")
                    break
                if num_cached > 0:
                    print(f"   {s.name}: 前缀缓存命中 {num_cached} 块! (省显存省计算)")
                else:
                    print(f"   {s.name}: 无缓存, 全新分配 {s.num_blocks} 块")
                bm.allocate(s, num_cached)
                s.num_scheduled_tokens = len(s) - s.num_cached_tokens
                token_id = fake_model_next_token(s)     # "前向"一次 -> 1 个新 token
                s.is_prefill = False
                s.status = "RUNNING"
                waiting.popleft()
                running.append(s)
                did_prefill = True
                bm.hash_blocks(s)
                s.num_cached_tokens += s.num_scheduled_tokens
                s.token_ids.append(token_id)
                if s.num_completion_tokens >= s.max_tokens:
                    s.status = "FINISHED"
                    bm.deallocate(s)
                    running.remove(s)
                    print(f"   {s.name}: 完成! 释放其块 (共享块引用计数-1)")
            if did_prefill:
                print_state(bm, [A, B, C], "prefill 后")
                continue

        # ---------- 阶段2: decode (每序列生成 1 个 token) ----------
        print(">> decode")
        if not running:
            print("   (没有可运行的序列, 等待新请求)")
        scheduled = []                          # 本轮处理完的序列, 轮末统一重新入队
        while running:
            s = running.popleft()                       # 取出队首处理
            while not bm.can_append(s):                 # 需要新块但不够?
                if running:
                    preempt(running.pop(), f"{s.name} 需要新块但显存已满")   # 踢队尾
                else:
                    preempt(s, f"{s.name} 是唯一运行序列且显存已满")
                    break
            if s.status == "WAITING":                   # 自己被踢了
                continue
            if len(s) % BLOCK_SIZE == 1:
                print(f"   {s.name}: 当前块写满, 分配新块")
            bm.may_append(s)                            # 块满则开新块
            s.num_scheduled_tokens = 1
            s.token_ids.append(fake_model_next_token(s))
            bm.hash_blocks(s)
            s.num_cached_tokens += 1
            if s.num_completion_tokens >= s.max_tokens:
                s.status = "FINISHED"
                bm.deallocate(s)
                print(f"   {s.name}: 生成满 {s.max_tokens} 个, 完成! 释放其块 (共享块引用计数-1)")
            else:
                scheduled.append(s)
                print(f"   {s.name}: 生成新token {s.token_ids[-1]}")
        running.extend(scheduled)                       # 处理完的排到队尾, 下一轮轮转
        print_state(bm, [A, B, C], "decode 后")

    print("=" * 72)
    print("全部完成! 最终结果:")
    for s in [A, B, C]:
        print(f"  {s.name}: prompt={list(s.prompt)}")
        print(f"           生成: {s.token_ids[len(s.prompt):]}")
    print()
    print("关键观察:")
    print("  1. A 和 B 共享前缀 [1..8] 的缓存块, 引用计数=2, 只占一份显存")
    print("  2. B 的 prefill 命中前缀缓存, 少分配块、少计算")
    print("  3. decode 时块写满就开新块; 显存耗尽时抢占: 把队尾/自己踢回等待队列")
    print("  4. 被抢占的序列释放的块立即被其他序列复用(第7轮 C 重新prefill时")
    print("     它的旧块已被 A/B 回收, 所以显示'无缓存' —— 前缀缓存命中不是永久的)")
    print("  5. 序列完成释放块时, 共享块引用计数-1, 不会误伤还在用的序列")


if __name__ == "__main__":
    main()
