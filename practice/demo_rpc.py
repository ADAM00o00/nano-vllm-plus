# -*- coding: utf-8 -*-
"""
第 6 层概念演示: 共享内存 + 事件 的进程间 RPC (复刻 model_runner.py 的通信协议)
============================================================================
纯 Python 标准库 (multiprocessing.shared_memory 在 Windows/Linux 都可用)。

nanovllm 张量并行(TP>1)时的架构:
  rank 0 在主进程; rank 1..n-1 是 spawn 出来的子进程。
  主进程把任务(pickle 序列化)写进共享内存, 用 Event 通知子进程;
  子进程 event.wait() 醒来, 读共享内存, 执行, 再等下一个任务。

本演示: 一个父进程 + 一个子进程, 子进程模拟"worker 跑模型一步"。
  对照代码: model_runner.py 的 write_shm / read_shm / loop / call

用法:  python practice/demo_rpc.py
"""

import pickle
import sys
import time
from multiprocessing import shared_memory
import multiprocessing as mp

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def log(*args):
    """带 flush 的打印, 避免子进程/主进程输出交错乱序。"""
    print(*args, flush=True)


SHM_NAME = "demo_rpc_shm"
SHM_SIZE = 1 << 16   # 64KB


# ---------------- 子进程(worker)侧: 对应 read_shm + loop ----------------

def read_shm(shm, event):
    """从共享内存读一条消息。协议: 前 4 字节 = 长度, 后面是 pickle 数据。"""
    event.wait()                                # 阻塞等待主进程发任务
    n = int.from_bytes(shm.buf[0:4], "little")  # 读长度前缀
    method_name, *args = pickle.loads(shm.buf[4:n + 4])
    event.clear()                               # 立即消费通知, 允许下一次 set
    return method_name, args


def worker_main(shm_name, event):
    """worker 进程主循环: 对应 model_runner.loop()"""
    shm = shared_memory.SharedMemory(name=shm_name)
    log(f"  [worker pid={mp.current_process().pid}] 已连接共享内存, 等待任务...")
    while True:
        method_name, args = read_shm(shm, event)
        if method_name == "exit":
            break
        seqs, is_prefill = args
        if is_prefill:
            # prefill: worker 拿到完整 token 列表, 要算整个 prompt
            n_tokens = sum(len(s) for s in seqs)
            log(f"  [worker] 执行 {method_name}: prefill {len(seqs)} 条序列, "
                f"共 {n_tokens} 个 token (输入: {seqs})")
        else:
            # decode: 经过 __getstate__ 瘦身(第 2 层), 只剩紧凑状态
            st = seqs[0]
            log(f"  [worker] 执行 {method_name}: decode 1 条序列, "
                f"输入 last_token={st['last_token']}, "
                f"已有 token 数={st['num_tokens']} (瘦身后只有 2 个字段!)")
    shm.close()
    log("  [worker] 收到 exit, 退出。")


# ---------------- 主进程侧: 对应 write_shm + call ----------------

def write_shm(shm, events, method_name, *args):
    """把一条消息写进共享内存并通知所有 worker。"""
    data = pickle.dumps([method_name, *args])
    n = len(data)
    shm.buf[0:4] = n.to_bytes(4, "little")     # 长度前缀
    shm.buf[4:n + 4] = data                    # 数据体
    for event in events:                       # nanovllm 里 rank0 有多个 event
        event.set()


def main():
    log("=" * 68)
    log("场景: 1 个主进程(rank 0) + 1 个 worker 进程(rank 1)")
    log("通信协议: 共享内存(pickle 数据) + Event(通知)")
    log("=" * 68)

    ctx = mp.get_context("spawn")              # nanovllm 用 spawn (见第 1 层)
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE)
    event = ctx.Event()
    worker = ctx.Process(target=worker_main, args=(SHM_NAME, event))
    worker.start()

    # --- 任务 1: prefill 阶段, 完整 token 列表 ---
    prefill_seqs = [[1, 2, 3, 4, 5], [10, 11, 12, 13]]
    data1 = pickle.dumps(["run", prefill_seqs, True])
    log(f"  [主进程] 发送 prefill 任务: pickle {len(data1)} 字节")
    write_shm(shm, [event], "run", prefill_seqs, True)
    time.sleep(0.1)      # 演示用延时: 等 worker 消费完事件再发下一个
    # ★ 时序说明: nanovllm 里没有这个 sleep —— rank 0 写完任务后要自己
    #   跑完本卡的模型前向, 那段时间足够 worker 读完消息并 event.clear(),
    #   所以下一个 event.set() 不会丢失。本演示没有计算, 用 sleep 模拟。

    # --- 任务 2: decode 阶段 (瘦身后的状态) ---
    decode_state = {"last_token": 100, "num_tokens": 100}
    data2 = pickle.dumps(["run", [decode_state], False])
    log(f"  [主进程] 发送 decode 任务: pickle {len(data2)} 字节 "
        f"(瘦身后只剩 {len(decode_state)} 个字段)")
    write_shm(shm, [event], "run", [decode_state], False)
    time.sleep(0.1)

    # --- 任务 3: 退出 ---
    log("  [主进程] 发送 exit")
    write_shm(shm, [event], "exit")

    worker.join()                              # 等 worker 退出
    shm.close()
    shm.unlink()                               # 释放共享内存(必须!)
    log("  [主进程] 已 unlink 共享内存, 全部结束。")
    log()
    log("对照 model_runner.py:")
    log("  - write_shm: pickle.dumps → 长度前缀 → shm.buf → event.set()")
    log("  - read_shm : event.wait() → 读长度 → pickle.loads → event.clear()")
    log("  - loop     : 死循环收任务, 收到 'exit' 才退出")
    log("  - 为什么用共享内存: 零拷贝级别的传输, 比 mp.Queue 更快, 且无额外进程")
    log("  - 时序保证: worker 读完后立即 event.clear(), 主进程的下一次 set 不会丢")


if __name__ == "__main__":
    main()
