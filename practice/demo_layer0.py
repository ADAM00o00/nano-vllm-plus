# -*- coding: utf-8 -*-
"""
第 0 层概念演示：token / logits / softmax / temperature / 自回归生成
===============================================================
纯 Python 标准库实现，不需要 torch / transformers / GPU，任何机器都能跑。

用法:  python practice/demo_layer0.py

演示内容:
  1. 玩具词表（真实模型词表约 15 万个 token，这里用 8 个汉字演示）
  2. softmax: 把 logits 变成概率分布
  3. temperature 对概率分布形状的影响（0.1 vs 1.0 vs 1.5）
  4. 两种采样方式对比:
     a) 朴素按概率采样
     b) Gumbel-max 技巧 —— nanovllm/layers/sampler.py 用的就是它
  5. 自回归生成循环: 一个"玩具模型"逐字写句子（decode 阶段）
"""

import math
import random
import sys
from collections import Counter

# 让中文在 Windows 控制台也能正常输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------- 1. 玩具词表 ----------
VOCAB = ["你", "好", "，", "世", "界", "！", "？", "。"]
VOCAB_SIZE = len(VOCAB)
print("=" * 62)
print("演示 1: token 与词表")
print("=" * 62)
print(f"玩具词表: {VOCAB}")
print(f"每个 token 对应一个整数 id: {{token: id}} = "
      f"{ {t: i for i, t in enumerate(VOCAB)} }")
print(f"真实模型 (Qwen3-0.6B) 的词表大小约 151,936, 这里是 {VOCAB_SIZE} 个。")
print()


# ---------- 2. softmax + temperature ----------
def softmax(logits, temperature=1.0):
    """logits -> 概率分布。

    nanovllm/layers/sampler.py 第 9 行做的事:
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
    这里用纯 Python 复刻。
    """
    scaled = [l / temperature for l in logits]
    m = max(scaled)  # 数值稳定性: 减去最大值, 防止 e^x 溢出(不常见但重要的细节)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


# 一个"模型给出的原始打分"（真实中来自 lm_head 的输出）
LOGITS = [2.0, 1.5, 0.8, 0.2, -0.5, -1.0, -1.5, -2.0]

print("=" * 62)
print("演示 2: logits -> softmax -> 概率")
print("=" * 62)
probs_t1 = softmax(LOGITS, 1.0)
for i, (t, p) in enumerate(zip(VOCAB, probs_t1)):
    bar = "#" * int(p * 60)
    print(f"  token {t!r:4}  logit={LOGITS[i]:+5.1f}  p={p:.3f}  {bar}")
print()


# ---------- 3. temperature 的影响 ----------
def entropy(probs):
    """信息熵: 衡量分布有多'平均'。越大越随机。"""
    return -sum(p * math.log(p) for p in probs if p > 0)


print("=" * 62)
print("演示 3: temperature 改变分布的'尖锐程度'")
print("=" * 62)
for temp in (0.1, 0.6, 1.0, 1.5):
    p = softmax(LOGITS, temp)
    top = VOCAB[max(range(VOCAB_SIZE), key=lambda i: p[i])]
    print(f"  temperature={temp:>4}  最高概率token={top!r}  "
          f"p(最高)={max(p):.3f}  熵={entropy(p):.3f}")
print("  结论: 温度越小分布越尖(几乎总选最高分, 输出机械);")
print("        温度越大分布越平(经常爆冷, 输出天马行空)。")
print()


# ---------- 4. 两种采样方式 ----------
def sample_direct(probs, rng):
    """朴素采样: 按概率分布随机抽一个下标。"""
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r < acc:
            return i
    return len(probs) - 1


def sample_gumbel(probs, rng):
    """Gumbel-max 技巧 (nanovllm sampler.py 第 11 行的等价物)。

    sampler.py 原文:
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1)
                                   .clamp_min_(1e-10)).argmax(dim=-1)
    即: 每个概率除以一个指数分布噪声, 再取 argmax。

    数学原理:
      设 E ~ Exp(1), 则 -log(E) 服从 Gumbel(0,1) 分布。
      argmax_i (p_i / E_i)  ≡  argmax_i (log p_i - log E_i)
                            ≡  argmax_i (log p_i + g_i)   (Gumbel-max)
      而 Gumbel-max 已被证明等价于按概率分布采样。
      好处: 全程只有张量运算(没有 Python 循环), 便于 torch.compile / CUDA Graph。
    """
    noise = [rng.expovariate(1.0) for _ in probs]      # E ~ Exp(1)
    noisy = [p / max(n, 1e-10) for p, n in zip(probs, noise)]
    return max(range(len(noisy)), key=lambda i: noisy[i])


print("=" * 62)
print("演示 4: 两种采样方式, 各自跑 20000 次, 对比频率分布")
print("=" * 62)
N_TRIALS = 20000
rng = random.Random(42)
freq_direct = Counter(sample_direct(probs_t1, rng) for _ in range(N_TRIALS))
rng = random.Random(42)  # 同样的随机种子, 公平对比
freq_gumbel = Counter(sample_gumbel(probs_t1, rng) for _ in range(N_TRIALS))
print(f"  {'token':6} {'理论概率':>8} {'朴素采样':>10} {'Gumbel':>10}")
for i, t in enumerate(VOCAB):
    print(f"  {t!r:6} {probs_t1[i]:8.3f} "
          f"{freq_direct[i] / N_TRIALS:10.3f} {freq_gumbel[i] / N_TRIALS:10.3f}")
print("  两种方法的频率都接近理论概率 => Gumbel 技巧和朴素采样等价。")
print()


# ---------- 5. 自回归生成循环 ----------
# 一个"玩具模型": 下一个 token 的概率只取决于当前最后一个 token (一阶马尔可夫链)。
# 真实模型比这复杂得多, 但"一次只生成一个 token, 新 token 依赖旧 token"的本质一样。
# key: 当前token, value: 下一个token的logits
TOY_MODEL = {
    "你": [2.0, 1.8, 0.5, 0.3, 0.0, -0.5, -1.0, -1.5],   # 你 -> 好 / 你
    "好": [0.2, 1.0, 1.5, 0.8, 0.5, 0.0, -0.5, -1.0],   # 好 -> ，/ 好 / 世
    "，": [0.0, 0.2, 0.3, 2.0, 1.8, 0.5, 0.0, -0.5],    # ， -> 世 / 界
    "世": [0.3, 0.2, 0.1, 1.0, 2.0, 0.5, 0.0, -0.5],    # 世 -> 界
    "界": [0.5, 0.3, 0.2, 0.5, 1.2, 1.8, 1.0, 0.5],     # 界 -> ！/ ？/ 界
    "！": [0.1, 1.0, 0.5, 0.5, 0.3, 0.2, 1.5, 1.0],     # ！ -> 好 / ？/ 。
    "？": [0.1, 1.0, 0.5, 0.5, 0.3, 0.2, 1.5, 1.0],
    "。": [0.3, 1.5, 0.8, 0.5, 0.3, 0.2, 0.5, 1.0],
}


def toy_next_token(last_token, temperature, rng):
    """玩具模型的'前向' -> 返回下一个 token。"""
    probs = softmax(TOY_MODEL[last_token], temperature)
    return VOCAB[sample_gumbel(probs, rng)]


print("=" * 62)
print("演示 5: 自回归生成 —— prompt 是 '你' (prefill), 之后逐个生成 (decode)")
print("=" * 62)
for temperature in (0.3, 1.0, 1.5):
    rng = random.Random(7)
    generated = ["你"]                      # 这就是"prompt"
    for _ in range(20):                     # 每次只生成 1 个 token!
        nxt = toy_next_token(generated[-1], temperature, rng)
        generated.append(nxt)               # 拼回去, 下一轮才能继续
    print(f"  temperature={temperature:>4}: {''.join(generated)}")
print()
print("  注意: 循环体里每次只调用一次 toy_next_token(相当于一次前向),")
print("  且输入依赖上一次的输出 —— 这就是 decode 阶段'一次只能生成一个 token'")
print("  以及 KV Cache 存在意义(把前面算过的 K/V 存起来, 避免重算)的原因。")
