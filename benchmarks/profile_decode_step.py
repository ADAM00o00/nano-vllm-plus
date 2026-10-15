# -*- coding: utf-8 -*-
"""Profile one decode step of nano-vllm: wall time vs GPU busy time.

This exists to settle a specific question: a decode step must stream the whole
model (596M params = 1.110 GiB bf16) out of DRAM every step, which at this
machine's measured 238 GB/s is a ~5.0 ms floor that no kernel can beat and that
does not shrink with batch size. If the measured step takes ~10 ms, then roughly
half the step is *not* spent moving data, and that half is where any end-to-end
win has to come from.

The harness drives ``LLMEngine.step`` directly (rather than ``generate``) so the
timing is not entangled with the scheduler loop or tqdm, and reports:

* per-step GPU wall time (CUDA events around the step, so host-side jitter in
  the Python loop does not hide in the number),
* host wall time for the same steps, so the CPU-side cost is visible,
* with ``--torch-profiler``, the summed GPU kernel time and the top kernels by
  self time -- the difference against the wall time is the non-kernel gap.

Usage (WSL ``ml`` env):

    python benchmarks/profile_decode_step.py --batch 8 --prompt-len 10 --steps 24
    python benchmarks/profile_decode_step.py --batch 8 --prompt-len 2048 --steps 12
    python benchmarks/profile_decode_step.py --batch 8 --prompt-len 10 --torch-profiler

Under Nsight Systems the same loop is what you want to trace, e.g.

    nsys profile -o /tmp/dec -f true --trace=cuda,nvtx \
        python benchmarks/profile_decode_step.py --batch 8 --prompt-len 10 --steps 24
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from nanovllm import LLM, SamplingParams

MODEL = "/home/adam/models/Qwen3-0.6B"

# Qwen3-0.6B, bf16: 596.0M params -> 1.110 GiB read from DRAM every step, plus
# 112 KiB of KV per token of context per sequence. Measured achievable read
# bandwidth on this machine is 238 GB/s.
WEIGHT_BYTES = 1_192_000_000
KV_BYTES_PER_TOKEN = 2 * 8 * 128 * 2 * 28  # k+v, kv_heads, head_dim, bf16, layers
MEASURED_PEAK_GBPS = 238.0


def roofline_floor_ms(batch: int, ctx: int) -> float:
    total = WEIGHT_BYTES + batch * ctx * KV_BYTES_PER_TOKEN
    return total / (MEASURED_PEAK_GBPS * 1e9) * 1e3


def build_engine(batch: int, prompt_len: int, max_tokens: int, enforce_eager: bool) -> LLM:
    llm = LLM(MODEL, enforce_eager=enforce_eager, max_num_seqs=max(batch, 16))
    # Token-id prompts so the context length is exact and sliceable per step.
    base = [1000 + (i % 500) for i in range(prompt_len)]
    sp = SamplingParams(temperature=1e-6, max_tokens=max_tokens)
    for i in range(batch):
        prompt = base[: prompt_len - 1] + [1000 + i]
        llm.add_request(prompt, sp)
    return llm


def run(args) -> int:
    torch.cuda.init()
    llm = build_engine(args.batch, args.prompt_len, args.steps + 1, args.enforce_eager)
    is_prefill = True

    # Warm up: the first scheduled step is prefill, and a couple of decode steps
    # make sure the CUDA graphs and any lazy compilation are done.
    llm.step()
    is_prefill = False
    steps = []

    profiler = None
    if args.torch_profiler:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        )
        profiler.start()

    if args.nsys_range:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()

    for _ in range(args.steps):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        t0 = perf_counter()
        start.record()
        _outputs, num_tokens = llm.step()
        end.record()
        t1 = perf_counter()
        torch.cuda.synchronize()
        if num_tokens > 0:  # a prefill slipped in; do not mix it into decode stats
            is_prefill = True
            continue
        steps.append((start.elapsed_time(end), (t1 - t0) * 1e3, -num_tokens))

    if args.nsys_range:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

    if profiler is not None:
        profiler.stop()
        print("=== torch.profiler: top kernels by self CUDA time ===")
        print(
            profiler.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=args.top_kernels
            )
        )
        events = profiler.key_averages()
        kernel_us = sum(e.self_device_time_total for e in events)
        cpu_us = sum(e.self_cpu_time_total for e in events)
    else:
        kernel_us = None
        cpu_us = None

    if not steps:
        print("no decode steps were measured", file=sys.stderr)
        return 1

    gpu_ms = [s[0] for s in steps]
    host_ms = [s[1] for s in steps]
    ctx = args.prompt_len + 1

    print()
    print(f"model={MODEL}")
    print(
        f"batch={args.batch} prompt_len={args.prompt_len} (decode ctx ~{ctx}) "
        f"steps={len(steps)} enforce_eager={args.enforce_eager}"
    )
    print(
        f"decode step GPU time: min {min(gpu_ms):.3f} ms  median {statistics.median(gpu_ms):.3f} ms  "
        f"max {max(gpu_ms):.3f} ms  mean {statistics.fmean(gpu_ms):.3f} ms"
    )
    print(
        f"decode step HOST time: min {min(host_ms):.3f} ms  median {statistics.median(host_ms):.3f} ms  "
        f"max {max(host_ms):.3f} ms"
    )
    print(f"tokens per step (batch actually scheduled): {steps[0][2]}")

    floor = roofline_floor_ms(args.batch, ctx)
    med = statistics.median(gpu_ms)
    print()
    print(f"roofline floor (weights + logical KV @ {MEASURED_PEAK_GBPS:.0f} GB/s) = {floor:.3f} ms")
    print(f"step efficiency (floor / measured GPU step)          = {100 * floor / med:.1f}%")
    print(f"non-bandwidth time per step                          = {med - floor:.3f} ms")

    if kernel_us is not None:
        kernel_ms = kernel_us / 1e3
        print()
        print(f"summed GPU kernel time per step (torch.profiler)     = {kernel_ms / len(steps):.3f} ms")
        print(f"gap (GPU step - summed kernels) / step               = "
              f"{(med - kernel_ms / len(steps)):.3f} ms")
        print(f"GPU busy fraction of the step                        = "
              f"{100 * (kernel_ms / len(steps)) / med:.1f}%")
        print(f"host CPU time per step (profiler self_cpu)           = {cpu_us / 1e3 / len(steps):.3f} ms")

    if args.json:
        import json

        Path(args.json).write_text(
            json.dumps(
                {
                    "batch": args.batch,
                    "prompt_len": args.prompt_len,
                    "ctx": ctx,
                    "steps": len(steps),
                    "enforce_eager": args.enforce_eager,
                    "gpu_ms": gpu_ms,
                    "host_ms": host_ms,
                    "roofline_floor_ms": floor,
                    "step_efficiency": floor / med,
                    "kernel_ms_total": (kernel_us / 1e3) if kernel_us is not None else None,
                },
                indent=2,
            )
        )
        print(f"wrote {args.json}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=10)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--top-kernels", type=int, default=20)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--torch-profiler", action="store_true")
    ap.add_argument(
        "--nsys-range", action="store_true",
        help="bracket just the measured steps with cudaProfilerStart/Stop so that "
             "`nsys profile --capture-range=cudaProfilerApi` records steady state "
             "instead of also recording model load and graph capture",
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
