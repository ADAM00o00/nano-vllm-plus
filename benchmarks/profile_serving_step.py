# -*- coding: utf-8 -*-
"""Run the real gRPC engine with per-decode-step timing instrumentation.

Motivation: driving ``LLMEngine`` directly, a batch-8 decode step takes 6.35 ms
(host time == GPU time, so nothing is starved). Under gRPC load the same engine
delivers a token to a client only every ~10 ms, and throughput saturates at ~800
tok/s from concurrency 8 upward -- even though a batch-16 step costs only 6.97 ms
(~2300 tok/s of capability). So either the step itself gets slower under load, or
the worker thread is not stepping back to back.

This harness answers that directly: it wraps ``LLMEngine.step`` and records, per
step, the host duration and the GPU duration (CUDA events), then dumps them when
it shuts down. Compare:

* GPU_ms ~= 6.3 and host_ms ~= 6.3  -> healthy, look at the transport
* GPU_ms ~= 6.3 and host_ms >> 6.3  -> the step is host/GIL bound
* GPU_ms >> 6.3                     -> the GPU really is busier

Usage:

    python benchmarks/profile_serving_step.py --model /home/adam/models/Qwen3-0.6B \
        --port 50051 --write /tmp/serving_steps.json
    # in another shell:
    python benchmarks/benchmark_grpc.py --target localhost:50051 \
        --requests 32 --concurrency 8 --max-tokens 32
    # then Ctrl-C the server (or SIGTERM) to flush the stats
"""

from __future__ import annotations

import argparse
import json
import signal
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import nanovllm.engine.llm_engine as llm_engine_module
from nanovllm.engine.llm_engine import LLMEngine

_RECORDS: list[tuple[float, float, float, int]] = []  # (monotonic_t, host_ms, gpu_ms, batch)
_LOCK = threading.Lock()
_ORIGINAL_STEP = LLMEngine.step


def _instrumented_step(self):
    """Time one engine step, and record how many sequences it actually batched.

    The batch size matters as much as the timing: if the engine steps every
    6.3 ms but only ever batches 1-2 sequences, the GPU is idle and the problem
    is scheduling, not the step itself.

    Each record carries a ``time.monotonic()`` stamp so one server run can be
    split into per-concurrency phases by an external load driver (CLOCK_MONOTONIC
    is system-wide on Linux, so the driver's stamps are comparable).
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    t_start = time.monotonic()
    start.record()
    result = _ORIGINAL_STEP(self)
    end.record()
    host_ms = (time.perf_counter() - t_start) * 1e3
    # The event pair covers the graph replay plus whatever ran on the stream in
    # between. elapsed_time needs a sync, which is fine here: the engine already
    # syncs every step when it pulls the sampled token back with .tolist().
    torch.cuda.synchronize()
    gpu_ms = start.elapsed_time(end)
    num_tokens = result[1] if isinstance(result, tuple) and len(result) > 1 else 0
    batch = -num_tokens if num_tokens < 0 else 0  # decode steps report -num_seqs
    with _LOCK:
        _RECORDS.append((t_start, host_ms, gpu_ms, batch))
    return result


def _dump(path: str) -> None:
    with _LOCK:
        records = list(_RECORDS)
    if not records:
        print("no steps recorded")
        return
    host = [r[1] for r in records]
    gpu = [r[2] for r in records]
    decode = [r for r in records if r[3] > 0]
    batches_d = [r[3] for r in decode]
    summary = {
        "steps": len(records),
        "decode_steps": len(decode),
        "prefill_steps": len(records) - len(decode),
        "batch": {
            "mean": statistics.fmean(batches_d) if batches_d else 0,
            "median": statistics.median(batches_d) if batches_d else 0,
            "max": max(batches_d) if batches_d else 0,
            "histogram": {str(b): batches_d.count(b) for b in sorted(set(batches_d))},
        },
        "host_ms": {
            "mean": statistics.fmean(host),
            "median": statistics.median(host),
            "p95": sorted(host)[max(0, int(0.95 * len(host)) - 1)],
        },
        "gpu_ms": {
            "mean": statistics.fmean(gpu),
            "median": statistics.median(gpu),
            "p95": sorted(gpu)[max(0, int(0.95 * len(gpu)) - 1)],
        },
        "sum_host_s": sum(host) / 1e3,
        "sum_gpu_s": sum(gpu) / 1e3,
        "tokens_emitted": sum(batches_d),
    }
    if decode:
        # What the GPU could have produced if it had run back to back.
        total = sum(host)  # ms of engine-thread time
        summary["engine_busy_fraction_of_step_time"] = (
            sum(gpu) / total if total else 0.0
        )
    print("\n=== decode step timing under serving load ===")
    print(json.dumps(summary, indent=2))
    Path(path).write_text(
        json.dumps({"summary": summary, "steps": [list(r) for r in records]}, indent=2)
    )
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/home/adam/models/Qwen3-0.6B")
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--metrics-port", type=int, default=9100)
    ap.add_argument("--write", default="/tmp/serving_steps.json")
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument(
        "--max-qps", type=float, default=100.0,
        help="admissions QPS token bucket. The default 100 with burst 100 rejects "
             "high-concurrency bursts with RESOURCE_EXHAUSTED 'backpressure: qps', "
             "which silently caps measured throughput; raise it when load-driving",
    )
    ap.add_argument(
        "--max-workers", type=int, default=10,
        help="grpc ThreadPoolExecutor size; each streaming Infer holds one thread "
             "for the whole stream, so this caps concurrent streams and therefore "
             "the engine's batch size (see docs/sm89_phase1_results.md)",
    )
    args = ap.parse_args()

    LLMEngine.step = _instrumented_step
    llm_engine_module.LLMEngine.step = _instrumented_step

    def _on_signal(signum, frame):
        _dump(args.write)
        # Restore and re-raise so serve()'s own cleanup runs.
        LLMEngine.step = _ORIGINAL_STEP
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    from nanovllm.serving.grpc_server import serve

    serve_kwargs = dict(
        model=args.model, port=args.port, metrics_port=args.metrics_port,
        max_workers=args.max_workers, max_qps=args.max_qps,
    )
    if args.max_num_seqs is not None:
        serve_kwargs["max_num_seqs"] = args.max_num_seqs
    try:
        serve(**serve_kwargs)
    finally:
        _dump(args.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
