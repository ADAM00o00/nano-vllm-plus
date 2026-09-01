# -*- coding: utf-8 -*-
"""Simple gRPC throughput/latency benchmark for nano-vllm serving.

Run after the server is up:
    python -m nanovllm.serving --model /path/to/model --port 50051
    python benchmarks/benchmark_grpc.py --target localhost:50051 --requests 64 --concurrency 8
"""
import argparse
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Allow running directly as `python benchmarks/benchmark_grpc.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grpc

from nanovllm.serving import nanovllm_pb2
from nanovllm.serving import nanovllm_pb2_grpc


def one_request(stub, prompt: str, max_tokens: int) -> tuple[int, float]:
    request_id = str(uuid.uuid4())
    request = nanovllm_pb2.InferRequest(
        request_id=request_id,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=1.0,
    )
    start = time.perf_counter()
    tokens = 0
    for response in stub.Infer(request):
        tokens += len(response.token_ids)
        if response.finished or response.aborted:
            break
    return tokens, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="localhost:50051")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt", default="Benchmark prompt: ")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    channel = grpc.insecure_channel(args.target)
    stub = nanovllm_pb2_grpc.NanoVLLMServiceStub(channel)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(one_request, stub, args.prompt, args.max_tokens)
            for _ in range(args.requests)
        ]
        results = [f.result() for f in futures]
    elapsed = time.perf_counter() - start

    total_tokens = sum(t for t, _ in results)
    latencies = sorted(d for _, d in results)
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p99 = latencies[int(len(latencies) * 0.99) - 1] if latencies else 0.0
    print(f"requests={args.requests} elapsed={elapsed:.2f}s "
          f"throughput={total_tokens / elapsed:.2f} tok/s "
          f"p50={p50 * 1000:.2f}ms p99={p99 * 1000:.2f}ms")


if __name__ == "__main__":
    main()
