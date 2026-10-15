# -*- coding: utf-8 -*-
"""CLI entry point: python -m nanovllm.serving --model /path/to/model"""
import argparse

from nanovllm.serving.grpc_server import serve


def main():
    parser = argparse.ArgumentParser(description="nano-vllm gRPC serving")
    parser.add_argument("--gateway", action="store_true",
                        help="Run as a routing gateway instead of an engine instance")
    parser.add_argument("--mock", action="store_true",
                        help="Run with a GPU-free mock engine")
    parser.add_argument("--token-interval", type=float, default=0.01,
                        help="Mock engine token interval in seconds")
    parser.add_argument("--model", default=None, help="HuggingFace model directory")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--metrics-port", type=int, default=9100)
    parser.add_argument("--max-qps", type=float, default=100.0)
    parser.add_argument("--high-watermark", type=float, default=0.90)
    parser.add_argument("--low-watermark", type=float, default=0.70)
    parser.add_argument("--max-workers", type=int, default=64,
                        help="gRPC worker threads. A streaming Infer holds one "
                             "thread for its whole stream, so this caps concurrent "
                             "requests and the engine's batch size; keep it above "
                             "the concurrency you serve (default 64)")
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--redis-url", default=None)
    parser.add_argument("--addr", default=None,
                        help="Address advertised to Redis, e.g. 10.0.0.1:50051")
    parser.add_argument("--heartbeat-interval", type=float, default=5.0)
    args = parser.parse_args()

    if args.gateway:
        if not args.redis_url:
            parser.error("--gateway requires --redis-url")
        tokenizer = None
        if args.model:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        from nanovllm.serving.gateway import serve_gateway
        serve_gateway(
            redis_url=args.redis_url,
            port=args.port,
            metrics_port=args.metrics_port,
            max_workers=args.max_workers,
            tokenizer=tokenizer,
        )
        return

    if not args.model and not args.mock:
        parser.error("--model is required unless --gateway or --mock is used")

    engine_kwargs = {}
    if args.max_num_seqs is not None:
        engine_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.enforce_eager:
        engine_kwargs["enforce_eager"] = True
    if args.mock:
        engine_kwargs["token_interval"] = args.token_interval

    serve(
        model=args.model,
        port=args.port,
        max_workers=args.max_workers,
        max_qps=args.max_qps,
        high_watermark=args.high_watermark,
        low_watermark=args.low_watermark,
        metrics_port=args.metrics_port,
        instance_id=args.instance_id,
        redis_url=args.redis_url,
        addr=args.addr,
        heartbeat_interval=args.heartbeat_interval,
        mock=args.mock,
        **engine_kwargs,
    )


if __name__ == "__main__":
    main()
