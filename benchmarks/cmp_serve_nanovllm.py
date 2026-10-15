# -*- coding: utf-8 -*-
"""Launch the nano-vllm gRPC server with an explicit engine envelope.

``python -m nanovllm.serving`` only forwards ``--max-num-seqs`` and
``--enforce-eager`` to the engine, so ``max_num_batched_tokens`` (and an
explicit ``gpu_memory_utilization``) cannot be set from the CLI.  For a fair
head-to-head against vLLM we need those knobs, so this launcher calls
``serve()`` directly instead of changing the project's CLI.

    python benchmarks/cmp_serve_nanovllm.py --model /home/adam/models/Qwen3-0.6B \
        --port 50051 --max-workers 256 --max-qps 10000 \
        --max-num-batched-tokens 16384 --max-num-seqs 512
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm.serving.grpc_server import serve  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/adam/models/Qwen3-0.6B")
    p.add_argument("--port", type=int, default=50051)
    p.add_argument("--metrics-port", type=int, default=9100)
    p.add_argument("--max-workers", type=int, default=256)
    p.add_argument("--max-qps", type=float, default=10000.0)
    p.add_argument("--high-watermark", type=float, default=0.90,
                   help="KV-cache usage ratio at which admission control starts "
                        "rejecting with RESOURCE_EXHAUSTED (default 0.90)")
    p.add_argument("--low-watermark", type=float, default=0.70)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--num-kvcache-blocks", type=int, default=168,
                   help="Pin the KV pool size (blocks of kvcache_block_size "
                        "tokens) so it is reproducible and byte-comparable with "
                        "vLLM's --kv-cache-memory")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--decode-attn", default=None)
    args = p.parse_args()

    if args.decode_attn:
        import os
        os.environ["NANOVLLM_DECODE_ATTN"] = args.decode_attn
        import nanovllm.layers.attention as _attn
        _attn.DECODE_ATTN_BACKEND = args.decode_attn

    # text_config.max_position_embeddings is 40960 for Qwen3-0.6B; Config caps
    # it at min(max_model_len, hf.max_position_embeddings).
    serve(
        model=args.model,
        port=args.port,
        metrics_port=args.metrics_port,
        max_workers=args.max_workers,
        max_qps=args.max_qps,
        high_watermark=args.high_watermark,
        low_watermark=args.low_watermark,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        enforce_eager=args.enforce_eager,
    )


if __name__ == "__main__":
    main()
