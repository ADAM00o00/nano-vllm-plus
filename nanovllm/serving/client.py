# -*- coding: utf-8 -*-
"""Minimal gRPC client for the nano-vllm serving layer."""
import argparse
import sys
import uuid
from pathlib import Path

# Allow running directly as `python nanovllm/serving/client.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import grpc

from nanovllm.serving import nanovllm_pb2
from nanovllm.serving import nanovllm_pb2_grpc


def infer(stub, prompt: str, max_tokens: int = 64, temperature: float = 1.0):
    request_id = str(uuid.uuid4())
    request = nanovllm_pb2.InferRequest(
        request_id=request_id,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    tokens: list[int] = []
    for response in stub.Infer(request):
        tokens.extend(response.token_ids)
        if response.aborted:
            print("aborted by server")
            break
        if response.finished:
            break
    return tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="localhost:50051")
    parser.add_argument("--prompt", default="Hello, Nano-vLLM.")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    channel = grpc.insecure_channel(args.target)
    stub = nanovllm_pb2_grpc.NanoVLLMServiceStub(channel)
    tokens = infer(stub, args.prompt, args.max_tokens)
    print(f"generated tokens: {tokens}")


if __name__ == "__main__":
    main()
