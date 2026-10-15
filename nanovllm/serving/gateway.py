# -*- coding: utf-8 -*-
"""Control-plane gateway: routes Infer requests via Redis prefix directory.

This is the entry point for multi-instance deployments.  Engine instances
register themselves in Redis via ``InstanceHeartbeat``; this gateway picks the
best instance per request and proxies the gRPC stream.
"""
import threading
import time
from concurrent import futures

import grpc

from nanovllm.serving import nanovllm_pb2
from nanovllm.serving import nanovllm_pb2_grpc
from nanovllm.serving.redis_router import RedisPrefixRouter


class GatewayServicer(nanovllm_pb2_grpc.NanoVLLMServiceServicer):

    def __init__(self, router: RedisPrefixRouter,
                 block_size: int = 256,
                 max_usage_ratio: float = 0.95,
                 tokenizer=None):
        self.router = router
        self.block_size = block_size
        self.max_usage_ratio = max_usage_ratio
        self.tokenizer = tokenizer
        self._request_map: dict[str, str] = {}
        self._request_map_lock = threading.Lock()

    def _resolve_prompt_token_ids(self, request) -> list[int]:
        if request.prompt_token_ids:
            return list(request.prompt_token_ids)
        if request.prompt and self.tokenizer is not None:
            return self.tokenizer.encode(request.prompt)
        return []

    def Infer(self, request, context):
        prompt_token_ids = self._resolve_prompt_token_ids(request)
        instance = self.router.route(
            prompt_token_ids,
            block_size=self.block_size,
            max_usage_ratio=self.max_usage_ratio,
        )
        if instance is None:
            context.abort(grpc.StatusCode.UNAVAILABLE,
                          "no alive nano-vllm instance")
            return

        addr = instance["addr"]
        with self._request_map_lock:
            self._request_map[request.request_id] = addr

        channel = grpc.insecure_channel(addr)
        stub = nanovllm_pb2_grpc.NanoVLLMServiceStub(channel)
        stop_watch = threading.Event()

        def watch_cancel():
            while context.is_active() and not stop_watch.is_set():
                time.sleep(0.2)
            if not context.is_active() and not stop_watch.is_set():
                try:
                    stub.Abort(nanovllm_pb2.AbortRequest(
                        request_id=request.request_id))
                except Exception:
                    pass

        watcher = threading.Thread(target=watch_cancel, daemon=True)
        watcher.start()
        try:
            for response in stub.Infer(request):
                yield response
        finally:
            stop_watch.set()
            with self._request_map_lock:
                self._request_map.pop(request.request_id, None)

    def Abort(self, request, context):
        with self._request_map_lock:
            addr = self._request_map.get(request.request_id)
        if addr is None:
            return nanovllm_pb2.AbortResponse(ok=False)
        channel = grpc.insecure_channel(addr)
        stub = nanovllm_pb2_grpc.NanoVLLMServiceStub(channel)
        return stub.Abort(request)

    def Health(self, request, context):
        instances = self.router.get_alive_instances()
        ready = len(instances) > 0
        if ready:
            usage = min(x["cache_usage_ratio"] for x in instances)
            waiting = sum(int(x.get("num_waiting", 0)) for x in instances)
            running = sum(int(x.get("num_running", 0)) for x in instances)
        else:
            usage = 0.0
            waiting = 0
            running = 0
        return nanovllm_pb2.HealthResponse(
            ready=ready,
            cache_usage_ratio=usage,
            num_waiting=waiting,
            num_running=running,
        )


def serve_gateway(redis_url: str,
                  port: int = 50050,
                  metrics_port: int = 9101,
                  max_workers: int = 64,
                  block_size: int = 256,
                  max_usage_ratio: float = 0.95,
                  tokenizer=None):
    """Start the routing gateway (blocking).

    ``max_workers`` caps concurrent requests for the same reason as the engine
    server: ``GatewayServicer.Infer`` holds its worker thread for the whole
    downstream stream (see ``nanovllm/serving/grpc_server.py:serve``).
    """
    if metrics_port > 0:
        try:
            from prometheus_client import start_http_server
            start_http_server(metrics_port)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: failed to start Prometheus on {metrics_port}: {exc}")

    import redis
    router = RedisPrefixRouter(redis.from_url(redis_url))
    servicer = GatewayServicer(router, block_size=block_size,
                               max_usage_ratio=max_usage_ratio,
                               tokenizer=tokenizer)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    nanovllm_pb2_grpc.add_NanoVLLMServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"nano-vllm gateway listening on :{port}, metrics on :{metrics_port}")
    server.wait_for_termination()
