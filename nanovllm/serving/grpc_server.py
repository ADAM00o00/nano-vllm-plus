# -*- coding: utf-8 -*-
"""gRPC server for nano-vllm serving.

Run after generating proto stubs:
    python nanovllm/serving/generate_proto.py
"""
import queue
import threading
from concurrent import futures

import grpc

from nanovllm.sampling_params import SamplingParams
from nanovllm.serving import metrics
from nanovllm.serving.admission import AdmissionController

from nanovllm.serving import nanovllm_pb2
from nanovllm.serving import nanovllm_pb2_grpc


class NanoVLLMServicer(nanovllm_pb2_grpc.NanoVLLMServiceServicer):

    def __init__(self, worker,
                 admission: AdmissionController | None = None):
        self.worker = worker
        self.admission = admission or AdmissionController()
        self._request_map: dict[str, int] = {}
        self._request_map_lock = threading.Lock()

    # ------------------------------------------------------------------
    # gRPC methods
    # ------------------------------------------------------------------
    def Infer(self, request, context):
        allowed, reason = self.admission.allow(self.worker.cache_usage_ratio)
        if not allowed:
            metrics.ADMISSION_REJECTED_TOTAL.labels(reason=reason).inc()
            context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                          f"backpressure: {reason}")
            return

        if request.prompt_token_ids:
            prompt = list(request.prompt_token_ids)
        else:
            prompt = request.prompt

        # Prefix-cache hit/miss observability (only when token ids are available).
        if isinstance(prompt, list):
            has_cached_prefix = getattr(self.worker, "has_cached_prefix", None)
            if callable(has_cached_prefix):
                try:
                    if has_cached_prefix(prompt):
                        metrics.PREFIX_CACHE_HIT_TOTAL.inc()
                    else:
                        metrics.PREFIX_CACHE_MISS_TOTAL.inc()
                except Exception:
                    pass

        sampling_params = SamplingParams(
            temperature=request.temperature if request.temperature > 0 else 1.0,
            max_tokens=request.max_tokens if request.max_tokens > 0 else 64,
            ignore_eos=request.ignore_eos,
        )

        handle = self.worker.submit(prompt, sampling_params)
        with self._request_map_lock:
            self._request_map[request.request_id] = handle.seq_id
        metrics.REQUESTS_TOTAL.inc()
        metrics.INFLIGHT_REQUESTS.inc()

        try:
            while True:
                if context.is_active() is False:
                    # Client disconnected/cancelled -> cascade abort.
                    self.worker.abort(handle.seq_id, handle=handle)
                    metrics.ABORT_TOTAL.inc()
                    break
                try:
                    chunk = handle.output_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                token_ids, finished, aborted = chunk
                if aborted:
                    metrics.ABORT_TOTAL.inc()
                yield self._infer_response(request.request_id, handle.seq_id,
                                           token_ids, finished, aborted)
                if finished or aborted:
                    break
        finally:
            metrics.INFLIGHT_REQUESTS.dec()
            with self._request_map_lock:
                self._request_map.pop(request.request_id, None)

    def Abort(self, request, context):
        with self._request_map_lock:
            seq_id = self._request_map.get(request.request_id)
        if seq_id is None:
            return nanovllm_pb2.AbortResponse(ok=False)
        self.worker.abort(seq_id)
        metrics.ABORT_TOTAL.inc()
        return nanovllm_pb2.AbortResponse(ok=True)

    def Health(self, request, context):
        return nanovllm_pb2.HealthResponse(
            ready=True,
            cache_usage_ratio=self.worker.cache_usage_ratio,
            num_waiting=self.worker.num_waiting,
            num_running=self.worker.num_running,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_response(request_id: str, seq_id: int,
                        token_ids: list[int], finished: bool, aborted: bool):
        return nanovllm_pb2.InferResponse(
            request_id=request_id,
            token_ids=token_ids,
            finished=finished,
            aborted=aborted,
            seq_id=seq_id,
        )


def serve(model: str | None = None, port: int = 50051,
          max_workers: int = 10,
          max_qps: float = 100.0,
          high_watermark: float = 0.90,
          low_watermark: float = 0.70,
          metrics_port: int = 9100,
          instance_id: str | None = None,
          redis_url: str | None = None,
          addr: str | None = None,
          heartbeat_interval: float = 5.0,
          mock: bool = False,
          **engine_kwargs):
    """Start the gRPC server (blocking).

    If ``mock`` is True, a GPU-free MockEngineWorker is used instead of the
    real nano-vllm engine.
    """
    if metrics_port > 0:
        try:
            from prometheus_client import start_http_server
            start_http_server(metrics_port)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: failed to start Prometheus on {metrics_port}: {exc}")

    if mock:
        from nanovllm.serving.mock_engine import MockEngineWorker
        worker = MockEngineWorker(token_interval=engine_kwargs.get("token_interval", 0.01))
    else:
        if model is None:
            raise ValueError("model is required when mock=False")
        from nanovllm.serving.engine_worker import EngineWorker
        worker = EngineWorker(model, **engine_kwargs)
    admission = AdmissionController(max_qps=max_qps,
                                    high_watermark=high_watermark,
                                    low_watermark=low_watermark)
    servicer = NanoVLLMServicer(worker, admission)

    # Optional Redis instance heartbeat for multi-instance prefix routing.
    heartbeat = None
    if instance_id and redis_url:
        import redis
        from nanovllm.serving.redis_router import InstanceHeartbeat, RedisPrefixRouter

        redis_client = redis.from_url(redis_url)
        router = RedisPrefixRouter(redis_client)
        actual_addr = addr or f"localhost:{port}"
        heartbeat = InstanceHeartbeat(
            router, instance_id, actual_addr,
            interval=heartbeat_interval,
            get_state=lambda: (worker.cache_usage_ratio,
                               worker.get_cached_prefix_hashes()),
        )
        heartbeat.start()
        print(f"[redis] instance {instance_id} registered at {actual_addr}")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    nanovllm_pb2_grpc.add_NanoVLLMServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"nano-vllm gRPC server listening on :{port}, metrics on :{metrics_port}")
    try:
        server.wait_for_termination()
    finally:
        if heartbeat is not None:
            heartbeat.stop()
        worker.stop()
