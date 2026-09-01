# -*- coding: utf-8 -*-
"""Core logic tests that do not require a GPU."""
import queue
import sys
import threading
import time
import unittest
from pathlib import Path

# Allow running directly as `python tests/test_serving_core.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.serving.admission import AdmissionController
from nanovllm.serving.redis_router import RedisPrefixRouter, compute_prefix_hash


class AdmissionTest(unittest.TestCase):
    def test_watermark_hysteresis(self):
        ac = AdmissionController(max_qps=100, burst=100)
        self.assertEqual(ac.allow(0.5), (True, "ok"))
        # High watermark rejects.
        self.assertEqual(ac.allow(0.95)[0], False)
        # Still overloaded in the hysteresis band.
        self.assertEqual(ac.allow(0.80)[0], False)
        # Below low watermark recovers.
        self.assertEqual(ac.allow(0.50), (True, "ok"))


class BlockManagerTest(unittest.TestCase):
    def test_prefix_cache_and_usage(self):
        Sequence.block_size = 4
        bm = BlockManager(16, 4)
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9],
                       SamplingParams(max_tokens=1))
        cached = bm.can_allocate(seq)
        bm.allocate(seq, cached)
        seq.num_cached_tokens = cached * 4
        seq.num_scheduled_tokens = seq.num_tokens - cached * 4
        bm.hash_blocks(seq)

        seq2 = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 99],
                        SamplingParams(max_tokens=1))
        self.assertGreater(bm.get_num_cached_prefix_blocks(seq2), 0)
        self.assertGreater(bm.usage_ratio, 0.0)

        bm.deallocate(seq)
        self.assertEqual(bm.num_free_blocks, 16)
        self.assertEqual(bm.usage_ratio, 0.0)


class SchedulerAbortTest(unittest.TestCase):
    def test_abort_releases_sequence(self):
        cfg = object.__new__(Config)
        cfg.model = "x"
        cfg.max_num_seqs = 4
        cfg.max_num_batched_tokens = 64
        cfg.max_model_len = 256
        cfg.gpu_memory_utilization = 0.9
        cfg.tensor_parallel_size = 1
        cfg.enforce_eager = True
        cfg.hf_config = None
        cfg.eos = 1
        cfg.kvcache_block_size = 4
        cfg.num_kvcache_blocks = 16

        Sequence.block_size = 4
        scheduler = Scheduler(cfg)
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8],
                       SamplingParams(max_tokens=1))
        scheduler.add(seq)
        self.assertTrue(scheduler.abort(seq.seq_id))
        self.assertEqual(seq.status, SequenceStatus.ABORTED)
        self.assertEqual(scheduler.num_waiting, 0)


class RedisRouterTest(unittest.TestCase):
    def test_route_and_fallback(self):
        import fakeredis
        router = RedisPrefixRouter(fakeredis.FakeStrictRedis())
        router.register_instance("i1", "localhost:50051", 0.2, [111, 222])
        router.register_instance("i2", "localhost:50052", 0.8, [333])
        alive = router.get_alive_instances()
        self.assertEqual({x["instance_id"] for x in alive}, {"i1", "i2"})

        # Fallback should pick the least-loaded instance.
        chosen = router.route(list(range(256)))
        self.assertEqual(chosen["instance_id"], "i1")

        # Explicit prefix mapping should win.
        h = compute_prefix_hash(list(range(256)), block_size=256)
        router.redis.set(f"nano:prefix:{h}", "i2", ex=30)
        chosen = router.route(list(range(256)))
        self.assertEqual(chosen["instance_id"], "i2")


class GrpcFakeWorkerTest(unittest.TestCase):
    def test_infer_streaming(self):
        import grpc
        from concurrent import futures
        from nanovllm.serving import nanovllm_pb2, nanovllm_pb2_grpc
        from nanovllm.serving.grpc_server import NanoVLLMServicer

        class FakeHandle:
            def __init__(self):
                self.seq_id = 1
                self.output_queue = queue.Queue()
                self.output_queue.put(([7, 8], False, False))
                self.output_queue.put(([9], True, False))

        class FakeWorker:
            cache_usage_ratio = 0.3
            num_waiting = 0
            num_running = 1

            def submit(self, prompt, sp):
                return FakeHandle()

            def abort(self, seq_id, handle=None):
                pass

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        nanovllm_pb2_grpc.add_NanoVLLMServiceServicer_to_server(
            NanoVLLMServicer(FakeWorker()), server)
        port = server.add_insecure_port("localhost:0")
        server.start()
        try:
            channel = grpc.insecure_channel(f"localhost:{port}")
            stub = nanovllm_pb2_grpc.NanoVLLMServiceStub(channel)
            responses = list(stub.Infer(nanovllm_pb2.InferRequest(
                request_id="r1", prompt="x", max_tokens=4)))
            self.assertEqual(
                [(tuple(r.token_ids), r.finished) for r in responses],
                [((7, 8), False), ((9,), True)],
            )
        finally:
            server.stop(0)


class GatewayRoutingTest(unittest.TestCase):
    def test_gateway_routes_to_engine_instance(self):
        import fakeredis
        import grpc
        from concurrent import futures
        from nanovllm.serving import nanovllm_pb2, nanovllm_pb2_grpc
        from nanovllm.serving.gateway import GatewayServicer
        from nanovllm.serving.grpc_server import NanoVLLMServicer

        class FakeHandle:
            def __init__(self):
                self.seq_id = 1
                self.output_queue = queue.Queue()
                self.output_queue.put(([5, 6], False, False))
                self.output_queue.put(([7], True, False))

        class FakeWorker:
            cache_usage_ratio = 0.1
            num_waiting = 0
            num_running = 1

            def submit(self, prompt, sp):
                return FakeHandle()

            def abort(self, seq_id, handle=None):
                pass

        # 1. Start a fake engine instance.
        engine_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        nanovllm_pb2_grpc.add_NanoVLLMServiceServicer_to_server(
            NanoVLLMServicer(FakeWorker()), engine_server)
        engine_port = engine_server.add_insecure_port("localhost:0")
        engine_server.start()

        # 2. Register it in the fake Redis directory.
        router = RedisPrefixRouter(fakeredis.FakeStrictRedis())
        router.register_instance("engine-1", f"localhost:{engine_port}", 0.1, [])

        # 3. Start the gateway.
        gateway_server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        nanovllm_pb2_grpc.add_NanoVLLMServiceServicer_to_server(
            GatewayServicer(router), gateway_server)
        gateway_port = gateway_server.add_insecure_port("localhost:0")
        gateway_server.start()

        try:
            channel = grpc.insecure_channel(f"localhost:{gateway_port}")
            stub = nanovllm_pb2_grpc.NanoVLLMServiceStub(channel)
            responses = list(stub.Infer(nanovllm_pb2.InferRequest(
                request_id="g1", prompt="hello", max_tokens=4)))
            self.assertEqual(
                [(tuple(r.token_ids), r.finished) for r in responses],
                [((5, 6), False), ((7,), True)],
            )
        finally:
            gateway_server.stop(0)
            engine_server.stop(0)


if __name__ == "__main__":
    unittest.main()
