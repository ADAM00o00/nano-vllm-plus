# -*- coding: utf-8 -*-
"""A GPU-free mock engine worker for local end-to-end testing.

It mimics the EngineWorker interface used by NanoVLLMServicer but emits fake
tokens instead of running an actual model.  Useful for testing gRPC, routing,
backpressure and cancellation without a GPU.
"""
import itertools
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from nanovllm.sampling_params import SamplingParams


@dataclass
class MockHandle:
    seq_id: int = -1
    output_queue: "queue.Queue[tuple[list[int], bool, bool]]" = field(
        default_factory=queue.Queue)
    done: bool = False
    aborted: bool = False
    abort_submitted_at: Optional[float] = None
    max_tokens: int = 64
    token_interval: float = 0.01


class MockEngineWorker:
    """Simulates an engine worker without loading any model."""

    def __init__(self, token_interval: float = 0.01):
        self._token_interval = token_interval
        self._seq_counter = itertools.count(1)
        self._handles: dict[int, MockHandle] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API (same shape as EngineWorker)
    # ------------------------------------------------------------------
    def submit(self, prompt, sampling_params: SamplingParams) -> MockHandle:
        handle = MockHandle(
            seq_id=next(self._seq_counter),
            max_tokens=sampling_params.max_tokens,
            token_interval=self._token_interval,
        )
        with self._lock:
            self._handles[handle.seq_id] = handle
        thread = threading.Thread(target=self._generate, args=(handle,),
                                  daemon=True)
        thread.start()
        return handle

    def abort(self, seq_id: int, handle: Optional[MockHandle] = None) -> None:
        target = handle
        if target is None:
            with self._lock:
                target = self._handles.get(seq_id)
        if target is None:
            return
        if target.abort_submitted_at is None:
            target.abort_submitted_at = time.monotonic()
        target.aborted = True
        target.done = True
        with self._lock:
            self._handles.pop(target.seq_id, None)
        target.output_queue.put(([], True, True))

    def stop(self) -> None:
        with self._lock:
            handles = list(self._handles.values())
        for h in handles:
            self.abort(h.seq_id, h)

    @property
    def cache_usage_ratio(self) -> float:
        return 0.3

    @property
    def num_waiting(self) -> int:
        return 0

    @property
    def num_running(self) -> int:
        with self._lock:
            return len(self._handles)

    def get_cached_prefix_hashes(self) -> list[int]:
        return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _generate(self, handle: MockHandle):
        try:
            for _ in range(handle.max_tokens):
                if handle.aborted or handle.done:
                    break
                time.sleep(handle.token_interval)
                if handle.aborted or handle.done:
                    break
                handle.output_queue.put(([1], False, False))
            if not handle.aborted and not handle.done:
                handle.done = True
                handle.output_queue.put(([], True, False))
        finally:
            with self._lock:
                self._handles.pop(handle.seq_id, None)
