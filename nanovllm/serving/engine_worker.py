# -*- coding: utf-8 -*-
"""Thread-safe wrapper around LLMEngine.

The LLMEngine is not thread-safe.  All engine calls are executed on a single
background worker thread.  External threads communicate through a command
queue and per-request output queues.
"""
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Union

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams
from nanovllm.serving.metrics import (
    CACHE_USAGE_RATIO,
    QUEUE_DEPTH,
    RECLAIM_DURATION_SECONDS,
    RUNNING_SEQS,
)
from nanovllm.serving.redis_router import compute_prefix_hash

TokenIds = list[int]
Prompt = Union[str, list[int]]


@dataclass
class RequestHandle:
    """Per-request state shared between gRPC thread and worker thread."""
    seq_id: int = -1
    output_queue: "queue.Queue[tuple[TokenIds, bool, bool]]" = field(
        default_factory=queue.Queue)
    cursor: int = 0
    finished: bool = False
    aborted: bool = False
    done: bool = False
    abort_submitted_at: Optional[float] = None


class EngineWorker:
    """Owns an LLMEngine and runs its step loop in a background thread."""

    def __init__(self, model: str, **kwargs):
        self.llm = LLMEngine(model, **kwargs)
        self._block_size = self.llm.scheduler.block_size
        self._cached_prefix_hashes: set[int] = set()
        self._cmd_queue: "queue.Queue" = queue.Queue()
        self._handles: dict[int, RequestHandle] = {}
        self._handles_lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(
            target=self._run, name="engine-worker", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------
    def submit(self, prompt: Prompt, sampling_params: SamplingParams) -> RequestHandle:
        handle = RequestHandle()
        self._cmd_queue.put(("add", prompt, sampling_params, handle))
        return handle

    def abort(self, seq_id: int, handle: Optional[RequestHandle] = None) -> None:
        if handle is not None:
            handle.abort_submitted_at = time.monotonic()
        self._cmd_queue.put(("abort", seq_id))

    def stop(self) -> None:
        self._cmd_queue.put(("stop",))
        self._thread.join(timeout=10.0)

    @property
    def cache_usage_ratio(self) -> float:
        return self.llm.cache_usage_ratio

    @property
    def num_waiting(self) -> int:
        return self.llm.num_waiting

    @property
    def num_running(self) -> int:
        return self.llm.num_running

    def get_cached_prefix_hashes(self) -> list[int]:
        return self.llm.get_cached_prefix_hashes()

    def has_cached_prefix(self, prompt_token_ids: list[int]) -> bool:
        """Thread-safe check using a worker-maintained snapshot of prefix hashes."""
        prefix_hash = compute_prefix_hash(prompt_token_ids, self._block_size)
        if prefix_hash is None:
            return False
        return int(prefix_hash) in self._cached_prefix_hashes

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------
    def _run(self):
        while not self._stop:
            self._consume_commands()

            if self.llm.is_finished():
                # No work: wait for a command.
                try:
                    cmd = self._cmd_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                self._handle_command(cmd)
                continue

            # There is work; drain commands once more right before a step so
            # aborts submitted from another thread take effect at step granularity.
            self._consume_commands()
            if self.llm.is_finished():
                continue

            outputs, _ = self.llm.step()
            self._cached_prefix_hashes = set(self.llm.get_cached_prefix_hashes())
            QUEUE_DEPTH.set(self.num_waiting)
            RUNNING_SEQS.set(self.num_running)
            CACHE_USAGE_RATIO.set(self.cache_usage_ratio)
            self._publish_finished(outputs)
            self._publish_incremental()

    def _consume_commands(self):
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_command(cmd)
            if self._stop:
                break

    def _handle_command(self, cmd):
        kind = cmd[0]
        if kind == "add":
            _, prompt, sampling_params, handle = cmd
            seq = self.llm.add_request(prompt, sampling_params)
            handle.seq_id = seq.seq_id
            with self._handles_lock:
                self._handles[handle.seq_id] = handle
        elif kind == "abort":
            _, seq_id = cmd
            self.llm.abort_request(seq_id)
            with self._handles_lock:
                handle = self._handles.pop(seq_id, None)
            if handle is not None:
                if handle.abort_submitted_at is not None:
                    RECLAIM_DURATION_SECONDS.observe(
                        time.monotonic() - handle.abort_submitted_at)
                handle.aborted = True
                handle.finished = True
                handle.done = True
                handle.output_queue.put(([], True, True))
        elif kind == "stop":
            self._stop = True

    def _publish_finished(self, outputs: list[tuple[int, TokenIds]]):
        for seq_id, token_ids in outputs:
            with self._handles_lock:
                handle = self._handles.pop(seq_id, None)
            if handle is None:
                continue
            new_tokens = token_ids[handle.cursor:]
            handle.cursor = len(token_ids)
            handle.finished = True
            handle.done = True
            handle.output_queue.put((new_tokens, True, False))

    def _publish_incremental(self):
        with self._handles_lock:
            running_handles = []
            for seq in self.llm.scheduler.running:
                handle = self._handles.get(seq.seq_id)
                if handle is not None and not handle.done:
                    running_handles.append((handle, seq.completion_token_ids))
        for handle, tokens in running_handles:
            if handle.cursor < len(tokens):
                new_tokens = tokens[handle.cursor:]
                handle.cursor = len(tokens)
                handle.output_queue.put((new_tokens, False, False))
