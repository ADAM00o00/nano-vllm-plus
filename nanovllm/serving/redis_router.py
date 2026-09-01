# -*- coding: utf-8 -*-
"""Redis-backed prefix cache directory and router.

Designed for a multi-instance nano-vllm deployment:

- Each engine instance registers itself in Redis and heartbeats.
- Each instance may report the prefix hashes it currently caches.
- The router maps a request's prefix hash to the instance most likely to
  have the prefix cached, with fallback to the least-loaded instance.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import xxhash

PREFIX_TTL = 30
INSTANCE_TTL = 10


def compute_prefix_hash(prompt_token_ids: list[int], block_size: int = 256) -> Optional[str]:
    """Return the hash of the first *complete* KV block, or None if the prompt
    is shorter than one block (nano-vllm only caches complete blocks).

    This intentionally mirrors ``BlockManager.compute_hash`` so the Redis
    directory keys match the hashes reported by engine instances.
    """
    if len(prompt_token_ids) < block_size:
        return None
    prefix = prompt_token_ids[:block_size]
    h = xxhash.xxh64()
    h.update(np.array(prefix).tobytes())
    return str(h.intdigest())


class RedisPrefixRouter:
    """Control-plane prefix router.

    Args:
        redis_client: a redis.Redis client.
    """

    def __init__(self, redis_client, prefix_ttl: int = PREFIX_TTL,
                 instance_ttl: int = INSTANCE_TTL):
        self.redis = redis_client
        self.prefix_ttl = prefix_ttl
        self.instance_ttl = instance_ttl

    # ------------------------------------------------------------------
    # Instance registration / heartbeat
    # ------------------------------------------------------------------
    def register_instance(self, instance_id: str, addr: str,
                          cache_usage_ratio: float = 0.0,
                          prefix_hashes: Optional[list[int]] = None,
                          num_waiting: int = 0,
                          num_running: int = 0) -> None:
        self.redis.sadd("nano:instances", instance_id)
        self._write_instance(instance_id, addr, cache_usage_ratio, prefix_hashes,
                             num_waiting, num_running)

    def heartbeat(self, instance_id: str, addr: str,
                  cache_usage_ratio: float,
                  prefix_hashes: Optional[list[int]] = None,
                  num_waiting: int = 0,
                  num_running: int = 0) -> None:
        self.redis.sadd("nano:instances", instance_id)
        self._write_instance(instance_id, addr, cache_usage_ratio, prefix_hashes,
                             num_waiting, num_running)

    def _write_instance(self, instance_id: str, addr: str,
                        cache_usage_ratio: float,
                        prefix_hashes: Optional[list[int]],
                        num_waiting: int = 0,
                        num_running: int = 0) -> None:
        key = f"nano:instance:{instance_id}"
        now = time.time()
        data = {
            "addr": addr,
            "cache_usage_ratio": float(cache_usage_ratio),
            "last_seen": now,
            "num_waiting": int(num_waiting),
            "num_running": int(num_running),
        }
        if prefix_hashes is not None:
            data["prefix_count"] = len(prefix_hashes)
        self.redis.hset(key, mapping=data)
        self.redis.expire(key, self.instance_ttl)
        if prefix_hashes:
            for h in prefix_hashes:
                self.redis.set(f"nano:prefix:{h}", instance_id, ex=self.prefix_ttl)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def get_alive_instances(self) -> list[dict]:
        """Return instances whose Redis key is still alive."""
        result = []
        for raw_instance_id in self.redis.smembers("nano:instances"):
            instance_id = raw_instance_id.decode() if isinstance(raw_instance_id, bytes) else raw_instance_id
            key = f"nano:instance:{instance_id}"
            raw = self.redis.hgetall(key)
            if not raw:
                self.redis.srem("nano:instances", raw_instance_id)
                continue
            data = {k.decode() if isinstance(k, bytes) else k:
                    v.decode() if isinstance(v, bytes) else v
                    for k, v in raw.items()}
            data["instance_id"] = instance_id
            data["cache_usage_ratio"] = float(data.get("cache_usage_ratio", 0.0))
            result.append(data)
        return result

    def route(self, prompt_token_ids: list[int],
              block_size: int = 256,
              max_usage_ratio: float = 0.95,
              prefer_least_loaded: bool = True) -> Optional[dict]:
        """Return the chosen instance dict, or None if no instance is alive."""
        instances = self.get_alive_instances()
        if not instances:
            return None

        prefix_hash = compute_prefix_hash(prompt_token_ids, block_size)
        if prefix_hash is not None:
            cached_instance_id = self.redis.get(f"nano:prefix:{prefix_hash}")
            if cached_instance_id is not None:
                if isinstance(cached_instance_id, bytes):
                    cached_instance_id = cached_instance_id.decode()
                for inst in instances:
                    if inst["instance_id"] == cached_instance_id and \
                            inst["cache_usage_ratio"] <= max_usage_ratio:
                        return inst

        # Fallback: least-loaded instance.
        if prefer_least_loaded:
            chosen = min(instances, key=lambda x: x["cache_usage_ratio"])
        else:
            chosen = instances[0]
        if prefix_hash is not None:
            self.redis.set(f"nano:prefix:{prefix_hash}", chosen["instance_id"],
                           ex=self.prefix_ttl)
        return chosen


class InstanceHeartbeat:
    """Background thread that keeps an engine instance alive in Redis."""

    def __init__(self, router: RedisPrefixRouter, instance_id: str, addr: str,
                 interval: float = 5.0,
                 get_state=None):
        self.router = router
        self.instance_id = instance_id
        self.addr = addr
        self.interval = interval
        self.get_state = get_state or (lambda: (0.0, []))
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="redis-heartbeat")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop:
            try:
                state = self.get_state()
                if len(state) >= 4:
                    usage_ratio, prefix_hashes, num_waiting, num_running = state[:4]
                else:
                    usage_ratio, prefix_hashes = state
                    num_waiting = num_running = 0
                self.router.heartbeat(self.instance_id, self.addr,
                                      usage_ratio, prefix_hashes,
                                      num_waiting=num_waiting,
                                      num_running=num_running)
            except Exception as exc:
                print(f"[redis-heartbeat] {exc}")
            time.sleep(self.interval)

