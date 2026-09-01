# -*- coding: utf-8 -*-
"""QPS + KV-cache watermark dual backpressure controller."""
import threading
import time


class AdmissionController:
    """Token bucket for QPS plus high/low watermark hysteresis for GPU memory.

    The controller is thread-safe.
    """

    def __init__(self, max_qps: float = 100.0, burst: int = 100,
                 high_watermark: float = 0.90, low_watermark: float = 0.70):
        assert 0.0 < max_qps
        assert burst > 0
        assert 0.0 < low_watermark <= high_watermark < 1.0
        self.max_qps = max_qps
        self.burst = burst
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        self._lock = threading.Lock()
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._overloaded = False

    def _refill(self):
        now = time.monotonic()
        self._tokens = min(self.burst, self._tokens + (now - self._last_refill) * self.max_qps)
        self._last_refill = now

    def allow(self, cache_usage_ratio: float | None = None) -> tuple[bool, str]:
        """Return (allowed, reason). reason is 'ok', 'qps' or 'memory'."""
        with self._lock:
            self._refill()
            if self._tokens < 1.0:
                return False, "qps"
            if cache_usage_ratio is not None:
                if self._overloaded:
                    if cache_usage_ratio <= self.low_watermark:
                        self._overloaded = False
                    else:
                        return False, "memory"
                else:
                    if cache_usage_ratio >= self.high_watermark:
                        self._overloaded = True
                        return False, "memory"
            self._tokens -= 1.0
            return True, "ok"
