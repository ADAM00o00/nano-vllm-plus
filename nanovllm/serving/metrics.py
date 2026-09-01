# -*- coding: utf-8 -*-
"""Prometheus metrics for the nano-vllm serving layer."""
from prometheus_client import Counter, Gauge, Histogram

REQUESTS_TOTAL = Counter(
    "nano_vllm_requests_total", "Total number of inference requests accepted")
INFLIGHT_REQUESTS = Gauge(
    "nano_vllm_inflight_requests", "Number of in-flight inference requests")
QUEUE_DEPTH = Gauge(
    "nano_vllm_queue_depth", "Number of sequences waiting in scheduler")
RUNNING_SEQS = Gauge(
    "nano_vllm_running_seqs", "Number of sequences running in scheduler")
CACHE_USAGE_RATIO = Gauge(
    "nano_vllm_cache_usage_ratio", "KV cache usage ratio")
PREFIX_CACHE_HIT_TOTAL = Counter(
    "nano_vllm_prefix_cache_hit_total", "Requests with prefix cache hit")
PREFIX_CACHE_MISS_TOTAL = Counter(
    "nano_vllm_prefix_cache_miss_total", "Requests without prefix cache hit")
ADMISSION_REJECTED_TOTAL = Counter(
    "nano_vllm_admission_rejected_total",
    "Admission control rejections", ["reason"])
ABORT_TOTAL = Counter(
    "nano_vllm_abort_total", "Requests aborted through cancellation")
RECLAIM_DURATION_SECONDS = Histogram(
    "nano_vllm_reclaim_duration_seconds",
    "Time from abort request to KV cache reclamation observation")
