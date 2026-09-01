__all__ = [
    "AdmissionController",
    "EngineWorker",
    "RequestHandle",
    "RedisPrefixRouter",
    "InstanceHeartbeat",
    "GatewayServicer",
    "serve_gateway",
    "compute_prefix_hash",
]


def __getattr__(name):
    if name == "AdmissionController":
        from nanovllm.serving.admission import AdmissionController
        return AdmissionController
    if name in ("EngineWorker", "RequestHandle"):
        from nanovllm.serving.engine_worker import EngineWorker, RequestHandle
        return {"EngineWorker": EngineWorker, "RequestHandle": RequestHandle}[name]
    if name == "RedisPrefixRouter":
        from nanovllm.serving.redis_router import RedisPrefixRouter
        return RedisPrefixRouter
    if name == "InstanceHeartbeat":
        from nanovllm.serving.redis_router import InstanceHeartbeat
        return InstanceHeartbeat
    if name in ("GatewayServicer", "serve_gateway"):
        from nanovllm.serving.gateway import GatewayServicer, serve_gateway
        return {"GatewayServicer": GatewayServicer,
                "serve_gateway": serve_gateway}[name]
    if name == "compute_prefix_hash":
        from nanovllm.serving.redis_router import compute_prefix_hash
        return compute_prefix_hash
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
