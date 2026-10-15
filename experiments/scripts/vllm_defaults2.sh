#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cat > /tmp/vdef.py <<'PY'
import dataclasses
def show(cls, names):
    print(f"--- {cls.__name__} ---")
    try:
        fs = {f.name: f for f in dataclasses.fields(cls)}
    except Exception as e:
        print("  fields err:", e); return
    for n in names:
        f = fs.get(n)
        if f is None:
            print(f"  {n}: <no field>")
        else:
            d = f.default
            if d is dataclasses.MISSING:
                d = f"factory={f.default_factory}"
            print(f"  {n}: {d}")

from vllm.config import SchedulerConfig, CacheConfig
show(SchedulerConfig, ["max_num_batched_tokens", "max_num_seqs", "enable_chunked_prefill",
                       "long_prefill_token_threshold", "scheduler_policy"])
show(CacheConfig, ["block_size", "gpu_memory_utilization", "enable_prefix_caching",
                   "num_gpu_blocks_override", "cache_dtype"])
try:
    from vllm.engine.arg_utils import EngineArgs
    ea = EngineArgs(model="/home/adam/models/Qwen3-0.6B")
    vc = ea.create_engine_config()
    sc = vc.scheduler_config
    print("--- resolved EngineArgs ---")
    for n in ["max_num_batched_tokens", "max_num_seqs", "enable_chunked_prefill"]:
        print(f"  scheduler.{n} = {getattr(sc, n, '<none>')}")
    cc = vc.cache_config
    for n in ["block_size", "gpu_memory_utilization", "enable_prefix_caching"]:
        print(f"  cache.{n} = {getattr(cc, n, '<none>')}")
except Exception as e:
    import traceback; traceback.print_exc()
PY
python /tmp/vdef.py 2>&1 | grep -v -E "^(INFO|WARNING)" | tail -40
