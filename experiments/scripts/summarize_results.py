import glob
import json
import sys

KEYS = ["kv_blocks", "block_size", "num_gpu_blocks", "gpu_memory_utilization",
        "max_num_batched_tokens", "max_num_seqs", "load_s", "decode_attn",
        "enforce_eager", "enable_prefix_caching", "errors", "n_ok",
        "prefix_cache_blocks_max", "prefix_cache_tokens_max_at_first_token",
        "max_num_seqs", "max_model_len"]

for path in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "results/*.json")):
    if "smoke" in path:
        continue
    d = json.load(open(path))
    t = d["ttft_ms"]
    extra = {k: d[k] for k in KEYS if k in d}
    print(path)
    print("   %s conc=%s n=%s prompt=%s  out_len(sample)=%s"
          % (d["mode"], d["concurrency"], d["n_requests"], d["prompt_len"],
             d["tokens_per_request"][:1]))
    print("   TTFT mean=%.1f p50=%.1f p99=%.1f ms | wall=%.3fs | gen=%.1f tok/s | total=%.1f tok/s"
          % (t.get("mean", -1), t.get("p50", -1), t.get("p99", -1),
             d["wall_s"], d["gen_throughput_tok_s"], d["total_throughput_tok_s"]))
    print("   extra:", extra)
    for key in ("prefix_cache_metrics_after",):
        if key in d and d[key]:
            short = {k.split("{")[0]: v for k, v in d[key].items()
                     if "hits" in k or "queries" in k or "hit_total" in k or "miss_total" in k}
            print("   %s: %s" % (key, short))
