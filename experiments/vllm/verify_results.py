import glob
import json

print("=== integrity checks ===")
for path in sorted(glob.glob("results/*.json")):
    if "smoke" in path:
        continue
    d = json.load(open(path))
    tr = d["tokens_per_request"]
    plens = (d["prompt_len_min"], d["prompt_len_max"])
    errs = d.get("errors", 0)
    n_ok = d.get("n_ok", d["ttft_count"])
    pfx = ""
    after = d.get("prefix_cache_metrics_after") or {}
    if after:
        hits = sum(v for k, v in after.items() if k.endswith("hit_total") or "hits_total" in k)
        miss = sum(v for k, v in after.items() if k.endswith("miss_total") or "queries_total" in k)
        pfx = f" prefix_hits={hits:.0f} prefix_queries={miss:.0f}"
    cached_blocks = d.get("prefix_cache_blocks_max")
    cached_tok = d.get("prefix_cache_tokens_max_at_first_token")
    print(f"{path:34s} prompt_len={plens} out_tokens=[{tr[0]}..{tr[-1]}] "
          f"ttft_n={d['ttft_count']}/{d['n_requests']} ok={n_ok} errors={errs}"
          f" | cached_blocks={cached_blocks} cached_tokens={cached_tok}{pfx}")

print()
print("=== decode-only rate (excluding prefill/TTFT) ===")
for path in sorted(glob.glob("results/*.json")):
    if "smoke" in path:
        continue
    d = json.load(open(path))
    gen = d["gen_tokens"]
    n = d["n_requests"]
    ttft_mean = d["ttft_ms"]["mean"] / 1000.0
    wall = d["wall_s"]
    dec_tokens = max(0, gen - n)
    dec_time = wall - ttft_mean
    print(f"{path:34s} gen={gen} wall={wall:.3f}s ttft={ttft_mean*1000:.0f}ms "
          f"-> decode-only {dec_tokens/max(dec_time,1e-9):8.1f} tok/s "
          f"(decode {dec_time:.3f}s, prefill-ish {ttft_mean:.3f}s)")
