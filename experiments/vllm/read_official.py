import glob
import json
import os

D = os.path.expanduser("~/nano-vllm-plus/results_official")
print(f"{'file':42s} {'tokens':>8s} {'wall_s':>8s} {'tok/s':>9s} {'TTFT ms':>9s} {'TPOT ms':>8s} {'ok':>6s}")
print("-" * 100)
for p in sorted(glob.glob(os.path.join(D, "*.json"))):
    r = json.load(open(p))
    name = os.path.basename(p).replace(".json", "")
    tok = r.get("actual_output_tokens") or r.get("total_output_tokens_actual")
    wall = r["wall_s"]
    tps = r.get("throughput_tok_s") or r.get("gen_throughput_tok_s")
    ttft = r.get("ttft_ms", {}).get("mean")
    tpot = r.get("tpot_ms", {}).get("mean")
    ok = f"{r.get('n_ok', '-')}/{r.get('n_requests')}"
    err = r.get("errors", 0)
    print(f"{name:42s} {tok:>8d} {wall:>8.2f} {tps:>9.1f} "
          f"{(f'{ttft:.1f}' if ttft else '-'):>9s} {(f'{tpot:.2f}' if tpot else '-'):>8s} {ok:>6s}"
          + (f"  ERRORS={err}" if err else ""))

print()
print("=== detail ===")
for p in sorted(glob.glob(os.path.join(D, "*.json"))):
    r = json.load(open(p))
    name = os.path.basename(p).replace(".json", "")
    print(f"\n{name}")
    for k in ("dataset", "n_requests", "prompt_tokens", "total_output_tokens",
              "actual_output_tokens", "total_output_tokens_expected",
              "total_output_tokens_actual", "wall_s", "throughput_tok_s",
              "gen_throughput_tok_s", "concurrency", "num_kvcache_blocks",
              "num_gpu_blocks", "kv_tokens", "block_size", "errors", "n_ok", "n_ttft"):
        if k in r:
            print(f"   {k} = {r[k]}")
    for k in ("ttft_ms", "tpot_ms", "e2e_ms"):
        if k in r:
            print(f"   {k} = {r[k]}")
