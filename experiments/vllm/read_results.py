import glob
import json
import os

D = os.path.expanduser("~/nano-vllm-plus/results")

print("=== OFFLINE (throughput only) ===")
for f in ("offline_nanovllm.json", "offline_vllm.json"):
    p = os.path.join(D, f)
    if not os.path.exists(p):
        print(f"{f}: MISSING")
        continue
    for r in json.load(open(p)):
        print(f"  {r['engine']:>9} n={r['n_requests']:>3} rep{r['rep']} "
              f"wall={r['wall_s']:7.3f}s gen={r['gen_throughput_tok_s']:8.1f} tok/s "
              f"total={r['total_throughput_tok_s']:8.1f} tok/s "
              f"kv_tokens={r.get('kv_tokens')} out=[{r['tokens_per_request_min']}..{r['tokens_per_request_max']}]")

print()
print("=== ONLINE (TTFT / TPOT / throughput) ===")
rows = {}
for p in sorted(glob.glob(os.path.join(D, "online_*.json"))):
    r = json.load(open(p))
    key = (r["engine"], r["concurrency"])
    rows.setdefault(key, []).append(r)
for (eng, c) in sorted(rows, key=lambda k: (k[0], k[1])):
    for r in sorted(rows[(eng, c)], key=lambda x: x["rep"]):
        t = r["ttft_ms"]; tp = r["tpot_ms"]
        flag = "" if r["n_ttft"] == r["n_requests"] and not r["errors"] else "  <-- INCOMPLETE"
        print(f"  {eng:>9} c={c:>2} rep{r['rep']} "
              f"TTFT mean={t.get('mean', -1):8.1f} p50={t.get('p50', -1):8.1f} p99={t.get('p99', -1):8.1f} ms | "
              f"TPOT mean={tp.get('mean', -1):6.2f} ms | gen={r['gen_throughput_tok_s']:8.1f} tok/s "
              f"| n_ok={r['n_ok']}/{r['n_requests']}{flag}")

print()
print("=== files present ===")
allf = sorted(os.path.basename(x) for x in glob.glob(os.path.join(D, "*.json")))
print(f"  {len(allf)} files")
