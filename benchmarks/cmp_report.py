# -*- coding: utf-8 -*-
"""Aggregate the nano-vllm vs vLLM comparison results into markdown tables.

    python benchmarks/cmp_report.py --dir results --md
    python benchmarks/cmp_report.py --dir results --md --out docs/cmp_vllm_vs_nanovllm.md
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(dirpath):
    rows = []
    for path in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        if os.path.basename(path).startswith("smoke_"):
            continue
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        obj["_file"] = os.path.basename(path)
        rows.append(obj)
    return rows


def fnum(v, nd=1):
    return "n/a" if v is None else f"{v:.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results")
    p.add_argument("--out", default=None)
    p.add_argument("--md", action="store_true")
    args = p.parse_args()

    rows = load(args.dir)
    if not rows:
        print(f"no results in {args.dir}", file=sys.stderr)
        return 1

    out = []
    out.append("| engine | mode | conc | reqs | TTFT mean (ms) | TTFT p50 | TTFT p99 | "
               "wall (s) | gen tok/s | total tok/s |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")

    def sort_key(r):
        return (r["mode"], r["concurrency"], r["engine"])

    for r in sorted(rows, key=sort_key):
        t = r["ttft_ms"]
        out.append(
            f"| {r['engine']} | {r['mode']} | {r['concurrency']} | {r['n_requests']} | "
            f"{fnum(t.get('mean'))} | {fnum(t.get('p50'))} | {fnum(t.get('p99'))} | "
            f"{fnum(r['wall_s'], 3)} | {fnum(r['gen_throughput_tok_s'])} | "
            f"{fnum(r['total_throughput_tok_s'])} |"
        )

    # head-to-head pairs
    out.append("")
    out.append("| mode | conc | TTFT ratio (ours/vLLM) | gen tok/s ours | gen tok/s vLLM | "
               "gen throughput ratio (ours/vLLM) |")
    out.append("|---|---|---|---|---|---|")
    by = {(r["mode"], r["concurrency"], r["engine"]): r for r in rows}
    keys = sorted({(m, c) for (m, c, _e) in by})
    for m, c in keys:
        a, b = by.get((m, c, "nanovllm")), by.get((m, c, "vllm"))
        if not a or not b:
            continue
        ta, tb = a["ttft_ms"].get("mean"), b["ttft_ms"].get("mean")
        ga, gb = a["gen_throughput_tok_s"], b["gen_throughput_tok_s"]
        out.append(
            f"| {m} | {c} | {fnum(ta / tb if ta and tb else None, 2)} | "
            f"{fnum(ga)} | {fnum(gb)} | {fnum(ga / gb if gb else None, 2)} |"
        )

    text = "\n".join(out)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
