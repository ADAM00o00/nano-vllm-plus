# -*- coding: utf-8 -*-
"""Online (serving) benchmark client for both engines -- official usage only.

* ``--engine nanovllm`` -> gRPC streaming ``Infer`` on the official nano-vllm
  server entry point (``nanovllm.serving.grpc_server.serve``).
* ``--engine vllm``     -> OpenAI-compatible ``/v1/completions`` with SSE
  streaming on an unmodified ``vllm serve``.

Both paths send raw token ids (so no tokenizer differences leak into the
measurement), force an exact output length with ``ignore_eos``, and timestamp
the arrival of the first generated token per request.  ``--concurrency`` is the
number of simultaneously in-flight requests (closed loop); a run is a single
burst so the engine sees all requests at once.

Reported per run: TTFT (mean/p50/p99), TPOT = (e2e - ttft) / (out_len - 1), and
throughput.  ``--rep`` only perturbs the dataset seed, so repetitions never
share prompt prefixes.

    python benchmarks/cmp_online_client.py --engine vllm --target http://localhost:8000 \
        --requests 64 --concurrency 32 --rep 1 --outdir results
"""
import argparse
import itertools
import json
import os
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.cmp_common import (  # noqa: E402
    DEFAULT_OUTPUT_LEN,
    DEFAULT_PROMPT_LEN,
    build_prompts,
    prompt_seed,
    summarize,
)


# ----------------------------------------------------------------------------
# senders
# ----------------------------------------------------------------------------
class NanoSender:
    name = "nanovllm"

    def __init__(self, target):
        from nanovllm.serving import nanovllm_pb2, nanovllm_pb2_grpc
        import grpc
        self._pb2 = nanovllm_pb2
        self._grpc = grpc
        self._pb2_grpc = nanovllm_pb2_grpc
        self.target = target
        self._local = threading.local()

    def _channel(self):
        ch = getattr(self._local, "ch", None)
        if ch is None:
            ch = self._grpc.insecure_channel(self.target)
            self._local.ch = ch
            self._local.stub = self._pb2_grpc.NanoVLLMServiceStub(ch)
        return self._local.ch, self._local.stub

    def send(self, ids, max_tokens, ignore_eos):
        _, stub = self._channel()
        req = self._pb2.InferRequest(
            request_id=str(uuid.uuid4()),
            prompt_token_ids=ids,
            temperature=1.0,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
        )
        t0 = time.perf_counter()
        first = None
        ntok = 0
        for resp in stub.Infer(req):
            if first is None and len(resp.token_ids) > 0:
                first = time.perf_counter() - t0
            ntok += len(resp.token_ids)
            if resp.finished or resp.aborted:
                break
        return first, time.perf_counter() - t0, ntok


class VllmSender:
    name = "vllm"

    def __init__(self, target):
        import requests
        self._requests = requests
        self.base = target.rstrip("/")
        if not self.base.startswith("http"):
            self.base = "http://" + self.base
        with urllib.request.urlopen(self.base + "/v1/models", timeout=30) as r:
            info = json.loads(r.read().decode())
        self.model = info["data"][0]["id"]
        self.url = self.base + "/v1/completions"
        self._local = threading.local()

    def _session(self):
        s = getattr(self._local, "s", None)
        if s is None:
            s = self._requests.Session()
            self._local.s = s
        return s

    def send(self, ids, max_tokens, ignore_eos):
        payload = {
            "model": self.model,
            "prompt": [ids],                      # batch of one token-id prompt
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "ignore_eos": ignore_eos,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        sess = self._session()
        t0 = time.perf_counter()
        first = None
        ntok = 0
        usage_tokens = None
        with sess.post(self.url, json=payload, stream=True, timeout=900) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                data = raw[6:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                u = obj.get("usage")
                if u:
                    usage_tokens = u.get("completion_tokens", usage_tokens)
                for ch in obj.get("choices") or []:
                    txt = ch.get("text") or ""
                    if not txt:
                        continue
                    if first is None:
                        first = time.perf_counter() - t0
                    ntok += 1
        if usage_tokens:
            ntok = usage_tokens
        return first, time.perf_counter() - t0, ntok


# ----------------------------------------------------------------------------
# load driver
# ----------------------------------------------------------------------------
def run_load(sender, prompts, out_lens, concurrency, ignore_eos):
    n = len(prompts)
    concurrency = max(1, min(concurrency, n))
    idx = itertools.count()
    lock = threading.Lock()
    results = [None] * n

    t_start = time.perf_counter()
    barrier = threading.Barrier(concurrency)

    def worker():
        barrier.wait()
        while True:
            with lock:
                try:
                    i = next(idx)
                except StopIteration:
                    return
            if i >= n:
                return
            try:
                first, e2e, ntok = sender.send(prompts[i], out_lens[i], ignore_eos)
                results[i] = (first, e2e, ntok, None)
            except Exception as exc:  # noqa: BLE001
                results[i] = (None, None, 0, f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_end = time.perf_counter()

    errs = [r[3] for r in results if r and r[3]]
    for e in errs[:3]:
        print(f"  request error: {e}", flush=True)
    ok = [r for r in results if r and r[0] is not None]
    ttft = [r[0] for r in ok]
    e2e = [r[1] for r in ok]
    # TPOT: decode time divided by the number of inter-token gaps
    tpot = [max(r[1] - r[0], 0.0) / max(r[2] - 1, 1) for r in ok]
    return {
        "ttft_s": ttft,
        "tpot_s": tpot,
        "e2e_s": e2e,
        "wall_s": t_end - t_start,
        "gen_tokens": sum(r[2] for r in results if r),
        "per_request_tokens": [r[2] for r in results if r],
        "errors": len(errs),
        "n_ok": len(ok),
    }


def scrape_prefix_cache(url):
    if not url:
        return {}
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            text = r.read().decode()
    except Exception as exc:  # noqa: BLE001
        return {"scrape_error": repr(exc)}
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or "prefix_cache" not in line:
            continue
        try:
            name, val = line.rsplit(" ", 1)
            out[name.split("{")[0].strip()] = float(val)
        except ValueError:
            continue
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["nanovllm", "vllm"], required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--metrics-url", default=None)
    p.add_argument("--dataset", choices=["uniform", "official"], default="uniform",
                   help="uniform = fixed 512/128 with fresh random prompts; "
                        "official = bench.py seed-0 workload (256 seqs, 100-1024 in/out)")
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--prompt-len", type=int, default=DEFAULT_PROMPT_LEN)
    p.add_argument("--output-len", type=int, default=DEFAULT_OUTPUT_LEN)
    p.add_argument("--vocab-size", type=int, default=151643)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--warmup-seqs", type=int, default=16,
                   help="warmup size for --dataset official (drawn from seed 1)")
    p.add_argument("--outdir", default=None)
    p.add_argument("--label", default="")
    args = p.parse_args()

    sender = NanoSender(args.target) if args.engine == "nanovllm" else VllmSender(args.target)
    print(f"[{args.engine}] target ready (model={getattr(sender, 'model', '-')})", flush=True)

    before = scrape_prefix_cache(args.metrics_url)

    if args.dataset == "official":
        from benchmarks.cmp_official_common import build_official_workload, describe
        prompts, out_lens = build_official_workload()
        args.requests = len(prompts)
        info = describe(prompts, out_lens)
        # warmup must NOT touch the official prompts (fixed dataset -> prefix
        # cache would carry over).  Use the seed-1 workload instead.
        if args.warmup > 0:
            from random import randint, seed as _seed
            _seed(1)
            nw = args.warmup_seqs
            wpr = [[randint(0, 10000) for _ in range(randint(100, 1024))]
                   for _ in range(nw)]
            wol = [randint(100, 1024) for _ in range(nw)]
            run_load(sender, wpr, wol, min(nw, args.concurrency), True)
            print(f"[{args.engine}] warmup done ({nw} seqs, seed 1)", flush=True)
        total_out_expected = info["output_tokens"]
        prompt_tokens = info["prompt_tokens"]
        dataset_label = "official bench.py (seed 0)"
    else:
        # two warmup rounds at the measured concurrency; engine warmup alone does
        # not cover the real paged prefill path
        if args.warmup > 0:
            nw = max(args.warmup, args.concurrency)
            wc = min(nw, args.concurrency)
            for rnd in range(2):
                ws = prompt_seed(args.engine, f"online-warm-{args.concurrency}", rnd)
                run_load(sender, build_prompts(nw, args.prompt_len, ws, args.vocab_size),
                         [min(args.output_len, 32)] * nw, wc, True)
            print(f"[{args.engine}] warmup done (2 rounds x {nw} reqs @ c={wc})", flush=True)
        seed = prompt_seed(args.engine, "online", args.concurrency) + args.rep
        prompts = build_prompts(args.requests, args.prompt_len, seed, args.vocab_size)
        out_lens = [args.output_len] * len(prompts)
        total_out_expected = sum(out_lens)
        prompt_tokens = sum(len(p) for p in prompts)
        dataset_label = "uniform 512/128, fresh random prompts per run"

    m = run_load(sender, prompts, out_lens, args.concurrency, True)
    after = scrape_prefix_cache(args.metrics_url)

    res = {
        "engine": args.engine,
        "mode": "online",
        "target": args.target,
        "dataset": dataset_label,
        "concurrency": args.concurrency,
        "rep": args.rep,
        "n_requests": len(prompts),
        "prompt_tokens": prompt_tokens,
        "total_output_tokens_expected": total_out_expected,
        "total_output_tokens_actual": m["gen_tokens"],
        "prompt_len_min": min(len(p) for p in prompts),
        "prompt_len_max": max(len(p) for p in prompts),
        "output_len_min": min(out_lens),
        "output_len_max": max(out_lens),
        "ttft_ms": {k: v * 1000 for k, v in summarize(m["ttft_s"]).items() if k != "count"},
        "tpot_ms": {k: v * 1000 for k, v in summarize(m["tpot_s"]).items() if k != "count"},
        "e2e_ms": {k: v * 1000 for k, v in summarize(m["e2e_s"]).items() if k != "count"},
        "wall_s": m["wall_s"],
        "gen_tokens": m["gen_tokens"],
        "gen_throughput_tok_s": m["gen_tokens"] / m["wall_s"],
        "total_throughput_tok_s": (m["gen_tokens"] + prompt_tokens) / m["wall_s"],
        "tokens_per_request_min": min(m["per_request_tokens"]) if m["per_request_tokens"] else 0,
        "tokens_per_request_max": max(m["per_request_tokens"]) if m["per_request_tokens"] else 0,
        "errors": m["errors"],
        "n_ok": m["n_ok"],
        "n_ttft": len(m["ttft_s"]),
        "prefix_cache_before": before,
        "prefix_cache_after": after,
    }
    tag = args.label or f"{args.engine}-online-c{args.concurrency}"
    print(f"RESULT {tag}-rep{args.rep} " + json.dumps({
        "conc": args.concurrency,
        "total_output_tokens": res["total_output_tokens_actual"],
        "wall_s": round(res["wall_s"], 3),
        "throughput_tok_s": round(res["gen_throughput_tok_s"], 1),
        "ttft_mean_ms": round(res["ttft_ms"].get("mean", -1), 1),
        "tpot_mean_ms": round(res["tpot_ms"].get("mean", -1), 2),
        "errors": m["errors"]}), flush=True)

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        path = os.path.join(args.outdir, f"online_{args.engine}_c{args.concurrency}_rep{args.rep}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"wrote {path}", flush=True)

    if m["errors"]:
        print(f"WARNING: {m['errors']} request(s) failed", flush=True)
    if len(m["ttft_s"]) != args.requests:
        print(f"WARNING: only {len(m['ttft_s'])}/{args.requests} produced a first token", flush=True)


if __name__ == "__main__":
    main()
