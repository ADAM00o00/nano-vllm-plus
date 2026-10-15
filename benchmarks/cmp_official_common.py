# -*- coding: utf-8 -*-
"""Reproduce the official nano-vllm benchmark workload exactly.

From ``bench.py`` (the upstream benchmark the README table is built from):

    seed(0)
    num_seqs = 256
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, 1024))]
                        for _ in range(256)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True,
                                      max_tokens=randint(100, 1024))
                       for _ in range(256)]

The README records **133,966** total output tokens for this workload, which is
``sum(max_tokens)``.  That is an exact invariant: if our reproduction sums to
133,966 we have regenerated the upstream dataset token-for-token, so the numbers
are directly comparable to the published table.

Note the call order matters: per outer iteration the *length* randint runs
first, then that many token randints; the 256 output lengths are drawn last.
"""
from random import randint, seed

OFFICIAL_NUM_SEQS = 256
OFFICIAL_MAX_INPUT_LEN = 1024
OFFICIAL_MAX_OUTPUT_LEN = 1024
OFFICIAL_TOKEN_ID_MAX = 10000
OFFICIAL_MIN_LEN = 100
OFFICIAL_TOTAL_OUTPUT_TOKENS = 133966


def build_official_workload(num_seqs: int = OFFICIAL_NUM_SEQS,
                            max_input_len: int = OFFICIAL_MAX_INPUT_LEN,
                            max_output_len: int = OFFICIAL_MAX_OUTPUT_LEN,
                            seed_value: int = 0):
    """Return (prompt_token_ids, output_lengths), byte-identical to bench.py."""
    seed(seed_value)
    prompts = [
        [randint(0, OFFICIAL_TOKEN_ID_MAX)
         for _ in range(randint(OFFICIAL_MIN_LEN, max_input_len))]
        for _ in range(num_seqs)
    ]
    out_lens = [randint(OFFICIAL_MIN_LEN, max_output_len) for _ in range(num_seqs)]
    return prompts, out_lens


def describe(prompts, out_lens):
    total_out = sum(out_lens)
    total_in = sum(len(p) for p in prompts)
    exact = (len(prompts) == OFFICIAL_NUM_SEQS
             and total_out == OFFICIAL_TOTAL_OUTPUT_TOKENS)
    print(f"[official workload] seqs={len(prompts)} "
          f"prompt_tokens={total_in} (min {min(len(p) for p in prompts)}, "
          f"max {max(len(p) for p in prompts)}, mean {total_in/len(prompts):.1f}) | "
          f"output_tokens={total_out} (min {min(out_lens)}, max {max(out_lens)}, "
          f"mean {total_out/len(out_lens):.1f})", flush=True)
    print(f"[official workload] invariant sum(max_tokens)==133966 : "
          f"{'MATCH ✅' if total_out == OFFICIAL_TOTAL_OUTPUT_TOKENS else 'MISMATCH ❌'}",
          flush=True)
    return {"prompt_tokens": total_in, "output_tokens": total_out,
            "matches_official": bool(exact)}
