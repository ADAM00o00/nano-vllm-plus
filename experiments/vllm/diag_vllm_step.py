# -*- coding: utf-8 -*-
"""Why did vLLM offline TTFT collapse to wall time? Inspect step() cadence."""
import time

from vllm import LLM, SamplingParams

MODEL = "/home/adam/models/Qwen3-0.6B"


def main():
    try:
        from vllm import RequestOutputKind
        print("RequestOutputKind:", list(RequestOutputKind))
    except Exception as exc:
        print("RequestOutputKind import failed:", exc)

    sp = SamplingParams(temperature=1.0, max_tokens=16, ignore_eos=True, detokenize=False)
    print("default output_kind   =", sp.output_kind)
    print("default stream_interval =", sp.stream_interval)

    llm = LLM(MODEL, max_model_len=4096, gpu_memory_utilization=0.80,
              max_num_batched_tokens=16384, max_num_seqs=512)
    engine = llm.llm_engine
    print("loaded; sp.output_kind still =", sp.output_kind)

    # does generate() mutate the shared SamplingParams?
    llm.generate([[1000 + i for i in range(32)]] * 2, sp, use_tqdm=False)
    print("after generate(): sp.output_kind =", sp.output_kind)

    prompts = [[2000 + (i * 7) % 1000 for _ in range(32)] for _ in range(4)]
    t0 = time.perf_counter()
    for i, ids in enumerate(prompts):
        engine.add_request(str(i), ids, sp, arrival_time=t0)
    nsteps = 0
    while engine.has_unfinished_requests():
        outs = engine.step()
        nsteps += 1
        if nsteps <= 6 or nsteps % 20 == 0:
            print(f"  step {nsteps:3d} t={1000*(time.perf_counter()-t0):7.1f}ms "
                  f"outs={len(outs)} "
                  f"tok_counts={[len(o.outputs[0].token_ids) if o.outputs else -1 for o in outs][:4]} "
                  f"finished={[o.finished for o in outs][:4]}")
    print(f"loop done in {nsteps} steps, {1000*(time.perf_counter()-t0):.1f} ms")


if __name__ == "__main__":
    main()
