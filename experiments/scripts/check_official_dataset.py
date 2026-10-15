import sys
sys.path.insert(0, "/home/adam/nano-vllm-plus")
from benchmarks.cmp_official_common import build_official_workload, describe

prompts, out_lens = build_official_workload()
info = describe(prompts, out_lens)
print("sum(out_lens) =", sum(out_lens))
print("prompt lens first 8 :", [len(x) for x in prompts[:8]])
print("out lens first 8    :", out_lens[:8])
print("prompt[0][:10]      :", prompts[0][:10])
print("max prompt len      :", max(len(x) for x in prompts))
print("min prompt len      :", min(len(x) for x in prompts))
print("max out len         :", max(out_lens))
print("min out len         :", min(out_lens))
