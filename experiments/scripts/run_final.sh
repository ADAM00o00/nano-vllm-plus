#!/bin/bash
# Final Phase 1 verification after all edits.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
cd ~/nano-vllm-plus

echo "############ decode-attention unit tests ############"
python -m unittest tests.test_decode_attention 2>&1 | tail -5
echo
echo "############ full suite ############"
python -m unittest discover -s tests 2>&1 | tail -5
echo
echo "############ benchmark smoke (markdown + L2 note + sweeps parse) ############"
python benchmarks/benchmark_decode_attention.py \
    --matrix "batch=1,8 ctx=2048" --check --markdown /tmp/final_table.md \
    --warmup 10 --rep 40 2>&1 | tail -14
echo
echo "--- markdown output ---"
cat /tmp/final_table.md
echo
echo "############ attention.py backend switch ############"
NANOVLLM_DECODE_ATTN=bogus python -c "import nanovllm.layers.attention" 2>&1 | tail -2
NANOVLLM_DECODE_ATTN=triton python -c "
import nanovllm.layers.attention as a
print('triton ->', a.DECODE_ATTENTION_BACKEND)
" 2>&1 | tail -1
python -c "
import nanovllm.layers.attention as a
print('default ->', a.DECODE_ATTENTION_BACKEND)
" 2>&1 | tail -1
