#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
F=/home/adam/miniconda3/envs/ml/lib/python3.12/site-packages/flash_attn/flash_attn_interface.py

echo "=== 'page_block_size' occurrences (whole file) ==="
grep -n "page_block_size" "$F"

echo
echo "=== does flash_attn_varlen_func accept block_table? ==="
python - <<'PY'
import inspect
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
for fn in (flash_attn_varlen_func, flash_attn_with_kvcache):
    sig = inspect.signature(fn)
    print(f"{fn.__name__}: block_table={'block_table' in sig.parameters}")
    if 'block_table' in sig.parameters:
        print("   ", sig.parameters['block_table'])
PY

echo
echo "=== varlen_func docstring: any 256 / page constraint? ==="
python - <<'PY'
import re
from flash_attn import flash_attn_varlen_func
d = flash_attn_varlen_func.__doc__ or ""
print("mentions 256:", [l.strip() for l in d.splitlines() if "256" in l])
print("mentions block_table:", [l.strip() for l in d.splitlines() if "block_table" in l][:6])
PY

echo
echo "=== which line does 1531 belong to (enclosing def)? ==="
awk 'NR<=1531 && /^def /{ln=NR; name=$0} END{}  NR==1531{print "enclosing def at line " ln ": " name}' "$F"
