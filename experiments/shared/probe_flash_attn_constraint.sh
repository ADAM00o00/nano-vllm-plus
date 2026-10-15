#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
FA=$(python -c "import flash_attn, os; print(os.path.dirname(flash_attn.__file__))")
echo "flash_attn dir: $FA"
echo
echo "=== interface file ==="
ls -la "$FA"/flash_attn_interface.py
echo
echo "=== 'multiple of' occurrences ==="
grep -rni "multiple of" "$FA"/flash_attn_interface.py
echo
echo "=== context around page_block_size / block_size 256 ==="
grep -n "256" "$FA"/flash_attn_interface.py
echo
echo "=== docstring for paged KV (block_table) ==="
grep -n -B3 -A12 "block_table: (batch_size, max_num_blocks_per_seq)" "$FA"/flash_attn_interface.py | head -60
