#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
bash /mnt/c/Users/Adam/PycharmProjects/nano-vllm-plus/.tmp/sync_to_wsl.sh >/dev/null
cd ~/nano-vllm-plus
echo "=== syntax check ==="
for f in benchmarks/cmp_official_common.py benchmarks/cmp_official_offline.py \
         benchmarks/cmp_online_client.py benchmarks/cmp_serve_nanovllm.py; do
  python -c "import ast,sys; ast.parse(open('$f').read()); print('OK  $f')" || echo "FAIL $f"
done
echo
echo "=== argparse check (must not raise) ==="
python benchmarks/cmp_official_offline.py --help >/dev/null && echo "OK  cmp_official_offline --help"
python benchmarks/cmp_online_client.py --help >/dev/null && echo "OK  cmp_online_client --help"
python benchmarks/cmp_serve_nanovllm.py --help >/dev/null && echo "OK  cmp_serve_nanovllm --help"
echo
echo "=== official dataset invariant ==="
python /mnt/c/Users/Adam/PycharmProjects/nano-vllm-plus/.tmp/check_official_dataset.py | head -3
