#!/bin/bash
# Probe WSL conda ml env for the vLLM vs nano-vllm comparison.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml
echo "=== python ==="
python -V
echo "=== vllm ==="
python -c "import vllm, sys; print('vllm', vllm.__version__, vllm.__file__)" 2>&1 | tail -5
echo "=== torch / cuda ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))" 2>&1 | tail -5
echo "=== vllm CLI ==="
which vllm
vllm --version 2>&1 | tail -3
echo "=== transformers ==="
python -c "import transformers; print('transformers', transformers.__version__)" 2>&1 | tail -2
echo "=== models ==="
ls -d /home/adam/models/*/ 2>/dev/null
echo "=== HF cache ==="
ls -d ~/.cache/huggingface/hub/*/ 2>/dev/null | head -20
echo "=== HF env ==="
env | grep -i -E "HF_|HUGGING|VLLM" || echo "(none)"
echo "=== repo copy ==="
ls ~/nano-vllm-plus/ 2>/dev/null | head
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv 2>&1 | head -5
echo "=== disk ==="
df -h ~ | tail -1
