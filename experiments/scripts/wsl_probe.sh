#!/bin/bash
echo "=== repo ==="
ls -la ~/nano-vllm-plus 2>&1 | head -40
echo "=== conda envs ==="
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda env list 2>&1
echo "=== activate ml ==="
conda activate ml 2>&1
which python
python - <<'PY'
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("dev", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
except Exception as e:
    print("torch ERR", e)
try:
    import triton
    print("triton", triton.__version__)
except Exception as e:
    print("triton ERR", e)
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn ERR", e)
PY
echo "=== models ==="
ls -la /home/adam/models 2>&1 | head
echo "=== nvcc ==="
nvcc --version 2>&1 | tail -3
