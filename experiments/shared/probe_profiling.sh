#!/bin/bash
# Probe what profiling tooling is available / installable in WSL.
echo "=== existing tools ==="
for t in nsys ncu nvprof ncu-ui nsys-ui; do
  p=$(command -v $t 2>/dev/null) && echo "$t -> $p" || echo "$t -> (absent)"
done
ls -d /opt/nvidia/nsight* 2>/dev/null || echo "no /opt/nvidia/nsight*"
ls -d /usr/local/cuda*/nsight* 2>/dev/null || echo "no cuda nsight dir"

echo
echo "=== os / driver / cuda ==="
. /etc/os-release && echo "$PRETTY_NAME"
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
ls -l /usr/local/cuda | head -3
/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -2

echo
echo "=== disk / mem ==="
df -h / /home 2>/dev/null | sed -n '1,5p'
free -g | sed -n '1,3p'

echo
echo "=== apt sources / nvidia repo ==="
grep -rhs -E '^deb ' /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null | sed 's/\[[^]]*\] //' | sort -u | head -20
echo "--- cuda-keyring? ---"
dpkg -l 2>/dev/null | grep -iE 'cuda-keyring|nvidia-driver|nsight' | head -10 || echo none

echo
echo "=== sudo ==="
sudo -n true 2>&1 && echo "passwordless sudo OK" || echo "sudo needs password (or unavailable)"

echo
echo "=== apt candidates ==="
apt-cache policy nsight-systems nsight-compute 2>&1 | head -20

echo
echo "=== profiling permission knob ==="
cat /proc/driver/nvidia/params 2>/dev/null | grep -i restrict || echo "cannot read driver params"
echo "--- wsl gpu passthrough ---"
ls -l /dev/dxg 2>/dev/null || echo "/dev/dxg absent"
ls /usr/lib/wsl/lib/ 2>/dev/null | grep -iE 'cuda|nvidia|cupti' | head -10
