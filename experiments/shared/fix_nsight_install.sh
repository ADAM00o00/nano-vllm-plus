#!/bin/bash
# Replace the slow/cuda-12.4 multiverse nsight packages with the current ones
# from NVIDIA's own repo (which matches CUDA 13.0).
set -u
export DEBIAN_FRONTEND=noninteractive

echo "=== stop the in-flight multiverse install ==="
pkill -f 'apt-get install' 2>/dev/null
sleep 3
pkill -9 -f 'apt-get install' 2>/dev/null
sleep 2
dpkg --configure -a 2>&1 | tail -3
apt-get clean

echo
echo "=== candidates from NVIDIA repo ==="
apt-cache policy nsight-systems-2026.1.3 nsight-compute-2026.2.1 \
    cuda-nsight-systems-13-0 cuda-nsight-compute-13-0 2>&1 | grep -E '^[a-z]|Candidate' | head -20

echo
echo "=== simulate install (no recommends) ==="
apt-get install -s --no-install-recommends nsight-systems-2026.1.3 nsight-compute-2026.2.1 2>&1 \
  | grep -E '^(Inst|E:|[0-9]+ upgraded)' | head -40
