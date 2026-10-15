#!/bin/bash
# What nsight versions does NVIDIA's own CUDA repo actually carry for wsl-ubuntu?
set -u
TMP=$(mktemp -d)
URL=https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/Packages.gz
echo "fetching $URL"
if curl -fsSL "$URL" -o "$TMP/Packages.gz"; then
  gunzip -c "$TMP/Packages.gz" > "$TMP/Packages"
  echo "index size: $(wc -l < "$TMP/Packages") lines"
  echo "--- packages matching nsight ---"
  grep -E '^Package: .*nsight' "$TMP/Packages" | sort -u
  echo "--- versions of interest ---"
  awk '/^Package: (nsight-systems|nsight-compute|nsight-systems-cli|nsight-compute-target)/{p=$2} /^Version:/{if(p){print p, $2; p=""}}' "$TMP/Packages" | sort -u | tail -20
else
  echo "fetch failed"
fi
echo
echo "--- current apt install progress ---"
du -sh /var/cache/apt/archives/partial 2>/dev/null
ls -la /var/cache/apt/archives/partial/ 2>/dev/null | tail -3
pgrep -a apt-get | head -3 || echo "apt-get not running"
rm -rf "$TMP"
