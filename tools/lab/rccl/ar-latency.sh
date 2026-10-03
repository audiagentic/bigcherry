#!/bin/bash
# Build and run the RCCL small-message AllReduce latency microbenchmark on 2 XTX, and on XTX+XTX+R9700.
# Usage: ar-latency.sh <out-dir>
set -u
out=$1; mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
rocm=${BC_HIP_PATH:-/mnt/vault/tmp/bc-rocm}
"$rocm/bin/hipcc" -O2 --offload-arch=gfx1100,gfx1201 -I"$rocm/include" "$here/ar-latency.hip" -L"$rocm/lib" -lrccl -o "$out/ar-latency" || exit 1
for vis in 0,1 0,1,2; do
  echo "== devices $vis"
  LD_LIBRARY_PATH="$rocm/lib:${LD_LIBRARY_PATH:-}" HIP_VISIBLE_DEVICES=$vis ROCR_VISIBLE_DEVICES=$vis timeout 600 "$out/ar-latency"
done
