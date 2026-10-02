#!/bin/bash
# Build and run the CPU-root one-shot AllReduce feasibility microbenchmark (2 and 3 ranks; 10/20/40 KB).
# Usage: cpu-root-ar.sh <out-dir>
set -u
out=$1; mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
rocm=${BC_HIP_PATH:-/mnt/vault/tmp/bc-rocm}
"$rocm/bin/hipcc" -O2 -mavx2 --offload-arch=gfx1100,gfx1201 "$here/cpu-root-ar.hip" -o "$out/cpu-root-ar" -pthread 2>&1 | grep -v warning | grep -E "error" ; [ -x "$out/cpu-root-ar" ] || exit 1
for vis in 0,1 0,1,2; do
  for n in 2560 5120 10240; do
    LD_LIBRARY_PATH="$rocm/lib:${LD_LIBRARY_PATH:-}" HIP_VISIBLE_DEVICES=$vis ROCR_VISIBLE_DEVICES=$vis timeout 300 "$out/cpu-root-ar" $n | grep -E "floor|agree|chain|VERIFY|verification"
  done
done
