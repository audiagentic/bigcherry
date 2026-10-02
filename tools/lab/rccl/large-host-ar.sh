#!/bin/bash
# Build and run the pipelined host-staged large-message AllReduce prototype (10 MB, 2 and 3 ranks,
# chunk 256 KB..4 MB, 2/4/8 CPU reducer threads). Usage: large-host-ar.sh <out-dir>
set -u
out=$1; mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
rocm=${BC_HIP_PATH:-/mnt/vault/tmp/bc-rocm}
"$rocm/bin/hipcc" -O2 -mavx2 --offload-arch=gfx1100,gfx1201 "$here/large-host-ar.hip" -o "$out/large-host-ar" -pthread 2>&1 | grep -E "error"
[ -x "$out/large-host-ar" ] || exit 1
for vis in 0,1,2 0,1; do
  for chunk in 65536 131072 262144 524288; do
    for t in 2 3 4 6; do
      LD_LIBRARY_PATH="$rocm/lib:${LD_LIBRARY_PATH:-}" HIP_VISIBLE_DEVICES=$vis ROCR_VISIBLE_DEVICES=$vis timeout 120 "$out/large-host-ar" 2621440 $chunk $t ${MODE:-pool}
    done
  done
done
