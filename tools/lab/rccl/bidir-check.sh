#!/bin/bash
# Build and run the per-GPU unidirectional vs bidirectional pinned-host copy bandwidth check (all 4 GPUs).
set -u
out=$1; mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
rocm=${BC_HIP_PATH:-/mnt/vault/tmp/bc-rocm}
"$rocm/bin/hipcc" -O2 --offload-arch=gfx1100,gfx1201,gfx1030 "$here/bidir-check.hip" -o "$out/bidir-check" 2>&1 | grep -E "error"
HIP_VISIBLE_DEVICES=0,1,2,3 ROCR_VISIBLE_DEVICES=0,1,2,3 "$out/bidir-check"
