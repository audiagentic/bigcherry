#!/bin/bash
# 1291 large-path chunk sweep (GPT diagnosis: per-chunk stream memops, not the CPU sum, cost prefill).
# Same binary, only BIGCHERRY_AR_CPU_ROOT_CHUNK_BYTES changes; each size runs cpu-root and an auto control
# (no MTP, -ts 4,4,3 ub1024). Args: <llama-server> <out-root>
set -u
bin=$1; root=$2
here=$(cd "$(dirname "$0")" && pwd)
export BIGCHERRY_PATCH_TRACE=1 BIGCHERRY_AR_CPU_ROOT_LARGE_MAX_BYTES=33554432
for cb in 8388608 2097152 4194304 1048576; do
  export BIGCHERRY_AR_CPU_ROOT_CHUNK_BYTES=$cb
  bash "$here/layout-probe.sh" "$bin" "$root/chunk-$cb" 'cr-nomtp-(cpuroot|auto)-a'
done
