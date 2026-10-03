#!/bin/bash
# 1291 small-message threshold sweep (MTP3 verify messages are 20-40 KB).
# Same binary, only BIGCHERRY_AR_CPU_ROOT_MAX_BYTES changes; each value runs cpu-root and an auto control
# (MTP3 draft on the 6900, -ts 4,4,3 ub1024). Args: <llama-server> <out-root>
set -u
bin=$1; root=$2
here=$(cd "$(dirname "$0")" && pwd)
export BIGCHERRY_PATCH_TRACE=1 BIGCHERRY_AR_CPU_ROOT_LARGE_MAX_BYTES=0
for cb in 32768 262144 16384 131072; do
  export BIGCHERRY_AR_CPU_ROOT_MAX_BYTES=$cb
  bash "$here/layout-probe.sh" "$bin" "$root/max-$cb" 'cr-mtp-(cpuroot|auto)-a'
done
