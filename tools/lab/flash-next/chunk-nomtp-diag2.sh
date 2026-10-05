#!/bin/bash
# QFP22 1332 no-MTP crash, round 2: faulting instruction + registers, realloc-debug calibration on the unchunked
# path, and 1327 (host remap) off. Usage: chunk-nomtp-diag2.sh <llama-server> <out-root>
set -u
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
export NO_MTP=1 DECODE_N=64 DEPTH=24576 CTK=f16 CTV=f16
bin=$1 root=$2
mkdir -p "$root"
run() { echo "$1: $(bash "$s" "$bin" "$root/$1" timing 2>&1 | grep -E '^timing: prompt|SERVER_FAILED' | tr '\n' ' ')"; }
printf 'run\nbt 4\nx/14i $pc-40\ninfo registers\n' > "$root/gdb.cmd"
BIGCHERRY_QSA_CHUNK=256 WRAP="gdb -q -batch -x $root/gdb.cmd --args" run gdb2
sed -n '/received signal/,$p' "$root/gdb2/timing.server.log" | cut -c1-200 | head -60
BIGCHERRY_QSA_CHUNK=0 GGML_SCHED_DEBUG_REALLOC=1 run realloc-c0
grep -nE "unexpected graph reallocation" "$root/realloc-c0/timing.server.log" | cut -c1-300 | head -3
BIGCHERRY_QSA_CHUNK=256 BIGCHERRY_QSA_HOST_REMAP=0 run remap-off
