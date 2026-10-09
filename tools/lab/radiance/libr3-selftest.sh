#!/bin/bash
# RR06: numeric check of libr3's kernels with radiance's own kernel selftest (r4d_selftest), which carries a host
# reference for each row and draws its own operands. This is the check for the rows rad-kbench cannot reach with a
# recorded fixture, above all the MXFP4 GEMMs (rad-kbench skips them for the MXFP4 container). The selftest dlopens
# the plugin it is given, so it runs libr3 on a gfx1100 card exactly as it runs libr4d on a gfx12 one.
# libr3 must have been built first (libr3-build.sh).
# Run as a queue SCRIPT job (one card):
#   VIS=0 SCRIPT r3-self tools/lab/radiance/libr3-selftest.sh @<any build run> <out-dir> [row...]
# The first argument (a llama-server path from the queue) is ignored. Default rows: the three MXFP4 GEMMs.
# Usage: libr3-selftest.sh <ignored> <out-dir> [row...]
# env: GPU (HIP index, 0), TARGET (gfx1100), PLUGIN (a .so to test instead of libr3, e.g. libr4d on the gfx12 card),
#      CASE_TIMEOUT (900 s), RADIANCE_SRC, WORK
set -u
out=$2; shift 2
mkdir -p "$out"
rows=("$@"); [ ${#rows[@]} -eq 0 ] && rows=(gemm_mxfp4a8_decode gemm_mxfp4a8_nt_m64 gemm_mxfp4a8_tiled)
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
work=${WORK:-/mnt/data/bigcherry-work/engines/radiance-rdna3}
target=${TARGET:-gfx1100}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
export PATH="$ROCM_PATH/bin:$PATH"
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
plugin=${PLUGIN:-$(find "$work/libr3-$target-build" -name 'libr3.so' 2> /dev/null | head -1)}
[ -n "$plugin" ] && [ -f "$plugin" ] || { echo "NO_PLUGIN: run libr3-build.sh first, or set PLUGIN"; exit 1; }
selftest=$src/build/bin/r4d_selftest
[ -x "$selftest" ] || { echo "NO_SELFTEST: $selftest"; exit 1; }
echo "radiance $(git -C "$src" rev-parse --short HEAD); plugin $plugin; HIP device ${GPU:-0}"
bad=0
for row in "${rows[@]}"; do
    HIP_VISIBLE_DEVICES=${GPU:-0} timeout "${CASE_TIMEOUT:-900}" "$selftest" "$plugin" --case "$row" > "$out/$row.log" 2>&1
    rc=$?
    pass=$(grep -c "^  ok  \|^  PASS\|^  pass" "$out/$row.log")
    fail=$(grep -c "FAIL" "$out/$row.log")
    echo "-- $row: exit $rc, $(grep -cE "^\s+(ok|PASS|pass)\b" "$out/$row.log") passed, $fail failed"
    grep -E "FAIL" "$out/$row.log" | head -8 | cut -c1-230
    tail -2 "$out/$row.log" | cut -c1-200
    [ $rc -ne 0 ] && bad=1
done
exit $bad
