#!/bin/bash
# RR07: weight bandwidth of the decode GEMMs, from radiance's own timing mode (r4d_selftest --perf N K M): each
# kernel's time over a weight replicated past the card's caches, as GB/s of the weight stream. The number is
# comparable with a plain stream of the same bytes, which is what says how much of a card's memory bandwidth a
# kernel design uses.
# libr3 must have been built first (libr3-build.sh).
# Run as a queue SCRIPT job (one card):
#   VIS=0 SCRIPT r3-perf tools/lab/radiance/libr3-perf.sh @<any build run> <out-dir> [mode...]
# The first argument (a llama-server path from the queue) is ignored. A mode is --perf, --perfmxfp4, --perfbf16, --perfw8, ...
# (default --perf). Shapes: the 27B's gate/up, down and attention projections at M = 1.
# Usage: libr3-perf.sh <ignored> <out-dir> [mode...]
# env: GPU (HIP index, 0), TARGET (gfx1100), PLUGIN (a .so to time instead of libr3, e.g. libr4d on the gfx12 card),
#      SHAPES ("N K M" separated by ';'), ITERS (50), R3_BT/R3_BT_B/R3_BT_NT for --perfmxfp4, RADIANCE_SRC, WORK
set -u
out=$2; shift 2
mkdir -p "$out"
modes=("$@"); [ ${#modes[@]} -eq 0 ] && modes=(--perf)
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
work=${WORK:-/mnt/data/bigcherry-work/engines/radiance-rdna3}
target=${TARGET:-gfx1100}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
export PATH="$ROCM_PATH/bin:$PATH"
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
plugin=${PLUGIN:-$(find "$work/libr3-$target-build" -name 'libr3.so' 2> /dev/null | head -1)}
[ -n "$plugin" ] && [ -f "$plugin" ] || { echo "NO_PLUGIN: run libr3-build.sh first, or set PLUGIN"; exit 1; }
if [ -n "${PLUGIN:-}" ]; then
    selftest=$src/build/bin/r4d_selftest
else
    # always rebuilt (incremental): an r3_selftest left by an earlier build does not know modes added since
    {
        GCC14_BIN=${GCC14_BIN:-/mnt/data/bigcherry-work/toolchains/gcc14/root/usr/bin}
        command -v g++-14 > /dev/null 2>&1 || export PATH="$GCC14_BIN:$PATH"
        cmake --build "$work/libr3-$target-build" --target r3_selftest > "$out/selftest-build.log" 2>&1 || {
            echo "SELFTEST_BUILD_FAILED"; grep -E "error|Error" "$out/selftest-build.log" | head -12 | cut -c1-220; exit 1;
        }
        selftest=$(find "$work/libr3-$target-build" -name r3_selftest -type f | head -1)
    }
fi
[ -x "$selftest" ] || { echo "NO_SELFTEST: $selftest"; exit 1; }
echo "radiance $(git -C "$src" rev-parse --short HEAD); plugin $plugin; HIP device ${GPU:-0}"
IFS=';' read -ra shapes <<< "${SHAPES:-17408 5120 1;5120 8704 1;8192 5120 1;17408 5120 8}"
for mode in "${modes[@]}"; do
    if [ "$mode" = --perfhc ]; then
        echo "== $mode M=${HC_M:-1} iters=${ITERS:-50}"
        HIP_VISIBLE_DEVICES=${GPU:-0} timeout "${CASE_TIMEOUT:-1800}" "$selftest" "$plugin" "$mode" \
            "${HC_M:-1}" "${ITERS:-50}" 2>&1 | tee "$out/perfhc-M${HC_M:-1}.log"
        rc=${PIPESTATUS[0]}
        [ "$rc" -eq 0 ] || { echo "PERFHC_FAILED rc=$rc"; exit "$rc"; }
        continue
    fi
    if [ "$mode" = --perfmoe ]; then
        echo "== $mode M=${MOE_M:-2048} experts=${MOE_EXPERTS:-512} topk=${MOE_TOPK:-10} pool=${MOE_POOL_PCT:-0} iters=${ITERS:-20}"
        HIP_VISIBLE_DEVICES=${GPU:-0} timeout "${CASE_TIMEOUT:-1800}" "$selftest" "$plugin" "$mode" \
            "${MOE_M:-2048}" "${ITERS:-20}" "${MOE_EXPERTS:-512}" "${MOE_TOPK:-10}" "${MOE_POOL_PCT:-0}" 2>&1 \
            | tee "$out/perfmoe-M${MOE_M:-2048}.log"
        rc=${PIPESTATUS[0]}
        [ "$rc" -eq 0 ] || { echo "PERFMOE_FAILED rc=$rc"; exit "$rc"; }
        continue
    fi
    if [ "$mode" = --perfwmma ]; then
        echo "== $mode ${ITERS:-8192}"
        HIP_VISIBLE_DEVICES=${GPU:-0} timeout 900 "$selftest" "$plugin" "$mode" "${ITERS:-8192}" 2>&1 \
            | tee "$out/perfwmma.log"
        continue
    fi
    for shape in "${shapes[@]}"; do
        echo "== $mode $shape"
        # shellcheck disable=SC2086
        HIP_VISIBLE_DEVICES=${GPU:-0} timeout 600 "$selftest" "$plugin" "$mode" $shape "${ITERS:-50}" 2>&1 \
            | grep -E "GB/s|error|FAIL|usage" | cut -c1-150
    done
done
