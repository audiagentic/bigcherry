#!/bin/bash
# RR01: build the independent gfx1100 plugin libr11 (tools/lab/radiance-gfx1100/libr11: hand-written native kernels)
# and run it beside libr3 (radiance's own kernels through the compatibility layer) and the reference under
# rad-kbench, for the ops libr11 implements. rad-kbench runs every library that has a row for a case, so the report
# shows the native kernel, the compatibility-layer kernel and libref on the same bytes: correct or not, and time.
# libr3 must have been built first (libr3-build.sh); its build tree is used as it is.
# Run as a queue SCRIPT job (one card):
#   VIS=0 SCRIPT libr11-cmp tools/lab/radiance/libr11-compare.sh @<any build run> <out-dir> [op...]
# The first argument (a llama-server path from the queue) is ignored. Default ops: add mul gemm_nt.
# Usage: libr11-compare.sh <ignored> <out-dir> [op...]
# env: RADIANCE_SRC, WORK, GPU (HIP index, 0), TARGET (gfx1100), SELFTESTS (1 = also run libr11's own ctest), GCC14_BIN
set -u
out=$2; shift 2
mkdir -p "$out"
ops=("$@"); [ ${#ops[@]} -eq 0 ] && ops=(add mul gemm_nt)
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
work=${WORK:-/mnt/data/bigcherry-work/engines/radiance-rdna3}
target=${TARGET:-gfx1100}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
export PATH="$ROCM_PATH/bin:$PATH"
GCC14_BIN=${GCC14_BIN:-/mnt/data/bigcherry-work/toolchains/gcc14/root/usr/bin}
if command -v g++-14 > /dev/null 2>&1; then
    export CC=gcc-14 CXX=g++-14
else
    export CC="$GCC14_BIN/x86_64-linux-gnu-gcc-14" CXX="$GCC14_BIN/x86_64-linux-gnu-g++-14"
fi
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
prefix=$work/radiance-prefix
[ -f "$prefix/.kdev-commit" ] || { echo "NO_PREFIX: run libr3-build.sh or kdev.sh first"; exit 1; }
r3so=$(find "$work/libr3-$target-build" -name 'libr3.so' 2> /dev/null | head -1)
[ -n "$r3so" ] || echo "note: no libr3 build found; comparing libr11 with the reference only"
echo "radiance $(git -C "$src" rev-parse --short HEAD); libr11 at $(git -C "$repo" rev-parse --short HEAD); target $target"

echo "== 1. build libr11"
build=$work/libr11-$target-build
t0=$(date +%s)
if ! cmake -S "$repo/tools/lab/radiance-gfx1100/libr11" -B "$build" -G Ninja -DCMAKE_PREFIX_PATH="$prefix" \
        -DRAD_GPU_TARGETS="$target" > "$out/configure.log" 2>&1; then
    echo "CONFIGURE_FAILED"; grep -vE "^\s*$" "$out/configure.log" | tail -12 | cut -c1-220; exit 1
fi
if ! cmake --build "$build" -j"$(nproc)" -- -k 0 > "$out/build.log" 2>&1; then
    echo "BUILD_FAILED"; grep -E "error:|FAILED:" "$out/build.log" | sed -E 's#.*/([^/:]+:[0-9]+):[0-9]+: #\1 #' | sort | uniq -c | sort -rn | head -20 | cut -c1-220
fi
echo "build in $(( $(date +%s) - t0 )) s"
so=$(find "$build" -name 'libr11.so' | head -1)
[ -n "$so" ] || { echo "NO_PLUGIN (libr11.so was not produced)"; exit 1; }

if [ "${SELFTESTS:-1}" = 1 ]; then
    echo "== 2. libr11's own hardware tests"
    HIP_VISIBLE_DEVICES=${GPU:-0} timeout 600 ctest --test-dir "$build" --output-on-failure > "$out/ctest.log" 2>&1
    grep -E "tests passed|tests failed|Passed|Failed|Not Run|\*\*\*" "$out/ctest.log" | cut -c1-160 | head -8
fi

echo "== 3. rad-kbench: libr11 beside libr3 and the reference, ops: ${ops[*]}"
home="$(dirname "$(dirname "$so")")"
kernels=libr11
[ -n "$r3so" ] && { home="$home:$(dirname "$(dirname "$r3so")")"; kernels=libr11,libr3; }
opts=()
for op in "${ops[@]}"; do opts+=(--op "$op"); done
HIP_VISIBLE_DEVICES=${GPU:-0} RADIANCE_HOME="$home:$src/build/radiance_home" timeout "${KBENCH_TIMEOUT:-1500}" \
    "$src/build/bin/rad-kbench" --kernels "$kernels,libref" "${opts[@]}" --bench \
    --fixture "${FIXTURE:-$src/build/radiance_home/kernels.rkb}" --report "$out/kbench.md" > "$out/kbench.log" 2>&1
echo "rad-kbench --kernels $kernels,libref: exit $?"
grep -E "checked,|memory:|refused|ABI|not loaded" "$out/kbench.log" | cut -c1-220
grep -E "FAIL  " "$out/kbench.log" | sed -E "s/.*FAIL  //; s/[0-9.]+e[+-][0-9]+/N/g" | sort | uniq -c | sort -rn | head -12 | cut -c1-200
[ -f "$out/kbench.md" ] || { echo "NO_REPORT"; tail -8 "$out/kbench.log" | cut -c1-220; exit 1; }
python3 "$here/kdev_report.py" "$out/kbench.md" --json "$out/kbench.json" | cut -c1-230
exit 0
