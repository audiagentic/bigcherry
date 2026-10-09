#!/bin/bash
# RR01: build libr3 (radiance's kernels for RDNA3, from libr4d's sources through engines/radiance/kernels/libr3's
# compatibility layer) and check it on one card with rad-kbench against the reference.
#   1. build: units built, units left out, compile errors by unit if any;
#   2. rad-info --plugins: how the engine sees libr3 on the card;
#   3. rad-kbench --kernels libr3,libref: every row libr3 offers, correct or not, with timings (kdev_report.py).
# Run as a queue SCRIPT job (one card is enough):
#   VIS=0 SCRIPT libr3-build tools/lab/radiance/libr3-build.sh @<any build run> <out-dir> [op...]
# The first argument (a llama-server path from the queue) is ignored. With ops given, only those are checked.
# Usage: libr3-build.sh <ignored> <out-dir> [op...]
# env: RADIANCE_SRC, WORK, GPU (HIP index, 0), TARGET (gfx1100), R3_ONLY (semicolon list: compile only these units,
#      no bench), KBENCH (1; 0 = build only), AGAINST (earlier kbench.json), GCC14_BIN
set -u
out=$2; shift 2
mkdir -p "$out"
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
[ -x "$src/build/bin/rad-kbench" ] || { echo "NO_RADIANCE_BUILD under $src (run build.sh first)"; exit 1; }
echo "radiance $(git -C "$src" rev-parse --short HEAD); libr3 at $(git -C "$repo" rev-parse --short HEAD); target $target"

prefix=$work/radiance-prefix
stamp=$(git -C "$src" rev-parse HEAD)
if [ "$(cat "$prefix/.kdev-commit" 2> /dev/null)" != "$stamp" ]; then
    rm -rf "$prefix"
    cmake --install "$src/build" --prefix "$prefix" > "$out/install.log" 2>&1 || { echo "INSTALL_FAILED"; tail -5 "$out/install.log"; exit 1; }
    echo "$stamp" > "$prefix/.kdev-commit"
fi

echo "== 1. build"
build=$work/libr3-$target-build${R3_ONLY:+-only}
t0=$(date +%s)
if ! cmake -S "$repo/engines/radiance/kernels/libr3" -B "$build" -G Ninja -DCMAKE_PREFIX_PATH="$prefix" \
        -DRADIANCE_SRC="$src" -DR3_TARGETS="$target" -DR3_ONLY="${R3_ONLY:-}" > "$out/configure.log" 2>&1; then
    echo "CONFIGURE_FAILED"; grep -vE "^\s*$" "$out/configure.log" | tail -12 | cut -c1-220; exit 1
fi
grep "libr3:" "$out/configure.log" | cut -c1-160
cmake --build "$build" -j"$(nproc)" -- -k 0 > "$out/build.log" 2>&1
rc=$?
echo "build exit $rc in $(( $(date +%s) - t0 )) s"
grep "r3_gen_stubs:" "$out/build.log" | tail -1
if [ $rc -ne 0 ]; then
    for unit in $(grep -oE "^FAILED: .*units/r4d_[a-z0-9_]+\.hip" "$out/build.log" | grep -oE "r4d_[a-z0-9_]+\.hip" | sort -u); do
        echo "  FAIL $unit: $(awk -v u="$unit" '$0 ~ "^FAILED: .*" u {on=1} on && /error:/ {print; exit}' "$out/build.log" | sed -E 's#.*/([^/:]+:[0-9]+):[0-9]+: #\1 #' | cut -c1-200)"
    done
    grep -E "error:|undefined" "$out/build.log" | grep -v "units/r4d_" | sort | uniq -c | sort -rn | head -6 | cut -c1-220
    exit 1
fi
so=$(find "$build" -name 'libr3.so' | head -1)
[ -n "$so" ] || { echo "NO_PLUGIN"; exit 1; }
[ "${KBENCH:-1}" = 1 ] && [ -z "${R3_ONLY:-}" ] || exit 0
home="$(dirname "$(dirname "$so")"):$src/build/radiance_home"

echo "== 2. the engine's view on HIP device ${GPU:-0}"
HIP_VISIBLE_DEVICES=${GPU:-0} RADIANCE_HOME="$home" "$src/build/bin/rad-info" --plugins > "$out/plugins.txt" 2>&1
grep -E "device:|libr3|refus|ABI|error" "$out/plugins.txt" | cut -c1-220

echo "== 3. rad-kbench --kernels libr3,libref"
opts=()
for op in "$@"; do opts+=(--op "$op"); done
HIP_VISIBLE_DEVICES=${GPU:-0} RADIANCE_HOME="$home" timeout "${KBENCH_TIMEOUT:-2400}" "$src/build/bin/rad-kbench" \
    --kernels libr3,libref "${opts[@]}" --bench --fixture "${FIXTURE:-$src/build/radiance_home/kernels.rkb}" \
    --report "$out/kbench.md" > "$out/kbench.log" 2>&1
echo "rad-kbench exit $?"
grep -E "checked,|memory:|declared kernel row|refused|ABI|error" "$out/kbench.log" | cut -c1-220
[ -f "$out/kbench.md" ] || { echo "NO_REPORT"; tail -8 "$out/kbench.log" | cut -c1-220; exit 1; }
python3 "$here/kdev_report.py" "$out/kbench.md" --json "$out/kbench.json" ${AGAINST:+--against "$AGAINST"} > "$out/table.txt" 2>&1
echo "table rc=$? ($(grep -c "" "$out/table.txt") lines in $out/table.txt)"
sed -n '/^== per kernel/,$p' "$out/table.txt" | cut -c1-200 | head -80
exit 0
