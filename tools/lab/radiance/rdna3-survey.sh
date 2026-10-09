#!/bin/bash
# RR01 step 1 and 2: what of radiance's device kernel library (libr4d, written for gfx12) builds and computes
# correctly on an RX 7900 XTX (gfx1100)? No code of ours: libr4d's sources are copied out of tree, its target filter
# is widened to gfx11, and each unit is compiled for gfx1100 with the build continuing past failures.
#   1. compile survey: units that build, units that fail, and the first error of each failing unit;
#   2. rad-info --plugins on the card: which libraries load and whether their device code runs there;
#   3. rad-kbench on the card: with the gfx1100 build of libr4d when it linked (every kernel against libref's
#      recorded answers: correct, wrong or skipped), otherwise with libref alone to show the harness runs there.
# Run as a queue SCRIPT job so the compile never overlaps a timed run:
#   VIS=0,1,2,3 SCRIPT rdna3-survey tools/lab/radiance/rdna3-survey.sh @<any build run> <out-dir>
# The first argument (a llama-server path from the queue) is ignored.
# Usage: rdna3-survey.sh <ignored> <out-dir>
# env: RADIANCE_SRC (/mnt/data/bigcherry-work/engines/radiance, built by build.sh), WORK
#      (/mnt/data/bigcherry-work/engines/radiance-rdna3), GPU (HIP index of an XTX, 0), TARGET (gfx1100), GCC14_BIN
set -u
out=$2
mkdir -p "$out"
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
echo "radiance $(git -C "$src" rev-parse --short HEAD) ($(git -C "$src" log -1 --format=%s)); target $target"

echo "== 1. compile libr4d's units for $target"
copy=$work/r4d-$target-src build=$work/r4d-$target-build
rm -rf "$copy" "$build"
mkdir -p "$copy"
cp -r "$src/libr4d/." "$copy/"
# the one edit: libr4d declares itself gfx12-only; the survey asks what happens when it is not
sed -i 's/rad_plugin_gpu_targets(R4D_GPU_TARGETS "^gfx12")/rad_plugin_gpu_targets(R4D_GPU_TARGETS "^gfx11")/' "$copy/CMakeLists.txt"
grep -q '"^gfx11"' "$copy/CMakeLists.txt" || { echo "TARGET_FILTER_NOT_FOUND in libr4d/CMakeLists.txt"; exit 1; }
if ! cmake -S "$copy" -B "$build" -G Ninja -DCMAKE_PREFIX_PATH="$src/build" -Dradiance_DIR="$src/build" \
        -DRAD_GPU_TARGETS="$target" > "$out/configure.log" 2>&1; then
    echo "CONFIGURE_FAILED"; grep -vE "^\s*$" "$out/configure.log" | tail -12 | cut -c1-220
else
    cmake --build "$build" -j"$(nproc)" -- -k 0 > "$out/build.log" 2>&1
    echo "build exit $?"
    total=$(grep -c '^  "r4d_' "$copy/CMakeLists.txt")
    failed=$(grep -oE "^FAILED: .*r4d_[a-z0-9_]+\.hip" "$out/build.log" | grep -oE "r4d_[a-z0-9_]+\.hip" | sort -u)
    echo "device units: $total listed, $(echo "$failed" | grep -c r4d_) failed to compile"
    for unit in $failed; do
        first=$(grep -m1 -E "$unit:[0-9]+:[0-9]+: (fatal )?error:|/r4d_[a-z0-9_]+\.h:[0-9]+:[0-9]+: (fatal )?error:" <(awk -v u="$unit" '$0 ~ "^FAILED: .*" u {on=1} on && /error:/ {print; exit}' "$out/build.log"))
        echo "  FAIL $unit: $(echo "$first" | sed -E 's#.*/(r4d_[^:]+:[0-9]+):[0-9]+: #\1 #' | cut -c1-190)"
    done
    echo "error kinds: $(grep -oE "error: [^\n]{0,90}" "$out/build.log" | sed -E "s/'[^']*'/X/g" | sort | uniq -c | sort -rn | head -8 | sed 's/^ *//' | tr '\n' '|' | cut -c1-700)"
    echo "plugin: $(find "$build" -name 'libr4d.so' | head -1 || true)"
fi

home="$src/build/radiance_home"
so=$(find "$build" -name 'libr4d.so' 2> /dev/null | head -1)
[ -n "$so" ] && home="$(dirname "$(dirname "$so")"):$home"

echo "== 2. plugins as the engine sees them on HIP device ${GPU:-0}"
HIP_VISIBLE_DEVICES=${GPU:-0} RADIANCE_HOME="$home" "$src/build/bin/rad-info" --plugins > "$out/plugins.txt" 2>&1
grep -vE "^\s*$" "$out/plugins.txt" | head -40 | cut -c1-200

echo "== 3. rad-kbench on HIP device ${GPU:-0}"
kernels=libref; [ -n "$so" ] && kernels=libr4d,libref
HIP_VISIBLE_DEVICES=${GPU:-0} RADIANCE_HOME="$home" timeout 3600 "$src/build/bin/rad-kbench" --kernels "$kernels" \
    --report "$out/kbench.md" --bench > "$out/kbench.log" 2>&1
echo "rad-kbench --kernels $kernels exit $?"
grep -vE "^\s*$" "$out/kbench.log" | tail -25 | cut -c1-200
[ -f "$out/kbench.md" ] && { echo "-- report head"; grep -vE "^\s*$" "$out/kbench.md" | head -60 | cut -c1-200; }
exit 0
