#!/bin/bash
# Flash-Next n-gram gather from a non-rank GPU. Builds/tests it, then runs TP=2 disk vs device.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
here=tools/lab/radiance
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
work=${WORK:-/mnt/data/bigcherry-work/engines/radiance-rdna3}
target=${TARGET:-gfx1100}
tag=${1:-fn-ngram-dev}
devs=${R3_NGRAM_DEV:-2}
dev=${devs%%,*}
vis=${VIS:-0,1,2}
out=${R:-/mnt/data/bigcherry-work/runs}/$tag-lab
mkdir -p "$out"

"$here/libr3-rewrite-check.sh"
KBENCH=0 "$here/libr3-build.sh" /dev/null "$out/build"
build=$work/libr3-$target-build
cmake --build "$build" --target r3_selftest r3_ngram_test > "$out/selftest-build.log" 2>&1 || {
    echo "R3_NGRAM_BUILD_FAILED"; grep -E "error:|Error" "$out/selftest-build.log" | head -20 | cut -c1-220; exit 1;
}
helper=$(find "$build" -path "*/helpers/libr3_ngram_helper.so" -type f | head -1)
test=$(find "$build" -name r3_ngram_test -type f | head -1)
[ -n "$helper" ] && [ -f "$helper" ] || { echo "R3_NGRAM_NO_HELPER"; exit 1; }
[ -n "$test" ] && [ -x "$test" ] || { echo "R3_NGRAM_NO_TEST"; exit 1; }

HIP_VISIBLE_DEVICES="$vis" "$test" "$helper" --selftest "$dev" | tee "$out/selftest.log"
HIP_VISIBLE_DEVICES="$vis" "$test" "$helper" --perf "$dev" 1 100 | tee "$out/perf-M1.log"
HIP_VISIBLE_DEVICES="$vis" "$test" "$helper" --perf "$dev" 2048 30 | tee "$out/perf-M2048.log"

avx=$(find "$src/build/radiance_home" -name libavx.so -type f | head -1)
[ -n "$avx" ] && [ -f "$avx" ] || { echo "R3_NGRAM_NO_LIBAVX"; exit 1; }

echo "== disk/libavx"
env -u R3_NGRAM_DEV KERNELS=libr3,libavx,libref PROFILE=1 \
    bash "$here/flashnext-tp2.sh" "$tag-off"

echo "== non-rank device"
R3_NGRAM_DEV="$devs" R3_NGRAM_HELPER_SO="$helper" R3_NGRAM_AVX="$avx" \
VIS="$vis" VISIBLE_GPU="$vis" KERNELS=libr3,libavx,libref PROFILE=1 \
    bash "$here/flashnext-tp2.sh" "$tag-on"

echo "R3_NGRAM_DONE off=$tag-off on=$tag-on"
