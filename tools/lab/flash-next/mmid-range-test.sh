#!/bin/bash
# MET02 / 1281 phase A: build and run the range-op reference test (tests/test-mul-mat-id-range.cpp) in a build tree,
# on the CPU backend. The test compares ggml_mul_mat_id_range bit for bit with the ordinary op on hand-translated ids
# and requires an exact +0 on every out-of-range lane.
# Usage: mmid-range-test.sh <llama-server of the build> <out-dir>
set -u
bin=$1 out=$2
tree=$(dirname "$(dirname "$bin")")
mkdir -p "$out"
cmake --build "$tree" --target test-mul-mat-id-range -j 16 > "$out/build.log" 2>&1 || { echo "BUILD_FAILED"; grep -E "error" "$out/build.log" | head -10; exit 1; }
HIP_VISIBLE_DEVICES= "$tree/bin/test-mul-mat-id-range" > "$out/test.log" 2>&1
echo "rc=$? $(grep -c '^ok  ' "$out/test.log") ok, $(grep -c '^FAIL' "$out/test.log") fail"
grep -E "^FAIL" "$out/test.log" | head -10
tail -1 "$out/test.log"
echo MMID_RANGE_TEST_DONE
