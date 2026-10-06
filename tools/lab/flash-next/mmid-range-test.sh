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
# GPU mode: the range op computed directly on every device against the CPU reference (phase B)
"$tree/bin/test-mul-mat-id-range" --gpu > "$out/test.gpu.log" 2>&1
echo "gpu rc=$? $(grep -c '^ok  ' "$out/test.gpu.log") ok, $(grep -c '^FAIL' "$out/test.gpu.log") fail"
grep '^FAIL' "$out/test.gpu.log" | awk '{print $2, $3, "tokens=" $7}' | sed 's/tokens=tokens=/tokens=/' | sort | uniq -c | sort -rn | head -40
grep -E "^FAIL" "$out/test.gpu.log" | head -6 | cut -c1-200
grep -E "error|abort|assert|ROCm error" "$out/test.gpu.log" | head -5 | cut -c1-200
tail -1 "$out/test.gpu.log"
echo MMID_RANGE_TEST_DONE
