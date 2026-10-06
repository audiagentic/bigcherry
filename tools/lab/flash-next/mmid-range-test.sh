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
# GPU mode: the range op computed directly on every device against the CPU reference (phase B), one band of batch
# sizes per run so a crash in a path that is not converted yet does not hide the bands that work
# hit = the range dedup path ran (QFP30; MMQ bands with a broadcast activation)
export BIGCHERRY_PATCH_TRACE=1
for band in ${BANDS:-"1 1" "2 4" "8 9" "33 33" "300 300"}; do
  set -- $band
  log="$out/test.gpu.$1-$2.log"
  MMID_RANGE_CONTROL=1 MMID_RANGE_MIN_TOKENS=$1 MMID_RANGE_MAX_TOKENS=$2 "$tree/bin/test-mul-mat-id-range" --gpu > "$log" 2>&1
  echo "gpu tokens $1..$2: rc=$? hit=$(grep -c "path=range_dedup" "$log") $(grep -c '^ok  ' "$log") ok, $(grep -c "^skip" "$log") skipped, $(grep -c '^FAIL' "$log") fail$(grep -m1 -oE "illegal memory access|Segmentation|abort" "$log" | sed 's/^/, /')"
  grep '^FAIL' "$log" | awk '{print $2, $3, $7}' | sort | uniq -c | sort -rn | head -12
  grep '^FAIL' "$log" | head -2 | cut -c1-200
  grep -a '^ctrl' "$log" | awk '{print $3, $4, $NF}' | sort | uniq -c | sort -rn | head -6
done
echo MMID_RANGE_TEST_DONE
