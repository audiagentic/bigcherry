#!/bin/bash
# QFP25 / 1334 correctness: upstream test-backend-ops flash-attention cases that set n_kv_max (sparse masks with a
# different random cell selection per query - the adversarial case for the per-tile union list), compared with the
# CPU backend, on every device of a build, with BIGCHERRY_FA_SPARSE off and on. The qwen-shaped case
# (hsk 256, 2 KV heads, gqa 12, kv 8192, 64 queries, n_kv_max 512) reaches the (256, 256, 8, 8) sparse kernel on
# RDNA WMMA when the flag is on. Also times the same cases (perf mode) so the flag's effect is visible.
# Usage: fa-sparse-backend-test.sh <llama-server of the build> <out-dir>
set -u
bin=$1 out=$2
tree=$(dirname "$(dirname "$bin")")
mkdir -p "$out"
cmake --build "$tree" --target test-backend-ops -j 16 > "$out/build.log" 2>&1 || { echo "BUILD_FAILED"; tail -5 "$out/build.log"; exit 1; }
t="$tree/bin/test-backend-ops"
for flag in 0 1; do
  BIGCHERRY_FA_SPARSE=$flag "$t" test -o FLASH_ATTN_EXT -p 'n_kv_max=(512|2048)' > "$out/test.sparse$flag.log" 2>&1
  echo "sparse=$flag rc=$? $(sed 's/\x1b\[[0-9;]*m//g' "$out/test.sparse$flag.log" | grep -cE ': OK$|OK$') ok, $(sed 's/\x1b\[[0-9;]*m//g' "$out/test.sparse$flag.log" | grep -cE 'FAIL') fail"
  sed 's/\x1b\[[0-9;]*m//g' "$out/test.sparse$flag.log" | grep -E "Backend [0-9]|FAIL|[0-9]+/[0-9]+ tests passed|NMSE" | head -40
done
for flag in 0 1; do
  BIGCHERRY_FA_SPARSE=$flag "$t" perf -o FLASH_ATTN_EXT -p 'hsk=256,hsv=256,nh=2,nr23=\[12,1\],kv=8192,nb=64.*n_kv_max=512' > "$out/perf.sparse$flag.log" 2>&1
  echo "perf sparse=$flag:"; sed 's/\x1b\[[0-9;]*m//g' "$out/perf.sparse$flag.log" | grep -E "Backend [0-9]|FLASH_ATTN_EXT" | cut -c1-230 | head -12
done
echo BACKEND_TEST_DONE
