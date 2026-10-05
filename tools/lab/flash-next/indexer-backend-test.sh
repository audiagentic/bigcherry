#!/bin/bash
# QFP17 / 1335 correctness: upstream test-backend-ops LIGHTNING_INDEXER cases against the CPU backend on every device
# of a build, with BIGCHERRY_INDEXER_TILE off and on. BIGCHERRY_PATCH_HIT lines (BIGCHERRY_PATCH_TRACE) show whether a
# test case reached the tile kernel (4 heads, 8 or more tokens).
# Usage: indexer-backend-test.sh <llama-server of the build> <out-dir>
set -u
bin=$1 out=$2
tree=$(dirname "$(dirname "$bin")")
mkdir -p "$out"
cmake --build "$tree" --target test-backend-ops -j 16 > "$out/build.log" 2>&1 || { echo "BUILD_FAILED"; tail -5 "$out/build.log"; exit 1; }
t="$tree/bin/test-backend-ops"
for flag in 0 1; do
  BIGCHERRY_PATCH_TRACE=1 BIGCHERRY_INDEXER_TILE=$flag "$t" test -o LIGHTNING_INDEXER > "$out/test.tile$flag.log" 2>&1
  echo "tile=$flag rc=$? $(sed 's/\x1b\[[0-9;]*m//g' "$out/test.tile$flag.log" | grep -cE 'OK$') ok, $(grep -cE 'FAIL' "$out/test.tile$flag.log") fail, $(sed 's/\x1b\[[0-9;]*m//g' "$out/test.tile$flag.log" | grep -cE 'n_head=4[,)]') case(s) with 4 heads"
  sed 's/\x1b\[[0-9;]*m//g' "$out/test.tile$flag.log" | grep -E "Backend [0-9]|FAIL|[0-9]+/[0-9]+ tests passed" | head -40
  echo "tile=$flag activation: $(grep -c "BIGCHERRY_PATCH_HIT patch=1335" "$out/test.tile$flag.log") marker line(s) $(grep -m1 -o "BIGCHERRY_PATCH_HIT patch=1335.*" "$out/test.tile$flag.log")"
done
# time the 4-head case that matches a prefill ubatch (kv 65536, 512 tokens, f16 keys), both ways
for flag in 0 1; do
  BIGCHERRY_INDEXER_TILE=$flag "$t" perf -o LIGHTNING_INDEXER -p 'n_head=4,.*n_kv=65536,n_batch=512.*type_K=f16' > "$out/perf.tile$flag.log" 2>&1
  echo "perf tile=$flag:"; sed 's/\x1b\[[0-9;]*m//g' "$out/perf.tile$flag.log" | grep -E "Backend [0-9]|LIGHTNING_INDEXER" | cut -c1-220 | head -12
done
echo BACKEND_TEST_DONE
