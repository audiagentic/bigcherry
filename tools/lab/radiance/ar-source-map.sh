#!/bin/bash
# Print the exact lab-host no-P2P all-reduce implementation exercised by the profiler.
set -eu
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
echo "radiance_commit=$(git -C "$src" rev-parse HEAD)"
echo "== TP3 all-reduce headings"
grep -nEi "all.?reduce|pinned|host|wire|wht6|wait|counter" "$src/docs/TP3.md" | head -120 || true
echo "== source references"
grep -RInE "ar_rmsnorm_quant_fp8|wht6|pinned|host.*(reduce|fold|scratch)|wait.*counter|peer.*scratch" \
  "$src/libr4d" "$src/core" 2>/dev/null | head -260 || true
