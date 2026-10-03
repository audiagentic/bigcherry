#!/bin/bash
# Prefill tier check: ub512 vs ub1024 for the context tiers found in fit-sweep-2 (10K cached prompt, MTP3):
# 4,4,3 f16 KV at 96K; 4,4,3 q8_0 at 160K; 2,2,3 q8_0 at 192K. ub1024 needs more compute buffer; a load
# failure means that tier stays on ub512. Usage: ub-sweep.sh <llama-server> <out-root>
set -u
bin=$1 root=$2
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
for tier in "4,4,3 f16 98304" "4,4,3 q8_0 163840" "2,2,3 q8_0 196608"; do
  set -- $tier; ts=$1 kv=$2 ctx=$3
  for ub in 512 1024 1024 512; do
    echo "== ts $ts kv $kv ctx $ctx ub $ub"
    TS=$ts CTK=$kv CTV=$kv CTX=$ctx UB=$ub DEPTH=8192 bash "$s" "$bin" "$root/ts${ts//,/}-$kv-c$ctx-ub$ub-$RANDOM" timing 2>&1 | grep -E "^timing:|SERVER_FAILED"
  done
done
