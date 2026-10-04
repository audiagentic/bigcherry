#!/bin/bash
# QFP17 1332: ubatch x QSA chunk sweep at the production context (CTX from the caller, f16 KV). Arms are UB:CHUNK
# pairs (CHUNK 0 = unchunked), balanced order (A B C .. C B A), one long-ctx-profile timing pass per arm at DEPTH
# (fill prefill t/s + decode t/s). SERVER_FAILED = does not fit.
# Usage: ubchunk-sweep.sh <llama-server> <out-root> <ub:chunk...>
set -u
bin=$1 root=$2; shift 2
s="$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh"
order=("$@"); rev=(); for ((i=${#order[@]}-1; i>=0; i--)); do rev+=("${order[i]}"); done
n=0
for arm in "${order[@]}" "${rev[@]}"; do
  n=$((n+1))
  ub=${arm%%:*} chunk=${arm##*:}
  out=$(BIGCHERRY_QSA_CHUNK=$chunk UB=$ub B=$ub DEPTH=${DEPTH:-65536} bash "$s" "$bin" "$root/ub$ub-c$chunk-$n" timing 2>&1 | grep -E "^timing:|SERVER_FAILED")
  echo "ub$ub c$chunk run$n: $(echo $out | tr '\n' ' ')"
done
