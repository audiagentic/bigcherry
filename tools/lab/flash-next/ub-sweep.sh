#!/bin/bash
# -ub/-b sweep at the production context (CTX from the caller, f16 KV): balanced order (A B C D D C B A) over UB values,
# each arm one long-ctx-profile timing pass at DEPTH (fill prefill t/s + decode t/s). SERVER_FAILED = does not fit.
# Usage: ub-sweep.sh <llama-server> <out-root> <ub values...>
set -u
export BIGCHERRY_QSA_HOST_REMAP=1  # sweeps run profile v6 (v5 + 1327)
bin=$1 root=$2; shift 2
s="$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh"
order=("$@"); rev=(); for ((i=${#order[@]}-1; i>=0; i--)); do rev+=("${order[i]}"); done
n=0
for ub in "${order[@]}" "${rev[@]}"; do
  n=$((n+1))
  out=$(UB=$ub B=$ub DEPTH=${DEPTH:-65536} bash "$s" "$bin" "$root/ub$ub-$n" timing 2>&1 | grep -E "^timing:|SERVER_FAILED")
  echo "ub$ub run$n: $(echo $out | tr '\n' ' ')"
done
