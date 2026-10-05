#!/bin/bash
# Prefill AllReduce provider sweep on one binary: the same uncached Flash-Next fill at DEPTH with each --allreduce
# provider, balanced order (A B C .. C B A). Reports fill prefill t/s, decode t/s and acceptance per run and the greedy
# md5 per provider. Large prefill reductions (5.2 MB at ub512) go to RCCL under the default cpu-root provider; this
# shows whether another existing provider is faster before any new code. Env (CTX, KV types, TS, profile) from caller.
# Usage: prefill-provider-sweep.sh <depth> <llama-server> <out-root> <provider>...
set -u
depth=$1 bin=$2 root=$3; shift 3
s="$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh"
order=("$@"); rev=(); for ((i=${#order[@]}-1; i>=0; i--)); do rev+=("${order[i]}"); done
n=0
for ar in "${order[@]}" "${rev[@]}"; do
  n=$((n+1))
  out=$(AR=$ar UB=${UB:-512} B=${UB:-512} DEPTH=$depth bash "$s" "$bin" "$root/$n-$ar" timing 2>&1 | grep -E "^timing:|SERVER_FAILED")
  echo "ar=$ar run$n: $(echo $out | tr '\n' ' ')"
done
for ar in "${order[@]}"; do echo "md5 $ar: $(md5sum "$root"/*-$ar/*.greedy.txt 2>/dev/null | awk '{print $1}' | sort -u | tr '\n' ' ')"; done
