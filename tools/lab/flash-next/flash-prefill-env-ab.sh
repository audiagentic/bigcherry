#!/bin/bash
# Flash-Next long-context prefill ABBA on ONE binary: A = caller env, B = caller env + the given extra env
# (e.g. BIGCHERRY_FA_SPARSE=1). long-ctx-profile timing pass per arm: fill prefill t/s, decode t/s, acceptance.
# The extra env is applied last, so it may also override UB and B for arm B.
# Usage: flash-prefill-env-ab.sh <depth> <llama-server> <out-root> <VAR=value>...
# env: ARMS (the order of the passes, default "A B B A"; "A B B B B B B" is one baseline and six runs of B, for the
#      question whether B repeats itself rather than how fast it is)
set -u
depth=$1 bin=$2 root=$3; shift 3
s="$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh"
n=0
for arm in ${ARMS:-A B B A}; do
  n=$((n+1))
  if [ "$arm" = A ]; then
    out=$(UB=${UB:-512} B=${UB:-512} DEPTH=$depth bash "$s" "$bin" "$root/d$depth-$n-A" timing 2>&1 | grep -E "^timing:|SERVER_FAILED")
  else
    out=$(env UB=${UB:-512} B=${UB:-512} DEPTH=$depth "$@" bash "$s" "$bin" "$root/d$depth-$n-B" timing 2>&1 | grep -E "^timing:|SERVER_FAILED")
  fi
  echo "d$depth $n $arm: $(echo $out | tr '\n' ' ')"
done
echo "md5 A: $(md5sum "$root"/d$depth-*-A/*.greedy.txt 2>/dev/null | awk '{print $1}' | sort -u | tr '\n' ' ')"
echo "md5 B: $(md5sum "$root"/d$depth-*-B/*.greedy.txt 2>/dev/null | awk '{print $1}' | sort -u | tr '\n' ' ')"
echo "B texts: $(md5sum "$root"/d$depth-*-B/*.greedy.txt 2>/dev/null | awk '{print substr($1, 1, 8)}' | sort | uniq -c | tr '\n' ' ')"
