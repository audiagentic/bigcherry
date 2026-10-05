#!/bin/bash
# Flash-Next long-context prefill ABBA of two builds at one depth (long-ctx-profile timing pass: fill prefill t/s,
# decode t/s, acceptance). Env (CTX, KV types, TS, profile) comes from the caller, as for ubchunk-sweep.sh.
# Usage: flash-prefill-ab.sh <depth> <llama-server A> <llama-server B> <out-root>
set -u
depth=$1 a=$2 b=$3 root=$4
s="$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh"
n=0
for arm in A B B A; do
  n=$((n+1))
  [ "$arm" = A ] && bin=$a || bin=$b
  out=$(UB=${UB:-512} B=${UB:-512} DEPTH=$depth bash "$s" "$bin" "$root/d$depth-$n-$arm" timing 2>&1 | grep -E "^timing:|SERVER_FAILED")
  echo "d$depth $n $arm: $(echo $out | tr '\n' ' ')"
done
md5sum "$root"/d$depth-*/*.greedy.txt 2>/dev/null | awk '{print $1}' | sort | uniq -c
