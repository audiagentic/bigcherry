#!/bin/bash
# Candidate v7 at the 80K fill: v6 (ub512, gather off) vs v7 (ub1024, chunk 512, gather on), balanced order via
# ubchunk-sweep; gather is toggled with the arm (512:0 = off, 1024:512 = on). Usage: v7-sweep.sh <bin> <root>
set -u
bin=$1 root=$2
s="$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh"
n=0
for arm in v6 v7 v7 v6; do
  n=$((n+1))
  if [ $arm = v6 ]; then ub=512 c=0 g=0; else ub=1024 c=512 g=1; fi
  out=$(BIGCHERRY_QSA_GATHER=$g BIGCHERRY_QSA_CHUNK=$c UB=$ub B=$ub DEPTH=${DEPTH:-65536} bash "$s" "$bin" "$root/$arm-$n" timing 2>&1 | grep -E "^timing:|SERVER_FAILED")
  echo "ub$ub c$c gather$g ($arm) run$n: $(echo $out | tr '\n' ' ')"
done
