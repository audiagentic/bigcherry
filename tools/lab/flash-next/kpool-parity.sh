#!/bin/bash
# 1292 parity: with MTP the greedy text varies run to run even on one binary (acceptance changes verify batch
# shapes), so compare without MTP: each binary twice at ~10K and ~80K cached context. 1292 changes only pool
# bookkeeping, so new must match base wherever base matches itself.
# Usage: kpool-parity.sh <base llama-server> <new llama-server> <out-root>
set -u
base=$1 new=$2 root=$3
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
export NO_MTP=1
for depth in 8192 65536; do
  for arm in base-a new-a new-b base-b; do
    bin=$base; [[ $arm == new-* ]] && bin=$new
    DEPTH=$depth bash "$s" "$bin" "$root/d$depth/$arm" timing
  done
  ref=$(ls "$root/d$depth/base-a/"*.greedy.txt)
  for arm in base-b new-a new-b; do
    cmp -s "$ref" "$root/d$depth/$arm/"*.greedy.txt && echo "parity d$depth $arm == base-a" || echo "parity d$depth $arm DIFFERS from base-a"
  done
done
