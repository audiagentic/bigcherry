#!/bin/bash
# 1293 (one scheduler input sync per split) A/B: ABBA decode at ~10K and ~80K cached context (MTP3), plus
# no-MTP greedy parity at 10K and the hipStreamSynchronize call-site count on the new build.
# Usage: sync-ab.sh <base llama-server (1291+1292)> <new llama-server (+1293)> <out-root>
set -u
base=$1 new=$2 root=$3
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
for depth in 8192 65536; do
  for arm in base-a new-a new-b base-b; do
    bin=$base; [[ $arm == new-* ]] && bin=$new
    DEPTH=$depth bash "$s" "$bin" "$root/d$depth/$arm" timing
  done
done
for arm in base new; do
  bin=$base; [ $arm = new ] && bin=$new
  NO_MTP=1 DEPTH=8192 bash "$s" "$bin" "$root/parity/$arm" timing
done
cmp -s "$root/parity/base/timing.8192.greedy.txt" "$root/parity/new/timing.8192.greedy.txt" && echo "parity 10K no-MTP: identical" || echo "parity 10K no-MTP: DIFFERS"
DEPTH=8192 bash "$s" "$new" "$root/synctrace" synctrace | grep -m1 "calls while armed"
