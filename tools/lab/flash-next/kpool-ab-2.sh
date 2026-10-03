#!/bin/bash
# 1292 follow-up: ABBA (1291 vs 1291+1292) at ~10K and ~80K cached context with greedy-output parity,
# then perf of the 1292 build at both depths to find the next depth-scaling host cost.
# Usage: kpool-ab-2.sh <base llama-server> <new llama-server> <out-root>
set -u
base=$1 new=$2 root=$3
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
for depth in 8192 65536; do
  for arm in base-a new-a new-b base-b; do
    bin=$base; [[ $arm == new-* ]] && bin=$new
    DEPTH=$depth bash "$s" "$bin" "$root/d$depth/$arm" timing
  done
  ref=$(ls "$root/d$depth/base-a/"*.greedy.txt)
  for arm in new-a new-b base-b; do
    cmp -s "$ref" "$root/d$depth/$arm/"*.greedy.txt && echo "greedy d$depth $arm == base-a" || echo "greedy d$depth $arm DIFFERS from base-a"
  done
done
for depth in 8192 65536; do
  DEPTH=$depth bash "$s" "$new" "$root/perf-d$depth" perf
done
