#!/bin/bash
# 1294 (deterministic radix TOP_K ties) A/B: ABBA decode at ~10K and ~80K cached context (MTP3), plus
# cross-start greedy determinism at 32K and 80K on the new build.
# Usage: topk-ab.sh <base llama-server (1291+1292)> <new llama-server (+1294)> <out-root>
set -u
base=$1 new=$2 root=$3
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
for depth in 8192 65536; do
  for arm in base-a new-a new-b base-b; do
    bin=$base; [[ $arm == new-* ]] && bin=$new
    DEPTH=$depth bash "$s" "$bin" "$root/d$depth/$arm" timing
  done
done
PROVIDERS=cpu-root DEPTHS="32768 65536" bash "$(dirname "$s")/determinism.sh" "$new" "$root/determinism"
