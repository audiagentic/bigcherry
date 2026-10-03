#!/bin/bash
# Rank balance vs decode: cpu-root consume averages ~121 us x 96 AllReduces per MTP step at 10K with
# -ts 2,2,3 (XTX ranks wait on the R9700). Same build, 32K context so every split fits, ~10K cached prompt,
# MTP3: -ts 2,2,3 / 4,4,3 / 3,3,2 / 1,1,1 in two rotated passes. Usage: balance-sweep.sh <llama-server> <out-root>
set -u
bin=$1 root=$2
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
arms=(2,2,3 4,4,3 3,3,2 1,1,1)
for pass in a b; do
  for ts in "${arms[@]}"; do
    echo "== ts $ts pass $pass"
    TS=$ts CTX=32768 DEPTH=8192 bash "$s" "$bin" "$root/ts${ts//,/}-$pass" timing
  done
  arms=(1,1,1 3,3,2 4,4,3 2,2,3)
done
