#!/bin/bash
# 192K deployment sweep on the 1291+1292 build: MTP draft depth (2/3/4) x RCCL Tree+Simple (on/off), each at
# ~10K and ~80K cached context; records fill prefill and decode (ms/step = time / (n - accepted)).
# Order is rotated per depth so no arm always runs first. Usage: deploy-sweep.sh <llama-server> <out-root>
set -u
bin=$1 root=$2
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
arms=("3 def" "3 tree" "2 def" "4 tree" "2 tree" "4 def")
for depth in 8192 65536; do
  for a in "${arms[@]}"; do
    set -- $a; n=$1 rc=$2
    envs=; [ "$rc" = tree ] && envs="NCCL_ALGO=Tree NCCL_PROTO=Simple"
    echo "== d$depth n$n $rc"
    env $envs SPEC_N=$n DEPTH=$depth bash "$s" "$bin" "$root/d$depth/n$n-$rc" timing
  done
  arms=("${arms[@]:3}" "${arms[@]:0:3}")
done
