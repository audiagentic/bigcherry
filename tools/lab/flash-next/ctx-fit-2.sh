#!/bin/bash
# Round-2 fit rows, then RCCL Tree+Simple vs default ABBA (cr-nomtp-auto-a probe, -ts 4,4,3 ub1024).
# Usage: ctx-fit-2.sh <llama-server> <out-root>
set -u
bin=$1; root=$2
here=$(cd "$(dirname "$0")" && pwd)
FIT_ONLY='256k-223-ub256|256k-223-ub512-kvq4|256k-223-ub256-kvq4|192k-223-ub1024-kvq4' bash "$here/long-ctx-fit.sh" "$bin" "$root/fit"
for arm in default-a tree-a tree-b default-b; do
  envs=; [[ $arm == tree-* ]] && envs="NCCL_ALGO=Tree NCCL_PROTO=Simple"
  echo "== $arm: ${envs:-<defaults>}"
  env $envs bash "$here/layout-probe.sh" "$bin" "$root/$arm" 'cr-nomtp-auto-a'
done
