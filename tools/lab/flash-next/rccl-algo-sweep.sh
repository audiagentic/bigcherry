#!/bin/bash
# Flash-Next in-model RCCL algorithm/CTA sweep (GPT ranking: Tree with the R9700 as a leaf, CTA counts; the
# earlier rccl-env-sweep only covered PROTO/channels/buffsize/SHM-CE). Same binary, auto (= RCCL for 3 GPUs),
# no MTP, -ts 4,4,3 ub1024; each variant is one cr-nomtp-auto-a probe, with an unset-env control first and last.
# Usage: rccl-algo-sweep.sh <llama-server> <out-root>
set -u
bin=$1; root=$2
here=$(cd "$(dirname "$0")" && pwd)
variants=(
  "control-a|"
  "tree-simple|NCCL_ALGO=Tree NCCL_PROTO=Simple"
  "ring-simple|NCCL_ALGO=Ring NCCL_PROTO=Simple"
  "tree-ll128|NCCL_ALGO=Tree NCCL_PROTO=LL128"
  "ring-ll128|NCCL_ALGO=Ring NCCL_PROTO=LL128"
  "ctas-2|NCCL_MIN_CTAS=2 NCCL_MAX_CTAS=2"
  "ctas-4|NCCL_MIN_CTAS=4 NCCL_MAX_CTAS=4"
  "ctas-8|NCCL_MIN_CTAS=8 NCCL_MAX_CTAS=8"
  "control-b|"
)
for v in "${variants[@]}"; do
  name=${v%%|*} envs=${v#*|}
  echo "== $name: ${envs:-<defaults>}"
  env $envs bash "$here/layout-probe.sh" "$bin" "$root/$name" 'cr-nomtp-auto-a'
done
