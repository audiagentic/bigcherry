#!/bin/bash
# Fit probe for larger ubatch at f16 KV: each "UB:TS:CTX" config runs one long-ctx-profile timing pass at DEPTH
# (fill prefill t/s + decode t/s) and records SERVER_FAILED when it does not fit; VRAM per GPU from the run.
# Usage: fit-probe.sh <llama-server> <out-root> <UB:TS:CTX ...>
set -u
bin=$1 root=$2; shift 2
s="$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh"
export BIGCHERRY_QSA_HOST_REMAP=1  # profile v6
for cfg in "$@"; do
  IFS=: read -r ub ts ctx <<< "$cfg"
  d="$root/ub$ub-ts$ts-c$ctx"
  out=$(UB=$ub B=$ub TS=$ts CTX=$ctx DEPTH=${DEPTH:-65536} bash "$s" "$bin" "$d" timing 2>&1 | grep -E "^timing:|SERVER_FAILED")
  vram=$(grep -h "VRAM Total Used" $d/*vram* 2>/dev/null | awk '{printf "%.1f ", $NF/1073741824}')
  echo "ub$ub ts$ts ctx$ctx: $(echo $out | tr '\n' ' ') | VRAM GiB: $vram"
done
