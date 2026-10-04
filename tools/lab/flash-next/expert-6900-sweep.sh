#!/bin/bash
# Move N layers' routed-expert tensors (ffn_*_exps) to the 6900 (ROCm3, also the MTP draft GPU) with -ot and run one
# long-ctx-profile timing pass at DEPTH per config (fill prefill t/s + decode t/s), 240K f16, profile v6.
# Config "LAYERS:UB" with LAYERS a |-separated block list or "none". VRAM per GPU recorded.
# Usage: expert-6900-sweep.sh <llama-server> <out-root> <LAYERS:UB ...>
set -u
bin=$1 root=$2; shift 2
s="$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh"
export BIGCHERRY_QSA_HOST_REMAP=1
base_ot='^token_embd\.weight$=CPU'
n=0
for cfg in "$@"; do
  n=$((n+1)); layers=${cfg%%:*}; ub=${cfg##*:}
  ot=$base_ot; [ "$layers" != none ] && ot="$base_ot,blk\.($layers)\.ffn_.*_exps\.weight=ROCm3"
  nl=$([ "$layers" = none ] && echo 0 || echo "$layers" | tr '|' '\n' | wc -l)
  d="$root/r$n-L$nl-ub$ub"
  out=$(EXTRA_OT="$ot" UB=$ub B=$ub DEPTH=${DEPTH:-65536} bash "$s" "$bin" "$d" timing 2>&1 | grep -E "^timing:|SERVER_FAILED")
  vram=$(grep -h "VRAM Total Used" $d/*vram* 2>/dev/null | awk '{printf "%.1f ", $NF/1073741824}')
  echo "run$n layers=$nl ub$ub: $(echo $out | tr '\n' ' ') | VRAM GiB: $vram"
done
