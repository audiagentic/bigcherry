#!/bin/bash
# Draft (6900) investigation on the 1291+1292+1294 build, 192K deployment config:
#  1. rocprof decode window at ~10K (compare with the ~80K trace flashnext-long-ctx-decode-3): is QSA attention
#     per token flat with depth (masked tiles skipped) or ~linear (full n_kv read)?
#  2. draft KV q8_0 vs f16 ABBA (target KV stays q8_0) at ~10K and ~80K, MTP3.
# Usage: draft-ab.sh <llama-server> <out-root>
set -u
bin=$1 root=$2
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
DEPTH=8192 bash "$s" "$bin" "$root/decode-10k" decode | grep -E "decode window|flash_attn|dequantize|ms/tok" | head -30
for depth in 8192 65536; do
  for arm in q8-a f16-a f16-b q8-b; do
    kv=q8_0; [[ $arm == f16-* ]] && kv=f16
    CTKD=$kv CTVD=$kv DEPTH=$depth bash "$s" "$bin" "$root/d$depth/$arm" timing
  done
done
