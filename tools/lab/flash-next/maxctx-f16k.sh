#!/bin/bash
# Max context with f16 K / q8_0 V (owner preference; q4 never). fit-sweep-2: 96K OK, 128K loaded but failed at
# request time with -ts 3,3,2 / 4,4,3 ub512. Try R9700-weighted splits and ub256 (compute buffers are the limit):
# per (split, ub) step context up from 112K and stop at the first config that cannot complete a ~10K request.
# Usage: maxctx-f16k.sh <llama-server> <out-root>
set -u
bin=$1 root=$2
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
for ts in 2,2,3 3,3,4 1,1,1; do
  for ub in 512 256; do
    for ctx in 114688 131072 147456 163840 180224 196608; do
      echo "== ts $ts ub $ub ctx $ctx (K f16, V q8_0)"
      out=$(TS=$ts UB=$ub CTX=$ctx CTK=f16 CTV=q8_0 DEPTH=8192 bash "$s" "$bin" "$root/ts${ts//,/}-ub$ub-c$ctx" timing 2>&1)
      line=$(echo "$out" | grep -E "^timing: prompt")
      if [ -n "$line" ]; then echo "   $(echo "$out" | grep -m1 'fill prefill')"; echo "   $line"; echo "$out" | grep "VRAM Total" | sed 's/^/   /'; else echo "   FAILED"; break; fi
    done
  done
done
