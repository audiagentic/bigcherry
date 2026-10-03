#!/bin/bash
# Long-context fit by split and KV type (owner: never q4 KV; f16 preferred; f16-K/q8_0-V allowed).
# For each -ts (3,3,2 and 4,4,3 decode ~6% faster per step than 2,2,3) and KV pair, step context up and stop at
# the first load failure. Each loaded config times a ~10K cached prompt (fill prefill + MTP3 decode).
# Usage: fit-sweep-2.sh <llama-server> <out-root>
set -u
bin=$1 root=$2
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
for ts in 3,3,2 4,4,3; do
  for kv in "f16 f16" "f16 q8_0" "q8_0 q8_0"; do
    set -- $kv; ctk=$1 ctv=$2
    for ctx in 65536 98304 131072 163840; do
      d="$root/ts${ts//,/}-k$ctk-v$ctv-c$ctx"
      echo "== ts $ts K $ctk V $ctv ctx $ctx"
      out=$(TS=$ts CTX=$ctx CTK=$ctk CTV=$ctv DEPTH=8192 bash "$s" "$bin" "$d" timing 2>&1)
      echo "$out" | grep -E "^timing:|SERVER_FAILED|VRAM Total" | sed 's/^/   /'
      echo "$out" | grep -q "SERVER_FAILED" && break
    done
  done
done
