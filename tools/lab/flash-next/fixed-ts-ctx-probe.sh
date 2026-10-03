#!/bin/bash
# Fixed-split context probe: for one -ts, try each KV (K/V) type pair at descending contexts; report the first that
# loads + serves (decode timing at ~24K depth). Used for the faster XTX-heavy splits found by queue-ts-skew-quick.sh.
# Usage: fixed-ts-ctx-probe.sh <llama-server> <out-root> <ts> "<ctx list desc>" "<ctk/ctv pairs>"
set -u
bin=$1 root=$2 ts=$3 ctxs=$4 kvs=$5
s=$(cd "$(dirname "$0")" && pwd)/long-ctx-profile.sh
export CTKD=f16 CTVD=f16 DECODE_N=256 DEPTH=24576 TS=$ts
for kv in $kvs; do
  for ctx in $ctxs; do
    out=$(CTK=${kv%/*} CTV=${kv#*/} CTX=$ctx bash "$s" "$bin" "$root/${kv/\//_}-$ctx" timing 2>&1 | grep -E "^timing: prompt|SERVER_FAILED")
    echo "ts=$ts kv=$kv ctx=$ctx ${out:-NO_RESULT}"
    case "$out" in *"decode "*) break;; esac
  done
done
