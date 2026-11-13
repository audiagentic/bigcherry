#!/bin/bash
# HL02 Gate B: three threeway_push sessions (forward, reverse, forward arm order). Run under the exclusive gpu lease:
#   agent-lease acquire gpu --ttl 1h --purpose "HL02 Gate B" -- tools/lab/radiance/hl02-host-attributes/run_threeway.sh BIN OUTDIR
# BIN is threeway_push built with: hipcc -O2 -mavx2 --offload-arch=gfx1100 --offload-arch=gfx1201 threeway_push.hip -o BIN -lpthread -lhsa-runtime64
set -u
BIN=$1; OUT=$2; mkdir -p "$OUT"
for s in 1 2 3; do
  r=""; [ $s = 2 ] && r=--reverse
  timeout 600 "$BIN" --sizes 327680,1048576,4194304,10485760 --iters 30 --cpus 0,2,4 --push-one 0 $r > "$OUT/session$s.jsonl" 2> "$OUT/session$s.err"
  echo "session $s rc=$?"
done
