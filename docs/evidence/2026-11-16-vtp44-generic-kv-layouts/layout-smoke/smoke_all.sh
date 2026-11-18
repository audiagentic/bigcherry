#!/bin/bash
# Queue each layout under its own gpu lease so other work can interleave between layouts.
F=<dev-root>/tmp/bigcherry/vtp44-fn
for L in flashnext-2xtx flashnext-4gpu flashnext-tp2xtx flashnext-r9700 flashnext-2xtx-rdna2-tail flashnext-2xtx-rdna2-head flashnext-2xtx-combine flashnext-2xtx-rdna2-tail-combine; do
  agent-lease acquire gpu --ttl 2h --est 45m --purpose "VTP44 layout smoke $L (bf16, bf16_fp8, bf16_i8)" --wait -- bash $F/smoke_layout.sh $L
  echo "$(date -u +%T) layout $L done (lease exit $?)" >> $F/smoke/queue.log
done
echo "$(date -u +%T) ALL DONE" >> $F/smoke/queue.log
