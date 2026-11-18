#!/bin/bash
# Second pass, after smoke_all.sh: the layouts it ran with the first (bf16-first) rule need their fp8 baseline, and 4gpu all four types.
F=<dev-root>/tmp/bigcherry/vtp44-fn
while kill -0 "$(cat $F/smoke/driver.pid)" 2>/dev/null; do sleep 20; done
run() { agent-lease acquire gpu --ttl 2h --est 30m --purpose "VTP44 layout smoke pass 2: $*" --wait -- bash $F/smoke_layout.sh "$@"; echo "$(date -u +%T) pass2 $* done (lease exit $?)" >> $F/smoke/queue.log; }
run flashnext-2xtx fp8
run flashnext-4gpu fp8 bf16_fp8 bf16_i8 bf16
run flashnext-tp2xtx fp8
echo "$(date -u +%T) PASS 2 DONE" >> $F/smoke/queue.log
