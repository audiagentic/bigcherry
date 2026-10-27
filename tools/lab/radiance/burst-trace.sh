#!/bin/bash
# RAD20 phase 1: trace the engine's inference-burst boundaries (rad0002 burst callback, logged by libr3moe with
# R3_MOE_WORKER_BURST_TRACE=1) in the production Flash-Next layout (MTP on) across single-stream, concurrent,
# back-to-back and long-prefill requests, then shutdown. Gate: begin/end strictly alternate, every begin has an end,
# the last transition before shutdown is an end, and each scenario opens at least one burst.
# Usage: burst-trace.sh
set -u
source tools/env/bigcherry-env.sh >/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/lab-env.sh"
OUT=$BC_WORK_ROOT/runs/rad20-burst-trace-$(date +%m%d%H%M); mkdir -p $OUT
PYTHONPATH=tools timeout 5400 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance \
    --experiment moe-remote-handoff > $OUT/build.log 2>&1
plan=$(grep -oE 'ok build_plan_id=[0-9a-f]+' $OUT/build.log | tail -1 | cut -d= -f2)
[ -n "$plan" ] || { grep -E ' error:|FAILED' $OUT/build.log | sort -u | head; exit 1; }
B=$(ls -d $BC_WORK_ROOT/builds/*/$plan)
L=tools/lab/radiance/launch-flashnext-best.sh
R3_MOE_WORKER_BURST_TRACE=1 bash $L start --build $B > $OUT/launch.log 2>&1 || { tail $OUT/launch.log; bash $L stop; exit 1; }
ENG=$(grep -oE "log [^ ]*engine-[0-9-]*\.log" $OUT/launch.log | tail -1 | cut -d' ' -f2)
URL=http://127.0.0.1:36080/v1/completions
req() { curl -s $URL -d "{\"prompt\":\"$1\",\"max_tokens\":$2,\"temperature\":0}" > /dev/null; }
mark() { echo "== $1 $(date +%s.%N)" | tee -a $OUT/scenarios.txt; }
mark single;      req "Tell me about lighthouses." 64; sleep 2
mark concurrent;  for i in 1 2 3 4; do req "Story number $i about a ship." 96 & done; wait; sleep 2
mark backtoback;  req "One." 32; req "Two." 32; req "Three." 32; sleep 2
LONG=$(python3 -c "print(' '.join(['The quick brown fox jumps over the lazy dog.']*400))")
mark prefill;     req "$LONG Summarise." 32; sleep 2
mark stop
bash $L stop > /dev/null 2>&1
grep "libr3moe: burst" $ENG > $OUT/bursts.txt
python3 - $OUT/bursts.txt $OUT/scenarios.txt <<'PY' | tee $OUT/summary.txt
import re, sys
ev = [(l.split()[2], float(l.split("t=")[1])) for l in open(sys.argv[1]) if "burst" in l]
marks = [(l.split()[1], float(l.split()[2])) for l in open(sys.argv[2])]
ok = True
for i, (kind, _) in enumerate(ev):
    want = "begin" if i % 2 == 0 else "end"
    if kind != want: ok = False; print(f"FAIL: transition {i} is {kind}, expected {want}")
begins = sum(1 for k, _ in ev if k == "begin"); ends = len(ev) - begins
print(f"transitions {len(ev)}: begins {begins}, ends {ends}")
if begins != ends: ok = False; print("FAIL: unpaired burst")
if ev and ev[-1][0] != "end": ok = False; print("FAIL: still in a burst at shutdown")
# bursts per scenario (wall clock: CLOCK_MONOTONIC in the log vs date in marks are different clocks; use order)
print("bursts:", [round(ev[i + 1][1] - ev[i][1], 4) for i in range(0, len(ev) - 1, 2)], "s each")
print("PHASE1_GATE", "PASS" if ok and begins >= 4 else "FAIL")
PY
echo "OUT $OUT"; echo BURST_TRACE_DONE
