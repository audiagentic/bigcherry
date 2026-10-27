#!/bin/bash
# RAD26 diagnosis: where does the large-lane wake get lost? Starts the production Flash-Next layout on this checkout's
# build, sends one single-stream request, then four concurrent ones (a multi-stream step above the small class goes to
# the large lane), each with a timeout, then stops and prints the engine's wake counters (queued/captured/fired) and the
# helpers' large-lane request and wake counts.
# Usage (from a checkout, e.g. through `bigcherry slice lab`): rad26-diag.sh
set -u
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
OUT=$BC_WORK_ROOT/runs/rad26-diag-$(date +%m%d%H%M); mkdir -p $OUT
source tools/lab/radiance/lab-build.sh
B=$(rad_build $OUT/build.log) || exit 1
L=tools/lab/radiance/launch-flashnext-best.sh
RUN=$BC_WORK_ROOT/flashnext
URL=http://127.0.0.1:36080/v1/completions   # the launcher's default port
bash $L start --build $B > $OUT/launch.log 2>&1 || { tail $OUT/launch.log; bash $L stop; exit 1; }
req() { curl -s -m 60 $URL -d "{\"prompt\":\"$1\",\"max_tokens\":32,\"temperature\":0}" | head -c 120; echo; }
echo "single: $(req 'One request.')"
for i in 1 2 3 4; do req "Concurrent request $i." > $OUT/c$i.txt & done; wait
for i in 1 2 3 4; do echo "concurrent $i: $(cat $OUT/c$i.txt)"; done
e=$(ls -t $RUN/engine-*.log | head -1)
grep -h "large-lane wake\|stream capture" $e | head -5
bash $L stop > $OUT/stop.log 2>&1
sleep 2
grep -h "large-lane wakes\|large-lane wake" $e | tail -3
for h in $(ls -t $RUN/helper*-*.log | head -2); do grep -h "large-lane requests" $h; done
echo "OUT $OUT"; echo RAD26_DIAG_DONE
