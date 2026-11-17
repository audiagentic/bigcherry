#!/bin/bash
# RHP01 coverage run: production layout flashnext-4gpu on the routing-evidence build (rad0010) with RAD_ROUTING_EVIDENCE=1.
# Prefill of several sizes and long prefill, then a clean stop so the engine prints its per-rank, per-phase counters at teardown.
# no set -u: tools/env/bigcherry-env.sh reads unset variables
D=<work-root>/runs/rhp01-coverage
P=<repo>
L=$P/tools/lab/radiance/launch-flashnext-best.sh
EB=<work-root>/builds/7c89e90763210fe88847aa2a357dcbbd/3462427eb60d432513673cf79930d6f4
PORT=18415
rm -f $D/done $D/status; echo $$ > $D/run.pid
LEASED=0; STARTED=0
finish() { [ $STARTED = 1 ] && bash $L stop > $D/final-stop.log 2>&1; [ $LEASED = 1 ] && agent-lease release gpu; touch $D/done; }
trap finish EXIT
trap 'exit 130' INT TERM HUP
while [ ! -e <work-root>/kbench-libr3d/ab1/done ]; do sleep 30; done
until agent-lease acquire gpu --ttl 3h --purpose "RHP01 routing-evidence coverage run (flashnext-4gpu, evidence build)"; do sleep 60; done
LEASED=1
cd $P; set --; source tools/env/bigcherry-env.sh >/dev/null 2>&1
export W4H2=1 GEMV_LARGE=32 FLASHNEXT_LAYOUT=flashnext-4gpu RAD_ROUTING_EVIDENCE=1
STARTED=1
bash $L start --build $EB --port $PORT --record-heat off --host-pool-mib 9216 --engine-arg --prefix-cache-mib --engine-arg 2048 > $D/launch.log 2>&1 || { echo "start failed" > $D/status; exit 1; }
sb() { local tag=$1; shift; PYTHONPATH=tools timeout 1800 python3 -m bigcherry stream-bench --url http://127.0.0.1:$PORT --temperature 0 --reps 1 --output $D/$tag.json "$@" > $D/$tag.log 2>&1; }
for s in 1 2 3 4; do sb d1s$s --streams 1 --prompt-tokens 512 --max-tokens 256 --seed $s; done
sb soak6 --streams 6 --prompt-tokens 2048 --max-tokens 512
for s in 41 42 43; do sb pf8k.s$s --streams 1 --prompt-tokens 8192 --max-tokens 16 --seed $s; done
sb pf32k --streams 1 --prompt-tokens 32768 --max-tokens 16 --seed 51
sb pf4x8k --streams 4 --prompt-tokens 8192 --max-tokens 16 --seed 61
bash $L stop > $D/stop.log 2>&1; STARTED=0
grep -h "RHP01 rank=" <work-root>/flashnext/engine-*.log > $D/rhp01-lines.txt 2>/dev/null
cp $(ls -t <work-root>/flashnext/engine-*.log | head -1) $D/engine.log 2>/dev/null
echo ok > $D/status
