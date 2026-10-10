#!/bin/bash
# RR07/RR08/RR10: one edit-measure cycle for libr3 on the RX 7900 XTX, through the queue: build, numeric selftest of
# the MXFP4 decode row with the kernel under test switched on, then serve runs. Each step prints one or two lines.
# Start it directly (it queues its own jobs), from the checkout whose libr3 is to be built:
#   bash tools/lab/radiance/libr3-cycle.sh <tag> [serve-config...]
# A serve config is  name:VAR=value,VAR=value  (the environment of that serve run; nothing after ':' = defaults).
# A GPU list is written with '+': GPU=0+1. A config with TP=2 locks both cards.
# Default configs: base (libr4d's staged form), dec11, dec11 with the drafter, two cards, two cards with dec11,
# two cards with dec11 and the drafter.
# env: R (runs root), BUILD (any build run id for the queue line, b-main2), SELFTEST_ENV (R3_DEC11=1), N (128)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}; shift
configs=("$@")
[ ${#configs[@]} -eq 0 ] && configs=(base: dec11:R3_DEC11=1 dec11spec:R3_DEC11=1,SPEC=auto tp2:TP=2,GPU=0+1
                                     tp2dec11:R3_DEC11=1,TP=2,GPU=0+1 tp2dec11spec:R3_DEC11=1,TP=2,GPU=0+1,SPEC=auto)
b=${BUILD:-b-main2}
jobs=$(mktemp)
ENVS=""
one() {  # <vis> <run> <script> [op...]; the run's extra environment is in $ENVS
    local vis=$1 run=$2 script=$3; shift 3
    rm -rf "$R/$run" "$R/$run.log"
    echo "VIS=$vis SCRIPT $run $script @$b $R/$run $*" > "$jobs"
    env $ENVS bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1
}
echo "== $tag at $(git rev-parse --short HEAD)"
# `add` keeps libr3-build.sh's checker to one op; the selftest below is the check for the MXFP4 rows
ENVS=""; one 0 "$tag-build" tools/lab/radiance/libr3-build.sh add
echo "build: $(grep -E "build exit|BUILD_FAILED|CONFIGURE_FAILED" "$R/$tag-build.log" | head -1 | cut -c1-80)"
if ! grep -q "build exit 0" "$R/$tag-build.log"; then
    grep -E "error:" "$R/$tag-build/build.log" | sed -E 's#.*/([^/:]+:[0-9]+):[0-9]+: #\1 #' | sort | uniq -c | head -8 | cut -c1-200
    rm -f "$jobs"; echo "CYCLE_DONE $tag"; exit 1
fi
ENVS=${SELFTEST_ENV:-R3_DEC11=1}; one 0 "$tag-self" tools/lab/radiance/libr3-selftest.sh gemm_mxfp4a8_decode
echo "selftest [$ENVS]: $(grep -E "^-- gemm_mxfp4a8_decode|verdict" "$R/$tag-self.log" | tr '\n' ' ' | cut -c1-140)"
grep -E "FAIL" "$R/$tag-self/gemm_mxfp4a8_decode.log" 2> /dev/null | head -4 | cut -c1-200
for cfg in "${configs[@]}"; do
    name=${cfg%%:*}
    ENVS="N=${N:-128} BENCH=${BENCH:-1} $(echo "${cfg#*:}" | tr ',' ' ' | tr '+' ',')"
    vis=$(echo " $ENVS " | grep -oE " GPU=[0-9,]+ " | tr -d ' ' | cut -d= -f2); vis=${vis:-0}  # lock the cards it runs on
    one "$vis" "$tag-$name" tools/lab/radiance/libr3-serve-smoke.sh
    log=$R/$tag-$name.log
    ok=$(grep -c "Berlin" "$log" 2> /dev/null)
    grep -E "decode, geometric mean|prefill at" "$log" 2> /dev/null | sed "s/^/    /" | cut -c1-170
    echo "$name [$ENVS]: smoke $(grep -E "^I req" "$log" 2> /dev/null | grep -oE "[0-9.]+ tok/s" | tr '\n' ' ')text $([ "${ok:-0}" -ge 1 ] && echo ok || echo WRONG) $(grep -E "SERVER_EXITED|LOAD_TIMEOUT" "$log" 2> /dev/null | head -1 | cut -c1-60)"
done
rm -f "$jobs"
echo "CYCLE_DONE $tag"
