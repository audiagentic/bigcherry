#!/bin/bash
# Flash-Next on two XTX with one libr3 switch off and on: build libr3, then a profiled and an unprofiled serve bench
# for each arm (flashnext-tp2.sh). For a new native form that has its own selftest, run that first; this script only
# says what the switch does to the served model.
# Start it directly (it queues its own jobs):   bash tools/lab/radiance/fn-switch-ab.sh <tag> <SWITCH> [SWITCH...]
#   e.g. fn-switch-ab.sh fnhc1 R3_HC_NATIVE          (arms: every switch 0, every switch 1)
# env: R (runs root), BUILD (b-main2), ARMS ("0 1"), PROFILED (1: also a --profile-ops run an arm), and
#      flashnext-tp2.sh's own (CTX, POOL_MIB, SPEC, DEPTHS, MORE, ...)
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
R=${R:-/mnt/data/bigcherry-work/runs}
tag=${1:?tag}; shift
[ $# -ge 1 ] || { echo "FN_SWITCH_AB_STOP no switch named"; exit 2; }
b=${BUILD:-b-main2}
jobs=$(mktemp)
rm -rf "$R/$tag-build.log" "$R/$tag-build"
echo "VIS=0 SCRIPT $tag-build tools/lab/radiance/libr3-build.sh @$b $R/$tag-build" > "$jobs"
env KBENCH=0 GPU=0 bash tools/lab/plan-qualification/queue.sh "$jobs" > /dev/null 2>&1
rm -f "$jobs"
tail -4 "$R/$tag-build.log" | cut -c1-200
grep -q "^SCRIPT_EXIT=0" "$R/$tag-build.log" || { echo "FN_SWITCH_AB_STOP build failed"; exit 1; }
for arm in ${ARMS:-0 1}; do
    set_env=()
    for s in "$@"; do set_env+=("$s=$arm"); done
    echo "######## arm $arm [${set_env[*]}]"
    env "${set_env[@]}" bash tools/lab/radiance/flashnext-tp2.sh "$tag-a$arm" | grep -vE "^FLASHNEXT_DONE"
    grep -qE "^decode, geometric" "$R/$tag-a$arm.log" || { echo "FN_SWITCH_AB_STOP arm $arm served nothing"; exit 1; }
    if [ "${PROFILED:-1}" = 1 ]; then
        env "${set_env[@]}" PROFILE=1 bash tools/lab/radiance/flashnext-tp2.sh "$tag-p$arm" > /dev/null
        python3 tools/lab/radiance/profile_ops_summary.py "$R/$tag-p$arm/server.log" 2>&1 | head -14 | cut -c1-150
    fi
done
echo "FN_SWITCH_AB_DONE $tag"
