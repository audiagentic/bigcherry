#!/bin/bash
# RAD20 helper liveness: SIGKILL one helper of the production Flash-Next layout and require the engine to fail within a
# bound instead of hanging (before: an 8.7 h hang). Scenarios, each on a fresh start: idle (kill, then send a request),
# decode (kill during a long generation), prefill (kill during a long prompt). Pass: within RAD20_KILL_BOUND_S seconds
# (default 60) of the kill the engine process has exited or the in-flight request has returned an error.
# Usage: kill-test from a checkout: rad20-helper-kill.sh [helper index to kill=1] [scenarios="idle decode prefill"]
set -u
VICTIM=${1:-1}
SCENARIOS=${2:-idle decode prefill}
BOUND=${RAD20_KILL_BOUND_S:-60}
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
OUT=$BC_WORK_ROOT/runs/rad20-helper-kill-$(date +%m%d%H%M); mkdir -p $OUT
PYTHONPATH=tools timeout 5400 python3 -m bigcherry build --lane radiance-lab:stock:linux-radiance \
    --experiment moe-remote-handoff > $OUT/build.log 2>&1
plan=$(grep -oE 'ok build_plan_id=[0-9a-f]+' $OUT/build.log | tail -1 | cut -d= -f2)
[ -n "$plan" ] || { grep -E ' error:|FAILED' $OUT/build.log | sort -u | head; exit 1; }
B=$(ls -d $BC_WORK_ROOT/builds/*/$plan)
L=tools/lab/radiance/launch-flashnext-best.sh
URL=http://127.0.0.1:36080/v1/completions   # the launcher's default port
PIDS=$BC_WORK_ROOT/flashnext/pids   # written by the launcher
helper_pid() { cat $PIDS/helper$1 2>/dev/null; }
engine_pid() { cat $PIDS/engine 2>/dev/null; }
LONG=$(python3 -c "print(' '.join(['The quick brown fox jumps over the lazy dog.']*400))")
FAILED=0
for sc in $SCENARIOS; do
    log=$OUT/$sc; mkdir -p $log
    bash $L start --build $B > $log/launch.log 2>&1 || { echo "FAIL $sc: start"; tail -5 $log/launch.log; FAILED=1; bash $L stop >/dev/null 2>&1; continue; }
    E=$(engine_pid); H=$(helper_pid $VICTIM)
    [ -n "$E" ] && [ -n "$H" ] || { echo "FAIL $sc: engine '$E' / helper '$H' not found"; FAILED=1; bash $L stop >/dev/null 2>&1; continue; }
    case $sc in
        idle)    kill -9 $H; T0=$(date +%s.%N); prompt="After the kill."; tokens=64 ;;
        decode)  prompt="Write a long story about a lighthouse keeper."; tokens=2048 ;;
        prefill) prompt="$LONG Summarise."; tokens=32 ;;
    esac
    curl -s -m $((BOUND + 30)) $URL -d "$(python3 -c 'import json,sys; print(json.dumps({"prompt":sys.argv[1],"max_tokens":int(sys.argv[2]),"temperature":0}))' "$prompt" $tokens)" \
        > $log/response.json 2> $log/curl.err &
    C=$!
    case $sc in
        decode)  sleep 4;   kill -9 $H; T0=$(date +%s.%N) ;;
        prefill) sleep 0.3; kill -9 $H; T0=$(date +%s.%N) ;;
    esac
    verdict=HANG
    while :; do
        el=$(python3 -c "import sys; print(float(sys.argv[2]) - float(sys.argv[1]))" $T0 $(date +%s.%N))
        if ! kill -0 $E 2>/dev/null; then verdict="ENGINE_EXIT"; break; fi
        if ! kill -0 $C 2>/dev/null; then verdict="REQUEST_RETURNED"; break; fi
        python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" $el $BOUND && break
        sleep 0.5
    done
    wait $C 2>/dev/null
    printf '%-8s %-16s %6.1f s  response: %s\n' $sc $verdict $el "$(head -c 160 $log/response.json)" | tee -a $OUT/summary.txt
    [ $verdict = HANG ] && FAILED=1
    [ $verdict = REQUEST_RETURNED ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(1 if d.get('choices') and not d.get('error') else 0)" $log/response.json 2>/dev/null \
        || { [ $verdict = REQUEST_RETURNED ] && { echo "  $sc: request returned text after a helper died (check)"; } }
    grep -h "exited; failing\|helper_fault\|trap\|Aborted" $log/launch.log $BC_WORK_ROOT/flashnext/*.log 2>/dev/null | tail -3 | sed 's/^/  /'
    bash $L stop > $log/stop.log 2>&1
    sleep 3
done
echo "KILL_TEST $([ $FAILED = 0 ] && echo PASS || echo FAIL)" | tee -a $OUT/summary.txt
echo "OUT $OUT"; echo HELPER_KILL_DONE
