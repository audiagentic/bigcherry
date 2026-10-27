#!/bin/bash
# RAD20 helper liveness: SIGKILL one helper of the production Flash-Next layout and require the engine to fail the
# in-flight (or next) request with an error within a bound instead of hanging (before: an 8.7 h hang). Each scenario
# runs on a fresh start:
#   idle     kill, then send a request
#   decode   stream a long generation; kill once tokens are arriving (decode confirmed in flight)
#   prefill  a ~16k-token prompt (several seconds of prefill); kill 0.5 s after it is sent
#   park     the victim stops itself (SIGSTOP, R3_MOE_WORKER_TEST_STOP_AT=park_drain:<i>) while draining a lane after a
#            burst; kill it there, then send a request (the engine finds the lane still flagged draining)
# A scenario passes only when: the kill succeeded, the engine logged the watcher's fault for that pid, and the request
# ended with an error (not text) within RAD20_KILL_BOUND_S (default 60) of the kill. Whether the engine process then
# exits is recorded separately (by design it stays up, refusing requests, until stopped).
# Usage (from a checkout): rad20-helper-kill.sh [victim helper indexes="1 0"] [scenarios="idle decode prefill park"]
set -u
VICTIMS=${1:-1 0}
SCENARIOS=${2:-idle decode prefill park}
BOUND=${RAD20_KILL_BOUND_S:-60}
set --   # the env script reads its own arguments
source tools/env/bigcherry-env.sh >/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/lab-env.sh"
OUT=$BC_WORK_ROOT/runs/rad20-helper-kill-$(date +%m%d%H%M); mkdir -p $OUT
git log -1 --format='commit %H %s' > $OUT/summary.txt
source tools/lab/radiance/lab-build.sh
B=$(rad_build $OUT/build.log) || exit 1
echo "build $B" >> $OUT/summary.txt
L=tools/lab/radiance/launch-flashnext-best.sh
RUN=$BC_WORK_ROOT/flashnext   # the launcher's pid files and logs
URL=http://127.0.0.1:36080/v1/completions   # the launcher's default port
LONG=$(python3 -c "print(' '.join(['The quick brown fox jumps over the lazy dog.']*1600))")
body() { python3 -c 'import json,sys; print(json.dumps({"prompt":sys.argv[1],"max_tokens":int(sys.argv[2]),"temperature":0,"stream":sys.argv[3]=="1"}))' "$@"; }
now() { date +%s.%N; }
FAILED=0
for victim in $VICTIMS; do
for sc in $SCENARIOS; do
    log=$OUT/h$victim-$sc; mkdir -p $log
    stop_at=""; [ $sc = park ] && stop_at="R3_MOE_WORKER_TEST_STOP_AT=park_drain:$victim"
    env $stop_at bash $L start --build $B > $log/launch.log 2>&1 || { echo "FAIL h$victim $sc: start" | tee -a $OUT/summary.txt; FAILED=1; bash $L stop >/dev/null 2>&1; continue; }
    E=$(cat $RUN/pids/engine); H=$(cat $RUN/pids/helper$victim)
    elog=$(ls -t $RUN/engine-*.log | head -1)
    case $sc in
        idle)
            kill -9 $H && T0=$(now) || { echo "FAIL h$victim $sc: kill" | tee -a $OUT/summary.txt; FAILED=1; }
            curl -s -m $BOUND $URL -d "$(body 'After the kill.' 64 0)" > $log/response.txt 2>&1 & C=$! ;;
        decode)
            curl -sN -m $((BOUND + 120)) $URL -d "$(body 'Write a long story about a lighthouse keeper.' 2048 1)" > $log/response.txt 2>&1 & C=$!
            for i in $(seq 600); do grep -q '"text"' $log/response.txt 2>/dev/null && break; sleep 0.2; done
            grep -q '"text"' $log/response.txt || { echo "FAIL h$victim $sc: no tokens before the kill (decode not confirmed)" | tee -a $OUT/summary.txt; FAILED=1; }
            kill -9 $H && T0=$(now) || { echo "FAIL h$victim $sc: kill" | tee -a $OUT/summary.txt; FAILED=1; } ;;
        prefill)
            curl -s -m $((BOUND + 120)) $URL -d "$(body "$LONG Summarise." 32 0)" > $log/response.txt 2>&1 & C=$!
            sleep 0.5
            kill -9 $H && T0=$(now) || { echo "FAIL h$victim $sc: kill" | tee -a $OUT/summary.txt; FAILED=1; } ;;
        park)
            curl -s -m 120 $URL -d "$(body 'One short burst.' 16 0)" > $log/first.txt 2>&1   # the burst ends -> park
            for i in $(seq 300); do [ "$(ps -o stat= -p $H 2>/dev/null | cut -c1)" = T ] && break; sleep 0.2; done
            [ "$(ps -o stat= -p $H 2>/dev/null | cut -c1)" = T ] || { echo "FAIL h$victim $sc: helper never stopped at park_drain" | tee -a $OUT/summary.txt; FAILED=1; }
            kill -9 $H && T0=$(now) || { echo "FAIL h$victim $sc: kill" | tee -a $OUT/summary.txt; FAILED=1; }
            curl -s -m $BOUND $URL -d "$(body 'After the kill.' 64 0)" > $log/response.txt 2>&1 & C=$! ;;
    esac
    ended=""
    while :; do
        el=$(python3 -c "import sys; print(round(float(sys.argv[2]) - float(sys.argv[1]), 1))" $T0 $(now))
        kill -0 $C 2>/dev/null || { ended=1; break; }
        python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" $el $BOUND && break
        sleep 0.2
    done
    [ -n "$ended" ] || { kill $C 2>/dev/null; wait $C 2>/dev/null; }
    engine_alive=$(kill -0 $E 2>/dev/null && echo yes || echo no)
    watcher=$(grep -c "helper $victim (pid $H) exited" $elog)
    # error ended: a JSON error object, or a stream whose last event is an error / no [DONE]
    errored=$(python3 - $log/response.txt $sc <<'PY'
import json, sys
t = open(sys.argv[1]).read().strip()
if sys.argv[2] == "decode":
    print("yes" if t and ("[DONE]" not in t or '"error"' in t) else "no")
else:
    try: d = json.loads(t); print("yes" if d.get("error") and not d.get("choices") else "no")
    except Exception: print("no")
PY
)
    ok=PASS
    [ -n "$ended" ] || ok="FAIL(hang)"
    [ "$errored" = yes ] || ok="FAIL(no error)"
    [ "$watcher" -ge 1 ] || ok="FAIL(no watcher log)"
    [ $ok = PASS ] || FAILED=1
    printf 'h%s %-8s %-18s %5.1f s  watcher %s  engine_alive %s  %s\n' $victim $sc "$ok" $el $watcher $engine_alive \
        "$(grep -oE '"message":"[^"]{0,120}' $log/response.txt | head -1)" | tee -a $OUT/summary.txt
    cp $elog $log/ 2>/dev/null; cp $RUN/helper*-$(basename $elog | sed 's/engine-//') $log/ 2>/dev/null
    bash $L stop > $log/stop.log 2>&1
    sleep 3
done
done
echo "KILL_TEST $([ $FAILED = 0 ] && echo PASS || echo FAIL)" | tee -a $OUT/summary.txt
echo "OUT $OUT"; echo HELPER_KILL_DONE
