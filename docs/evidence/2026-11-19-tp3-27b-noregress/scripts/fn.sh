#!/bin/bash
# flashnext-4gpu, two consecutive policy-on starts in ONE run dir (same --sets), 20 s apart; the second must replay.
D=<dev-root>/tmp/bigcherry/tp3-check
BUILD=$D/fixbuild
REPO=<dev-root>/projects/bigcherry/workspaces/main.worktrees/s8-inventory-identity
cd "$REPO" || exit 1
set --
source tools/env/bigcherry-env.sh >/dev/null 2>&1
RUN=$BC_WORK_ROOT/fncheck-replay; rm -rf "$RUN"
BC_SERVE="$BUILD/radiance_home/bin/bc-serve"; PORT=18454
for n in 1 2; do
  for w in $(seq 1 60); do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$a" -ge 78 ] && break; sleep 5; done
  OUT=$D/f$n; mkdir -p $OUT
  ARGS=(--run-dir "$RUN" --model-root "$BC_MODEL_ROOT" --engine-binary "$BUILD/bin/radiance" --set "engine.port=$PORT" --power-profile stock --allow-power-apply
        --configs "$REPO/engines/radiance/configs" --set "engine.radiance_home=$BUILD/radiance_home"
        --set "handoff.heat=<dev-root>/projects/bigcherry/workspaces/main/tools/lab/radiance/heat/flashnext-mixed.hot-first"
        --calibration "$BC_WORK_ROOT/live-policy-calibration.toml" --set "handoff.histogram=$D/f-heat.live" --set "handoff.histogram_period_s=1")
  "$BC_SERVE" serve flashnext-4gpu "${ARGS[@]}" > $OUT/owner.log 2>&1 &
  PID=$!; echo "start $n pid=$PID at $(date -Is) MemAvailable=${a}GiB" >> $OUT/run.log
  until=$((SECONDS+1500)); state=""
  while [ $SECONDS -lt $until ]; do
    state=$("$BC_SERVE" status --run-dir "$RUN" --json 2>/dev/null | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("state",""))
except Exception: print("")')
    [ "$state" = running ] && break
    kill -0 $PID 2>/dev/null || break
    sleep 2
  done
  echo "state=$state at $(date -Is)" >> $OUT/run.log
  [ "$state" = running ] && sleep 20
  "$BC_SERVE" stop --run-dir "$RUN" >> $OUT/owner.log 2>&1; wait $PID; echo "start $n exit=$? at $(date -Is)" >> $D/fn.log
  sleep 20
done
