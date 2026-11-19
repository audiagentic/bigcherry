#!/bin/bash
# one launch of a dense TP3 config on a main build: TTFT sweep (one stream, distinct prompts) then decode at 1/4/6/8 streams.
# usage: one.sh BUILD OUT CONFIG
set -u
BUILD=$1; OUT=$2; CFG=$3
REPO=<dev-root>/projects/bigcherry/workspaces/main.worktrees/tp3-noregress-check
cd "$REPO" || exit 1
set --
source tools/env/bigcherry-env.sh >/dev/null 2>&1
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
RUN=${RUN_DIR:-$BC_WORK_ROOT/tp3check-$(basename "$OUT")}; PORT=18453; URL=http://127.0.0.1:$PORT
BC_SERVE="$BUILD/radiance_home/bin/bc-serve"
ARGS=(--run-dir "$RUN" --model-root "$BC_MODEL_ROOT" --engine-binary "$BUILD/bin/radiance" --set "engine.port=$PORT"
      --power-profile stock --allow-power-apply)
[ -f "$BUILD/configs/qwen27b.toml" ] || ARGS+=(--configs "$REPO/engines/radiance/configs")
[ -d "$BUILD/radiance_home" ] && ARGS+=(--set "engine.radiance_home=$BUILD/radiance_home")
"$BC_SERVE" serve "$CFG" "${ARGS[@]}" > "$OUT/owner.log" 2>&1 &
PID=$!; echo "config=$CFG owner pid=$PID start=$(date -Is)" > "$OUT/run.log"
stop() { "$BC_SERVE" stop --run-dir "$RUN" >> "$OUT/owner.log" 2>&1; wait $PID; echo "owner exit=$? at $(date -Is)" >> "$OUT/run.log"; }
trap stop EXIT
until=$((SECONDS+2400)); state=""
while [ $SECONDS -lt $until ]; do
  state=$("$BC_SERVE" status --run-dir "$RUN" --json 2>/dev/null | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("state",""))
except Exception: print("")')
  [ "$state" = running ] && break
  kill -0 $PID 2>/dev/null || { echo "owner exited early: state=$state" >> "$OUT/run.log"; exit 1; }
  sleep 2
done
echo "state=$state after ${SECONDS}s" >> "$OUT/run.log"
[ "$state" = running ] || exit 1
# manifest of this launch, for the effective engine argv
python3 - "$RUN/session.json" "$OUT" <<'PY'
import json,sys,shutil
s=json.load(open(sys.argv[1])); shutil.copy(s["manifest"], sys.argv[2]+"/manifest.json")
PY
if [ "${QUICK:-0}" = 1 ]; then sleep 15; echo "QUICK DONE $(date -Is)" >> "$OUT/run.log"; exit 0; fi
bench() { # NAME STREAMS PROMPT_TOKENS MAX_TOKENS REPS SEED
  PYTHONPATH=tools python3 -m bigcherry stream-bench --url "$URL" --streams "$2" --prompt-tokens "$3" --max-tokens "$4" --reps "$5" \
    --temperature 0 --seed "$6" --timeout 900 --output "$OUT/$1.json" > "$OUT/$1.log" 2>&1 || echo "bench $1 FAILED" >> "$OUT/run.log"
  kill -0 $PID 2>/dev/null || { echo "owner gone after $1" >> "$OUT/run.log"; exit 2; }
}
bench warm 1 512 8 1 99                                  # first request after start is not measured
i=0
for n in 512 2048 8003 30000 60000 100366; do i=$((i+1)); bench ttft-$n 1 $n 8 1 $((300+i)); done
bench decode 1,4,6,8 512 256 2 7
echo "BENCH DONE $(date -Is)" >> "$OUT/run.log"
