#!/bin/bash
# one vision mode on TP3 27B: start, record VRAM and engine media/capacity lines, send an image request, stop.
# usage: mode.sh NAME CONFIG [extra --set ...]
set -u
NAME=$1; CFG=$2; shift 2
D=<dev-root>/tmp/bigcherry/media; OUT=$D/$NAME; mkdir -p $OUT
W=<dev-root>/projects/bigcherry/workspaces/main.worktrees/radiance-media-build
BUILD=<work-root>/builds/36aef34754713df1487f4e97febd74bb/7be93669d8cae9ef44f049ef3f6419a4
cd $W; set --; source tools/env/bigcherry-env.sh >/dev/null 2>&1
RUN=$BC_WORK_ROOT/media-$NAME; PORT=18454; URL=http://127.0.0.1:$PORT
BC_SERVE="$BUILD/radiance_home/bin/bc-serve"
ARGS=(--run-dir "$RUN" --model-root "$BC_MODEL_ROOT" --engine-binary "$BUILD/bin/radiance" --set "engine.port=$PORT" --power-profile stock --allow-power-apply)
[ -f "$BUILD/configs/qwen27b.toml" ] || ARGS+=(--configs "$W/engines/radiance/configs")
[ -d "$BUILD/radiance_home" ] && ARGS+=(--set "engine.radiance_home=$BUILD/radiance_home")
for x in ${EXTRA_SETS:-}; do ARGS+=(--set "$x"); done
"$BC_SERVE" serve "$CFG" "${ARGS[@]}" > $OUT/owner.log 2>&1 &
PID=$!; echo "mode=$NAME cfg=$CFG sets=${EXTRA_SETS:-} pid=$PID start=$(date -Is)" > $OUT/run.log
stop() { "$BC_SERVE" stop --run-dir "$RUN" >> $OUT/owner.log 2>&1; wait $PID; echo "owner exit=$? $(date -Is)" >> $OUT/run.log; }
trap stop EXIT
st=""; until=$((SECONDS+2400))
while [ $SECONDS -lt $until ]; do
  st=$("$BC_SERVE" status --run-dir "$RUN" --json 2>/dev/null | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("state",""))
except Exception: print("")')
  [ "$st" = running ] && break
  kill -0 $PID 2>/dev/null || { echo "owner exited early" >> $OUT/run.log; exit 1; }
  sleep 2
done
echo "state=$st after ${SECONDS}s" >> $OUT/run.log
[ "$st" = running ] || exit 1
sleep 5
rocm-smi --showmeminfo vram --json > $OUT/vram.json 2>/dev/null
python3 - "$RUN/session.json" $OUT <<'PY'
import json,sys,shutil
s=json.load(open(sys.argv[1])); shutil.copy(s["manifest"], sys.argv[2]+"/manifest.json")
PY
find $RUN -name "*.log" | head -5 > $OUT/logs.txt
python3 $W/tools/lab/radiance/image_probe.py --url $URL --expect "${EXPECT:-answered}" --out $OUT/image.json > $OUT/image.log 2>&1
echo "image rc=$? $(date -Is)" >> $OUT/run.log
