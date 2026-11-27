#!/bin/bash
# Flash-Next performance parity for a Radiance pin bump (.claude/skills/bump-radiance/SKILL.md section 4): lock in the
# current cut's cold prefill and engine step time before the bump, measure the bump build the same way, compare.
# Tokens/s is never compared across builds (MTP draft acceptance differs per build); acceptance is reported, never gated.
#
# Usage:
#   bash tools/lab/radiance/flashnext-parity.sh measure --build DIR --out DIR [--launches N] [--port P] [--native-config YAML]   (YAML: the config to copy; default flashnext-4gpu)
#   bash tools/lab/radiance/flashnext-parity.sh baseline --out DIR --name NAME --to FILE
#   bash tools/lab/radiance/flashnext-parity.sh compare --out DIR --baseline FILE
#
# measure: N launches (default 2) of native bc-serve serve flashnext-4gpu with production defaults (layout flashnext-4gpu, 262144
#   context, 8 sessions; the live expert-placement policy is on when $BC_WORK_ROOT/live-policy-calibration.toml exists),
#   handoff.histogram_period_s=1 from RSC12. Each launch warms up (only with the policy on:
#   (min_samples + 1) heat windows plus a second, from the native manifest), then measures cold prefill, decode step time at 1 and 6 streams, 8 sessions and a 256K
#   context, stops the native session (always, also on failure) and keeps the helper's `policy placement` line. Raw records go
#   to OUT/launch-<i>/ and each launch is committed as a run record (the SHA is OUT/launch-<i>/record.sha256; there is no
#   summary file). It uses all four cards: hold the
#   gpu lease (agent-lease acquire gpu ... -- bash tools/lab/radiance/flashnext-parity.sh measure ...). It runs in the
#   foreground for about 20 minutes a launch (measured 2026-11-08: ~7.5 min policy warmup, ~3.5 min cold/doc/decode/sess8,
#   ~3 min for the two 256K long rounds, plus start and stop); run it detached (setsid nohup ... > OUT.log 2>&1 &).
#   To isolate one change (a launcher, a flag) compare two arms of the SAME build in one session, A B A B with --launches 1
#   each: a stored baseline drifts a few percent from day to day (cold sess8: 28.9 vs 30.4 ms for one build a day apart).
#   For kernels without a model use tools/lab/radiance/kbench_ab.py (rad-kbench per row, two builds).
# baseline: `bigcherry results baseline` over the OUT's launch records into --to (e.g.
#   docs/reference/engines/radiance/baselines/flashnext-4gpu-parity.json); the baseline embeds each SHA-verified launch.
# compare: `bigcherry results compare` of the OUT's launch records with a baseline: one-sided gates and provenance
#   differences.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1
REPO=$(pwd)
PARITY=tools/lab/radiance/flashnext_parity.py

# The workload: this is the measurement's definition, identical before and after a bump (compare refuses a mismatch).
COLD_FILLS=8                  # cold single-stream prefills per launch
COLD_SEED0=501                # their seeds are COLD_SEED0 .. COLD_SEED0+COLD_FILLS-1
COLD_PROMPT_TOKENS=27000      # prompt size of a cold single-stream prefill
COLD_MAX_TOKENS=8             # completion tokens of a cold prefill request (TTFT is what is measured)
DOC_FILLS=6                   # long document prefills per launch, each at its own corpus offset
DOC_PROMPT_TOKENS=44000       # tokens of docs/**/*.md text in one document prefill (max_tokens 1)
DECODE_STREAMS="1 6"          # concurrent streams of the decode step-time rounds
DECODE_PROMPT_TOKENS=900      # prompt size of a decode round
DECODE_MAX_TOKENS=256         # completion tokens of a decode round
DECODE_REPS=2                 # rounds per decode stream count
DECODE_SEED=7                 # seed of the decode prompts
SESS_STREAMS=8                # concurrent sessions of the multi-session rounds
SESS_PROMPT_TOKENS=13600      # prompt size of a multi-session stream
SESS_MAX_TOKENS=32            # completion tokens of a multi-session stream
SESS_REPS=2                   # multi-session rounds (the first one is cold)
SESS_SEED=11                  # seed of the multi-session prompts
LONG_SEEDS="7 8"              # one long-context request set per seed
LONG_PROMPT_TOKENS=256000     # prompt size of a long-context request
LONG_MAX_TOKENS=256           # completion tokens of a long-context request
LONG_REPS=2                   # rounds per long-context seed (the first is the cold prefill)
WARMUP_PROMPT_TOKENS=27000    # prompt size of a warmup single-stream request
WARMUP_MAX_TOKENS=8           # completion tokens of a warmup request
WARMUP_SEED0=900              # warmup request k uses seed WARMUP_SEED0+k
RECORD_PERIOD=1               # the launcher's heat recording period in seconds, so the policy sees fresh heat
REQUEST_TIMEOUT_S=600         # client timeout of one request (stream-bench's own default)

usage() {
  sed -n '6,9p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 2
}

CMD=${1:-}
[ $# -gt 0 ] && shift
BUILD="" OUT="" LAUNCHES=2 PORT=18450 BASELINE="" NAME="" TO="" NATIVE_CONFIG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --build) [ $# -ge 2 ] || usage; BUILD=$2; shift 2 ;;
    --out) [ $# -ge 2 ] || usage; OUT=$2; shift 2 ;;
    --launches) [ $# -ge 2 ] || usage; LAUNCHES=$2; shift 2 ;;
    --port) [ $# -ge 2 ] || usage; PORT=$2; shift 2 ;;
    --native-config) [ $# -ge 2 ] || usage; NATIVE_CONFIG=$2; shift 2 ;;
    --baseline) [ $# -ge 2 ] || usage; BASELINE=$2; shift 2 ;;
    --name) [ $# -ge 2 ] || usage; NAME=$2; shift 2 ;;
    --to) [ $# -ge 2 ] || usage; TO=$2; shift 2 ;;
    *) echo "flashnext-parity: unknown argument: $1" >&2; usage ;;
  esac
done

case "$CMD" in
  baseline|compare)
    # The run records are the result. These two are `bigcherry results baseline|compare` over this OUT's launch receipts.
    [ -n "$OUT" ] || usage
    if [ "$CMD" = baseline ]; then [ -n "$NAME" ] && [ -n "$TO" ] || usage; else [ -n "$BASELINE" ] || usage; fi
    compgen -G "$OUT/launch-*/record.sha256" >/dev/null || { echo "flashnext-parity: $OUT has no launch records" >&2; exit 2; }
    source tools/env/bigcherry-env.sh >/dev/null 2>&1 || exit 2
    RECORD_ARGS=()
    for receipt in "$OUT"/launch-*/record.sha256; do
      [ -s "$receipt" ] || { echo "flashnext-parity: $receipt: no run record for this launch" >&2; exit 2; }
      RECORD_ARGS+=(--record "$(cat "$receipt")")
    done
    [ "${#RECORD_ARGS[@]}" -gt 0 ] || { echo "flashnext-parity: $OUT has no launch records" >&2; exit 2; }
    if [ "$CMD" = baseline ]; then
      exec env PYTHONPATH=tools python3 -m bigcherry results baseline --work-root "$BC_WORK_ROOT" --engine radiance \
        --layout flashnext-4gpu --workload flashnext-parity --name "$NAME" --to "$TO" "${RECORD_ARGS[@]}"
    fi
    exec env PYTHONPATH=tools python3 -m bigcherry results compare --work-root "$BC_WORK_ROOT" \
      --baseline "$BASELINE" "${RECORD_ARGS[@]}" ;;
  measure) ;;
  *) usage ;;
esac

[ -n "$BUILD" ] && [ -n "$OUT" ] || usage
[[ "$LAUNCHES" =~ ^[1-9][0-9]*$ ]] || { echo "flashnext-parity: --launches must be a positive integer" >&2; exit 2; }
[ -x "$BUILD/bin/radiance" ] || { echo "flashnext-parity: $BUILD has no bin/radiance" >&2; exit 2; }
source tools/env/bigcherry-env.sh >/dev/null 2>&1
[ -n "${BC_WORK_ROOT:-}" ] || { echo "flashnext-parity: BC_WORK_ROOT not set (tools/env/bigcherry-env.sh)" >&2; exit 1; }
RUN=$BC_WORK_ROOT/flashnext
URL=http://127.0.0.1:$PORT
BUILD=$(cd "$BUILD" && pwd -P)
BC_SERVE="$BUILD/radiance_home/bin/bc-serve"
[ -x "$BC_SERVE" ] || { echo "flashnext-parity: $BUILD has no native radiance_home/bin/bc-serve" >&2; exit 2; }
[ -n "${BC_MODEL_ROOT:-}" ] || { echo "flashnext-parity: BC_MODEL_ROOT not set" >&2; exit 2; }
# The native owner reads a config from its native YAML and takes no --set: an arm authors its own YAML (arm_yaml.py: a copy of
# the bundled or given config with the changed values) and passes it with --native-config.
NATIVE_ARGS=(--run-dir "$RUN" --model-root "$BC_MODEL_ROOT" --engine-binary "$BUILD/bin/radiance"
             --allow-power-apply
             --work-root "$BC_WORK_ROOT")   # the host-wide model-identity cache lives there
# The default is stock (observe only): bc-serve refuses a named non-stock profile without its rule file, and the stored
# baseline was taken with the cards' own settings. The power profile is explicit so a held GPU lease can apply only qualified,
# journalled settings; a missing lease or missing qualification fails closed.
# The owner derives the runtime home from --engine-binary.
# The helpers need the workload heat file (it has no portable path). The default is the Flash-Next mixed-workload heat,
# the launcher's default when the v1.17.0 baseline was taken (that run's own copy was lost; this is the 2026-11-18
# re-recording). perf51 (a different layout's recording) costs ~10% of 256K prefill on flashnext-4gpu
# (docs/evidence/2026-11-19-flashnext-4gpu-regression). FLASHNEXT_PARITY_HEAT overrides it for a heat comparison.
HEAT="${FLASHNEXT_PARITY_HEAT:-$REPO/tools/lab/radiance/heat/flashnext-mixed.hot-first}"
[ -f "$HEAT" ] || { echo "flashnext-parity: heat file $HEAT not found" >&2; exit 2; }
# FLASHNEXT_PARITY_SETS: extra space-separated PATH=VALUE changes for an experiment arm, as YAML paths of the config
# (models.flashnext.kv-cache-dtype=bf16_i8, x-bigcherry.helper.combine=false); the arm's YAML in OUT records them.
if [ -n "$NATIVE_CONFIG" ]; then BASE_YAML=$NATIVE_CONFIG
elif [ -f "$BUILD/configs/flashnext-4gpu.yaml" ]; then BASE_YAML=$BUILD/configs/flashnext-4gpu.yaml
else BASE_YAML=$REPO/engines/radiance/configs/flashnext-4gpu.yaml; fi
[ -f "$BASE_YAML" ] || { echo "flashnext-parity: no flashnext-4gpu.yaml for this build" >&2; exit 2; }
CALIBRATION_FILE=""
if [ -s "$BC_WORK_ROOT/live-policy-calibration.toml" ]; then
  CALIBRATION_FILE="$BC_WORK_ROOT/live-policy-calibration.toml"
  NATIVE_ARGS+=(--calibration "$CALIBRATION_FILE")
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

LONG_COUNT=$(wc -w <<< "$LONG_SEEDS")
PYTHONPATH=tools python3 "$PARITY" run-info "$OUT" --build "$BUILD" --launches "$LAUNCHES" --port "$PORT" \
  --set COLD_FILLS=$COLD_FILLS --set COLD_SEED0=$COLD_SEED0 --set COLD_PROMPT_TOKENS=$COLD_PROMPT_TOKENS \
  --set COLD_MAX_TOKENS=$COLD_MAX_TOKENS --set DOC_FILLS=$DOC_FILLS --set DOC_PROMPT_TOKENS=$DOC_PROMPT_TOKENS \
  --set "DECODE_STREAMS=$DECODE_STREAMS" --set DECODE_PROMPT_TOKENS=$DECODE_PROMPT_TOKENS \
  --set DECODE_MAX_TOKENS=$DECODE_MAX_TOKENS --set DECODE_REPS=$DECODE_REPS --set DECODE_SEED=$DECODE_SEED \
  --set SESS_STREAMS=$SESS_STREAMS --set SESS_PROMPT_TOKENS=$SESS_PROMPT_TOKENS --set SESS_MAX_TOKENS=$SESS_MAX_TOKENS \
  --set SESS_REPS=$SESS_REPS --set SESS_SEED=$SESS_SEED --set "LONG_SEEDS=$LONG_SEEDS" --set LONG_SEEDS_COUNT=$LONG_COUNT \
  --set LONG_PROMPT_TOKENS=$LONG_PROMPT_TOKENS --set LONG_MAX_TOKENS=$LONG_MAX_TOKENS --set LONG_REPS=$LONG_REPS \
  --set WARMUP_PROMPT_TOKENS=$WARMUP_PROMPT_TOKENS --set WARMUP_MAX_TOKENS=$WARMUP_MAX_TOKENS \
  --set WARMUP_SEED0=$WARMUP_SEED0 --set RECORD_PERIOD=$RECORD_PERIOD || exit

LAUNCHED=0
NATIVE_PID=
stop_native() {
  [ "$LAUNCHED" = 1 ] || return 0
  # Native stop verifies exit of each child; never kill a recorded PID.
  local result=0
  if "$BC_SERVE" stop --run-dir "$RUN" >> "$L/owner.log" 2>&1; then
    wait "$NATIVE_PID" ||
      { echo "flashnext-parity: owner exited with error (see $L/owner.log)" >&2; result=1; }
    LAUNCHED=0
  else
    echo "flashnext-parity: native stop failed (see $L/owner.log)" >&2
    result=1
  fi
  python3 "$PARITY" archive-session "$RUN/session.json" --owner-pid "$NATIVE_PID" --out "$L" || result=1
  return "$result"
}
trap 'stop_native' EXIT
trap 'stop_native; exit 130' INT TERM HUP
fail() { echo "flashnext-parity: launch $i: $*" >&2; stop_native; exit 1; }
wait_native_ready() {
  local until=$((SECONDS + 2700))
  while [ "$SECONDS" -lt "$until" ]; do
    if "$BC_SERVE" status --run-dir "$RUN" --json 2>/dev/null |
        python3 -c 'import json,sys
try:
    r=json.load(sys.stdin)
    assert r.get("state")=="running"
except (ValueError,AssertionError):
    sys.exit(1)' 2>/dev/null; then
      return 0
    fi
    kill -0 "$NATIVE_PID" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}
# Native session record owns the committed manifest path.
session_manifest() {
  python3 - "$RUN/session.json" "$NATIVE_PID" <<'PYTHON'
import json,sys
with open(sys.argv[1], encoding="utf-8") as f:
    r=json.load(f)
if r["owner_pid"] != int(sys.argv[2]):
    sys.exit(1)
print(r["manifest"])
PYTHON
}
# stream-bench NAME STREAMS PROMPT_TOKENS MAX_TOKENS REPS SEED: one record, raw JSON in $L/NAME.json, log in $L/NAME.log
stream_bench() {
  PYTHONPATH=tools python3 -m bigcherry stream-bench --url "$URL" --streams "$2" --prompt-tokens "$3" --max-tokens "$4" \
    --reps "$5" --temperature 0 --seed "$6" --timeout "$REQUEST_TIMEOUT_S" --output "$L/$1.json" > "$L/$1.log" 2>&1 \
    || fail "stream-bench $1 failed (see $L/$1.log)"
}

doc_prefill() {   # SLOT FILE
  python3 "$PARITY" doc-prefill --url "$URL" --calibration "$L/doc-calibration.json" --slot "$1" \
    --timeout "$REQUEST_TIMEOUT_S" --out "$2" || fail "document prefill $1 failed"
}

# true while the warmup time has not yet elapsed since t0
elapsed() { awk -v a="$t0" -v b="$EPOCHREALTIME" -v w="$WARMUP_S" 'BEGIN { exit !(b - a < w) }'; }

for i in $(seq 1 "$LAUNCHES"); do
  L=$OUT/launch-$i
  mkdir -p "$L/warmup"
  # Pin inputs before the native owner starts; verify they did not mutate.
  cp "$HEAT" "$L/heat-input.hot-first" || fail "cannot snapshot selected heat"
  if [ -n "$CALIBRATION_FILE" ]; then
    cp "$CALIBRATION_FILE" "$L/calibration-input.toml" || fail "cannot snapshot calibration store"
  fi
  OWNER_ARGS=("${NATIVE_ARGS[@]}")
  # Route-heat recording at the period the live policy needs (RSC12), into the run's own folder.
  ARM_SETS=(--set "x-bigcherry.power.profile=stock" --set "server.port=$PORT" --set "x-bigcherry.handoff.heat=$HEAT"
            --set "x-bigcherry.handoff.histogram=$OUT/heat.live" --set "x-bigcherry.handoff.histogram-period-s=1")
  for s in ${FLASHNEXT_PARITY_SETS:-}; do ARM_SETS+=(--set "$s"); done
  python3 tools/lab/radiance/arm_yaml.py --base "$BASE_YAML" --out "$L/native-input.yaml" "${ARM_SETS[@]}" ||
    fail "cannot write this launch's arm YAML"
  OWNER_ARGS+=(--native-config "$L/native-input.yaml")
  CAPTURE_ARGS=(--heat "$L/heat-input.hot-first")
  if [ -n "$CALIBRATION_FILE" ]; then CAPTURE_ARGS+=(--calibration "$L/calibration-input.toml"); fi
  python3 "$PARITY" capture-provenance "${CAPTURE_ARGS[@]}" "$L" -- "$BC_SERVE" serve flashnext-4gpu "${OWNER_ARGS[@]}" ||
    fail "cannot capture native owner invocation"
  echo "flashnext-parity: launch $i/$LAUNCHES -> $L" >&2
  "$BC_SERVE" serve flashnext-4gpu "${OWNER_ARGS[@]}" > "$L/owner.log" 2>&1 &
  NATIVE_PID=$!
  LAUNCHED=1
  wait_native_ready || fail "native bc-serve refused or exited (see $L/owner.log)"
  MANIFEST=$(session_manifest) || fail "native session has no committed manifest"
  [ -f "$MANIFEST" ] || fail "native session did not commit a serving manifest"
  cp "$MANIFEST" "$L/manifest.json" || fail "cannot archive the manifest"
  cp "$RUN/feature-states.v1.json" "$L/feature-states.v1.json" ||
    fail "native MRC03 feature-state snapshot is absent"
  cmp "$HEAT" "$L/heat-input.hot-first" ||
    fail "selected heat mutated before measurement"
  if [ -n "$CALIBRATION_FILE" ]; then
    cmp "$CALIBRATION_FILE" "$L/calibration-input.toml" ||
      fail "selected calibration changed before measurement"
  fi
  python3 "$PARITY" warmup-plan "$L/manifest.json" --out "$L/launch.json" >&2 || fail "cannot read the manifest's live policy"
  WARMUP_S=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["warmup_s"])' "$L/launch.json")
  # FLASHNEXT_PARITY_WARMUP_S: an experiment arm's warmup in seconds, used only when the policy is on and only if it is
  # longer than the derived one (never shortens the policy's own sampling period). The log line below states it.
  if [ -n "${FLASHNEXT_PARITY_WARMUP_S:-}" ] && awk -v w="$WARMUP_S" -v o="$FLASHNEXT_PARITY_WARMUP_S" 'BEGIN { exit !(w > 0 && o > w) }'; then
    WARMUP_S=$FLASHNEXT_PARITY_WARMUP_S
  fi

  python3 "$PARITY" doc-calibrate --url "$URL" --tokens "$DOC_PROMPT_TOKENS" --measured "$DOC_FILLS" \
    --timeout "$REQUEST_TIMEOUT_S" --out "$L/doc-calibration.json" || fail "document calibration failed"

  # warmup: alternate a single-stream request and a document prefill until the policy has had its windows
  t0=$EPOCHREALTIME; k=0
  while elapsed; do
    stream_bench "warmup/stream-$k" 1 "$WARMUP_PROMPT_TOKENS" "$WARMUP_MAX_TOKENS" 1 $((WARMUP_SEED0 + k))
    elapsed || break
    doc_prefill "warmup:$k" "$L/warmup/doc-$k.json"
    k=$((k + 1))
  done
  WARMUP_ELAPSED_S=$(awk -v a="$t0" -v b="$EPOCHREALTIME" 'BEGIN { printf "%.6f", b-a }')
  echo "flashnext-parity: launch $i warmup ${WARMUP_S}s, $k rounds (elapsed ${WARMUP_ELAPSED_S}s)" >&2
  # Actual completed warmup, including any longer experimental override.
  python3 -c 'import json,sys; from pathlib import Path
p=Path(sys.argv[1]); data=json.loads(p.read_text())
data["actual_warmup_s"]=float(sys.argv[2]); data["warmup_target_s"]=float(sys.argv[3])
data["warmup_rounds"]=int(sys.argv[4])
p.write_text(json.dumps(data,sort_keys=True)+chr(10))' "$L/launch.json" "$WARMUP_ELAPSED_S" "$WARMUP_S" "$k" ||
    fail "cannot record actual warmup"
  python3 "$PARITY" capture-host-facts "$L" --run-dir "$RUN" ||
    fail "cannot pin sealed native owner start receipt"

  # a. cold prefill
  for s in $(seq "$COLD_SEED0" $((COLD_SEED0 + COLD_FILLS - 1))); do
    stream_bench "cold-fill-$s" 1 "$COLD_PROMPT_TOKENS" "$COLD_MAX_TOKENS" 1 "$s"
  done
  for d in $(seq 0 $((DOC_FILLS - 1))); do doc_prefill "measure:$d" "$L/doc-prefill-$d.json"; done
  # b. decode step time
  for n in $DECODE_STREAMS; do
    stream_bench "decode$n" "$n" "$DECODE_PROMPT_TOKENS" "$DECODE_MAX_TOKENS" "$DECODE_REPS" "$DECODE_SEED"
  done
  # c. multi-session
  stream_bench sess8 "$SESS_STREAMS" "$SESS_PROMPT_TOKENS" "$SESS_MAX_TOKENS" "$SESS_REPS" "$SESS_SEED"
  # d. long context
  for s in $LONG_SEEDS; do
    stream_bench "long-$s" 1 "$LONG_PROMPT_TOKENS" "$LONG_MAX_TOKENS" "$LONG_REPS" "$s"
  done

  stop_native || fail "native session shutdown or evidence collection failed"
  if [ -f "$L/helper0.log" ]; then
    grep 'policy placement' "$L/helper0.log" | tail -1 > "$L/policy-placement.txt" || :
  fi
  python3 "$PARITY" policy-line "$L/helper0.log" --out "$L/policy.json" >&2
  cmp "$HEAT" "$L/heat-input.hot-first" ||
    fail "selected heat changed during measurement"
  if [ -n "$CALIBRATION_FILE" ]; then
    cmp "$CALIBRATION_FILE" "$L/calibration-input.toml" ||
      fail "selected calibration changed during measurement"
  fi
  python3 "$PARITY" record-launch "$OUT" --launch "$i" --work-root "$BC_WORK_ROOT" > "$L/record.sha256" ||
    fail "run-record v1 publication failed"
  echo "flashnext-parity: launch $i record $(cat "$L/record.sha256")" >&2
done
trap - EXIT INT TERM HUP

echo "flashnext-parity: run records: $(cat "$OUT"/launch-*/record.sha256 | tr '\n' ' ')" >&2
echo "flashnext-parity: compare with: bash tools/lab/radiance/flashnext-parity.sh compare --out $OUT --baseline FILE" >&2
