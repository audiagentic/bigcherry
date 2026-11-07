#!/bin/bash
# Flash-Next performance parity for a Radiance pin bump (.claude/skills/bump-radiance/SKILL.md section 4): lock in the
# current cut's cold prefill and engine step time before the bump, measure the bump build the same way, compare.
# Tokens/s is never compared across builds (MTP draft acceptance differs per build); acceptance is reported, never gated.
#
# Usage:
#   bash tools/lab/radiance/flashnext-parity.sh measure --build DIR --out DIR [--launches N] [--port P]
#   bash tools/lab/radiance/flashnext-parity.sh baseline --out DIR --name NAME [--to FILE]
#   bash tools/lab/radiance/flashnext-parity.sh compare --out DIR --baseline FILE
#
# measure: N launches (default 2) of launch-flashnext-best.sh with production defaults (layout flashnext-4gpu, 262144
#   context, 8 sessions; the live expert-placement policy is on when $BC_WORK_ROOT/live-policy-calibration.toml exists),
#   --record-period 1. Each launch warms up (only with the policy on: (min_samples + 1) heat windows plus a second, from the
#   manifest the launch wrote), then measures cold prefill, decode step time at 1 and 6 streams, 8 sessions and a 256K
#   context, stops the launcher (always, also on failure) and keeps the helper's `policy placement` line. Raw records go
#   to OUT/launch-<i>/, the summary to OUT/parity.json (flashnext_parity.py summarise). It uses all four cards: hold the
#   gpu lease (agent-lease acquire gpu ... -- bash tools/lab/radiance/flashnext-parity.sh measure ...). It runs in the
#   foreground for about an hour a launch; run it detached (setsid nohup ... > OUT.log 2>&1 &).
# baseline: docs/reference/engines/radiance/baselines/flashnext-4gpu-parity.json (or --to) from a measured OUT, each
#   gated metric's tolerance derived from the spread the run measured; refuses a single launch.
# compare: the table against a baseline; exit 1 on a gated regression beyond its allowance.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
REPO=$(pwd)
LAUNCHER=tools/lab/radiance/launch-flashnext-best.sh
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
BUILD="" OUT="" LAUNCHES=2 PORT=18450 BASELINE="" NAME="" TO=""
while [ $# -gt 0 ]; do
  case "$1" in
    --build) [ $# -ge 2 ] || usage; BUILD=$2; shift 2 ;;
    --out) [ $# -ge 2 ] || usage; OUT=$2; shift 2 ;;
    --launches) [ $# -ge 2 ] || usage; LAUNCHES=$2; shift 2 ;;
    --port) [ $# -ge 2 ] || usage; PORT=$2; shift 2 ;;
    --baseline) [ $# -ge 2 ] || usage; BASELINE=$2; shift 2 ;;
    --name) [ $# -ge 2 ] || usage; NAME=$2; shift 2 ;;
    --to) [ $# -ge 2 ] || usage; TO=$2; shift 2 ;;
    *) echo "flashnext-parity: unknown argument: $1" >&2; usage ;;
  esac
done

case "$CMD" in
  baseline)
    [ -n "$OUT" ] && [ -n "$NAME" ] || usage
    [ -f "$OUT/parity.json" ] || python3 "$PARITY" summarise "$OUT" || exit
    exec python3 "$PARITY" baseline "$OUT" --name "$NAME" ${TO:+--to "$TO"} ;;
  compare)
    [ -n "$OUT" ] && [ -n "$BASELINE" ] || usage
    [ -f "$OUT/parity.json" ] || python3 "$PARITY" summarise "$OUT" || exit
    exec python3 "$PARITY" compare "$OUT" --baseline "$BASELINE" ;;
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
BUILD=$(cd "$BUILD" && pwd)
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

LONG_COUNT=$(wc -w <<< "$LONG_SEEDS")
python3 "$PARITY" run-info "$OUT" --build "$BUILD" --launches "$LAUNCHES" --port "$PORT" \
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
stop_launcher() {
  [ "$LAUNCHED" = 1 ] || return 0
  LAUNCHED=0
  bash "$LAUNCHER" stop >> "$L/launcher.log" 2>&1 || echo "flashnext-parity: launcher stop failed (see $L/launcher.log)" >&2
}
trap 'stop_launcher' EXIT
trap 'stop_launcher; exit 130' INT TERM HUP

fail() { echo "flashnext-parity: launch $i: $*" >&2; stop_launcher; exit 1; }

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
  echo "flashnext-parity: launch $i/$LAUNCHES -> $L" >&2
  LAUNCHED=1   # a start that fails half way is stopped as well
  bash "$LAUNCHER" start --build "$BUILD" --port "$PORT" --record-period "$RECORD_PERIOD" > "$L/launcher.log" 2>&1 \
    || fail "launcher start failed (see $L/launcher.log)"
  # the start's stamp names its logs and manifest: engine-STAMP.log, helper0-STAMP.log, manifest-STAMP-PID.json
  ENGINE_LOG=$(sed -n 's/^ *engine .* log \(.*engine-[0-9-]*\.log\)$/\1/p' "$L/launcher.log" | tail -1)
  [ -n "$ENGINE_LOG" ] || fail "the launcher printed no engine log path"
  STAMP=$(basename "$ENGINE_LOG" .log); STAMP=${STAMP#engine-}
  MANIFEST=$(ls -t "$RUN"/manifest-"$STAMP"-*.json 2>/dev/null | head -1)
  [ -n "$MANIFEST" ] || fail "no manifest $RUN/manifest-$STAMP-*.json from this start"
  cp "$MANIFEST" "$L/manifest.json"
  python3 "$PARITY" warmup-plan "$L/manifest.json" --out "$L/launch.json" >&2 || fail "cannot read the manifest's live policy"
  WARMUP_S=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["warmup_s"])' "$L/launch.json")

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
  echo "flashnext-parity: launch $i warmup ${WARMUP_S}s, $k rounds" >&2

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

  stop_launcher
  cp "$ENGINE_LOG" "$L/engine.log" 2>/dev/null
  if cp "$RUN/helper0-$STAMP.log" "$L/helper0.log" 2>/dev/null; then
    grep 'policy placement' "$L/helper0.log" | tail -1 > "$L/policy-placement.txt"
  fi
  python3 "$PARITY" policy-line "$L/helper0.log" --out "$L/policy.json" >&2
done
trap - EXIT INT TERM HUP

python3 "$PARITY" summarise "$OUT" || exit
echo "flashnext-parity: $OUT/parity.json" >&2
