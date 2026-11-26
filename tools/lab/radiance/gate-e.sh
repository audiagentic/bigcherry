#!/bin/bash
# Gate E for a published radiance release: install it into an isolated root, verify the root, then start, canary and
# stop two production layouts (flashnext-4gpu and qwen27b-tp3), one short GPU lease per start.
#
# Usage:
#   bash tools/lab/radiance/gate-e.sh TAG [--wait [--wait-timeout MIN]] [--result FILE] [--keep]
#
#   TAG              the release tag, e.g. bc-radiance-2020.11.2 (or the bare version 2020.11.2)
#   --wait           poll the GitHub release until its three assets exist (archive, .sha256, .receipt.json); safe to run
#                    detached before the release is published. Without it, missing assets are an error.
#   --wait-timeout   minutes to wait for the assets (default 240; only with --wait)
#   --result FILE    also append the one-line result to FILE (the only thing kept; keep it next to the evidence)
#   --keep           keep the temp directory and install root (default: both are deleted at the end, pass or fail)
#
# Steps: wait for assets; download to a temp directory; check the archive's sha256; `bc-release install` into a fresh
# root with the ROCm SDK; `bc-release verify`; then per layout, under `agent-lease acquire gpu`, a native
# `bc-serve serve` with a fresh work root (so the start includes planning), seconds until state=running, a
# temperature-0 canary ("The capital of France is", eight tokens, the text must contain "Paris"), and a clean stop (stop
# and serve exit codes 0).
#
# Output, one line to stdout (and to --result):
#   TAG | archive id (first 8) | flashnext-4gpu to running (s) | qwen27b-tp3 to running (s) | PASS or FAIL: reasons
# the same columns as the Gate E evidence table. Exit 0 only if every check passed.
#
# Needs: gh (authenticated), agent-lease, curl, the ROCm SDK at $ROCM_PATH (default /opt/rocm-7.2.4) and the model
# store ($BC_MODEL_ROOT) with the two layouts' models. It deletes only the temp directory it created.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1

REPO=${GATE_E_REPO:-audiagentic/bigcherry}
LAYOUTS=(flashnext-4gpu qwen27b-tp3)
TAG="" WAIT=0 WAIT_MIN=240 RESULT="" KEEP=0

usage() {
  sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --wait) WAIT=1; shift ;;
    --wait-timeout) [ $# -ge 2 ] || usage; WAIT_MIN=$2; shift 2 ;;
    --result) [ $# -ge 2 ] || usage; RESULT=$2; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) usage ;;
    -*) echo "gate-e: unknown option: $1" >&2; usage ;;
    *) [ -z "$TAG" ] || usage; TAG=$1; shift ;;
  esac
done
[ -n "$TAG" ] || usage
case "$TAG" in bc-radiance-*) ;; *) TAG=bc-radiance-$TAG ;; esac
[[ "$WAIT_MIN" =~ ^[1-9][0-9]*$ ]] || { echo "gate-e: --wait-timeout must be a positive integer (minutes)" >&2; exit 2; }

source tools/env/bigcherry-env.sh >/dev/null 2>&1
: "${BC_WORK_ROOT:?gate-e: BC_WORK_ROOT is not set}"
: "${BC_MODEL_ROOT:?gate-e: BC_MODEL_ROOT is not set}"
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
export HIP_PATH=$ROCM_PATH
export LD_LIBRARY_PATH=/opt/ffmpeg6/lib:$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=$ROCM_PATH/bin:$PATH

TMP=$(mktemp -d "${GATE_E_TMP:-$BC_WORK_ROOT}/gate-e.XXXXXX") || { echo "gate-e: cannot make a temp directory" >&2; exit 1; }
cleanup() {
  if [ "$KEEP" = 1 ]; then echo "gate-e: kept $TMP" >&2; else rm -rf -- "${TMP:?}"; fi
}
trap cleanup EXIT

FAILS=()
fail() { FAILS+=("$1"); echo "gate-e: FAIL: $1" >&2; }

# ---- 1. the three assets -------------------------------------------------------------------------------------------
assets() { gh release view "$TAG" --repo "$REPO" --json assets --jq '.assets[].name' 2>/dev/null; }
have_all() {
  local a; a=$(assets)
  grep -q '\.tar\.gz$' <<<"$a" && grep -q '\.tar\.gz\.sha256$' <<<"$a" && grep -q '\.receipt\.json$' <<<"$a"
}
deadline=$(( $(date +%s) + WAIT_MIN * 60 ))
until have_all; do
  if [ "$WAIT" != 1 ]; then echo "gate-e: $TAG does not have its three assets yet (use --wait)" >&2; exit 1; fi
  [ "$(date +%s)" -lt "$deadline" ] || { echo "gate-e: gave up waiting for $TAG's assets" >&2; exit 1; }
  sleep 60
done

# ---- 2. download, checksum, install, verify ------------------------------------------------------------------------
DL=$TMP/dl X=$TMP/x ROOT=$TMP/root
mkdir -p "$DL" "$X"
(cd "$DL" && gh release download "$TAG" --repo "$REPO" --clobber >/dev/null 2>&1) || { echo "gate-e: download failed" >&2; exit 1; }
ARC=$(ls "$DL"/*.tar.gz 2>/dev/null | head -1)
REC=$(ls "$DL"/*.receipt.json 2>/dev/null | head -1)
[ -n "$ARC" ] && [ -n "$REC" ] || { echo "gate-e: the release's archive or receipt is missing after download" >&2; exit 1; }
ID=$(basename "$ARC" .tar.gz); ID=${ID##*-}; ID=${ID:0:8}
(cd "$DL" && sha256sum -c ./*.sha256 >/dev/null 2>&1) || fail "archive sha256 mismatch"
tar -xzf "$ARC" -C "$X" || { echo "gate-e: cannot extract the archive" >&2; exit 1; }
BR=$(find "$X" -name bc-release -type f | head -1)
[ -n "$BR" ] || { echo "gate-e: no bc-release in the archive" >&2; exit 1; }

install_out=$("$BR" install --tag "$TAG" --record "$REC" --archive "$ARC" --root "$ROOT" --sdk "$ROCM_PATH" 2>&1) \
  || fail "bc-release install exited nonzero"
grep -q '"status": *"verified"' <<<"$install_out" || fail "install did not report verified"
verify_out=$("$BR" verify --root "$ROOT" --sdk "$ROCM_PATH" 2>&1) || fail "bc-release verify exited nonzero"
grep -q '"status": *"verified"' <<<"$verify_out" || fail "verify did not report verified"

# ---- 3. one start per layout, one short lease each -----------------------------------------------------------------
BS=$ROOT/releases/$TAG/radiance_home/bin/bc-serve
[ -x "$BS" ] || { echo "gate-e: the installed root has no bc-serve" >&2; fail "no bc-serve in the installed root"; }

cat >"$TMP/start-one.sh" <<'EOF'
#!/bin/bash
# start-one.sh LAYOUT OUTDIR: start, wait for running, canary, stop; writes OUTDIR/result (key=value lines)
cfg=$1; o=$2; mkdir -p "$o"
t0=$(date +%s); state=down
"$BS" serve "$cfg" --model-root "$BC_MODEL_ROOT" --run-dir "$o/run" --work-root "$o/wr" \
  >"$o/serve.log" 2>&1 &
sp=$!
for _ in $(seq 1 300); do
  kill -0 "$sp" 2>/dev/null || break
  "$BS" status --run-dir "$o/run" --json >"$o/status.json" 2>/dev/null
  if grep -q '"state": *"running"' "$o/status.json" 2>/dev/null; then state=running; break; fi
  sleep 2
done
t1=$(date +%s)
port=$(grep -o '"port": *[0-9]*' "$o/status.json" 2>/dev/null | head -1 | grep -o '[0-9]*$')
canary=""
if [ "$state" = running ] && [ -n "$port" ]; then
  model=$(curl -s -m 30 "http://127.0.0.1:$port/v1/models" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
  canary=$(curl -s -m 180 "http://127.0.0.1:$port/v1/completions" -H 'content-type: application/json' \
    -d '{"model":"'"$model"'","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
    | grep -o '"text":"[^"]*"' | head -1)
fi
"$BS" stop --run-dir "$o/run" >"$o/stop.log" 2>&1; stop_rc=$?
wait "$sp"; serve_rc=$?
printf 'state=%s\nseconds=%s\ncanary=%s\nstop=%s\nserve=%s\n' "$state" "$((t1 - t0))" "$canary" "$stop_rc" "$serve_rc" >"$o/result"
EOF
chmod +x "$TMP/start-one.sh"

declare -A SECS
if [ -x "$BS" ]; then
  export BS BC_MODEL_ROOT
  for cfg in "${LAYOUTS[@]}"; do
    o=$TMP/out-$cfg
    agent-lease acquire gpu --ttl 15m --est 10m --wait --wait-timeout 4h \
      --purpose "Gate E $TAG start $cfg" -- bash "$TMP/start-one.sh" "$cfg" "$o" >"$TMP/lease-$cfg.log" 2>&1 \
      || fail "$cfg: the leased start did not run (see lease output)"
    r=$o/result
    if [ ! -f "$r" ]; then SECS[$cfg]="-"; continue; fi
    get() { sed -n "s/^$1=//p" "$r" | head -1; }
    SECS[$cfg]=$(get seconds)
    [ "$(get state)" = running ] || fail "$cfg: never reached running"
    grep -q 'Paris' <<<"$(get canary)" || fail "$cfg: the canary text lacks Paris ($(get canary))"
    [ "$(get stop)" = 0 ] || fail "$cfg: stop exit code $(get stop)"
    [ "$(get serve)" = 0 ] || fail "$cfg: serve exit code $(get serve)"
  done
fi

# ---- 4. the result line ----------------------------------------------------------------------------------------------
if [ ${#FAILS[@]} -eq 0 ]; then verdict=PASS; else verdict="FAIL: $(IFS='; '; echo "${FAILS[*]}")"; fi
line="${TAG#bc-radiance-} | $ID | ${SECS[flashnext-4gpu]:--} s | ${SECS[qwen27b-tp3]:--} s | $verdict"
echo "$line"
[ -z "$RESULT" ] || echo "$line" >>"$RESULT"
[ ${#FAILS[@]} -eq 0 ]
