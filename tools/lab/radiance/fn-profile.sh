#!/bin/bash
# Kernel/copy trace of one prefill request under rocprofv3, taken from the NATIVE serving owner's own processes.
# The owner (`bc-serve serve NAME --native-config YAML`) admits and starts the engine and its helpers exactly as a
# normal start does; this tool attaches rocprofv3 to the engine and helper PIDs that the owner's session record
# (RUN/session.json) names, sends one request, detaches, and stops the session with `bc-serve stop`. It never
# matches or signals a process by name: only the owner is waited on and only `bc-serve stop` ends the session.
# Compare two traces with fn-profile-compare.py to split a slowdown into kernel time vs gaps between kernels.
#
# Usage: tools/lab/radiance/fn-profile.sh --build DIR --native-config YAML --name NAME --out DIR --port P --prompt-tokens N
#   --build DIR           a Radiance build or release tree (bin/radiance, radiance_home/bin/bc-serve)
#   --native-config YAML  the serving config to profile (the same one a normal start takes)
#   --name NAME           the config's serving name (bc-serve serve NAME)
#   --out DIR             trace and log folder (must be new or empty); the session run dir is OUT/run
#   --port P              the port the config serves on (the request goes there)
#   --prompt-tokens N     size of the one profiled prefill request
# Needs BC_WORK_ROOT and BC_MODEL_ROOT (tools/env/bigcherry-env.sh). Hold the gpu lease around it.
set -u
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)

usage() { sed -n '9,16p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
die() { echo "fn-profile: $*" >&2; exit "${2:-1}"; }

BUILD="" CONFIG="" NAME="" OUT="" PORT="" TOKENS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --build) [ $# -ge 2 ] || usage; BUILD=$2; shift 2 ;;
    --native-config) [ $# -ge 2 ] || usage; CONFIG=$2; shift 2 ;;
    --name) [ $# -ge 2 ] || usage; NAME=$2; shift 2 ;;
    --out) [ $# -ge 2 ] || usage; OUT=$2; shift 2 ;;
    --port) [ $# -ge 2 ] || usage; PORT=$2; shift 2 ;;
    --prompt-tokens) [ $# -ge 2 ] || usage; TOKENS=$2; shift 2 ;;
    *) echo "fn-profile: unknown argument: $1" >&2; usage ;;
  esac
done
[ -n "$BUILD" ] && [ -n "$CONFIG" ] && [ -n "$NAME" ] && [ -n "$OUT" ] && [ -n "$PORT" ] && [ -n "$TOKENS" ] || usage
[[ "$PORT" =~ ^[1-9][0-9]*$ ]] || die "--port must be a positive integer" 2
[[ "$TOKENS" =~ ^[1-9][0-9]*$ ]] || die "--prompt-tokens must be a positive integer" 2
[ -f "$CONFIG" ] || die "no native config $CONFIG" 2
[ -x "$BUILD/bin/radiance" ] || die "$BUILD has no bin/radiance" 2
BC_SERVE="$BUILD/radiance_home/bin/bc-serve"
[ -x "$BC_SERVE" ] || die "$BUILD has no native radiance_home/bin/bc-serve" 2
command -v rocprofv3 >/dev/null || die "rocprofv3 not found" 2
if [ -z "${BC_WORK_ROOT:-}" ] || [ -z "${BC_MODEL_ROOT:-}" ]; then
  source "$repo/tools/env/bigcherry-env.sh" >/dev/null 2>&1
fi
[ -n "${BC_WORK_ROOT:-}" ] || die "BC_WORK_ROOT not set (tools/env/bigcherry-env.sh)" 2
[ -n "${BC_MODEL_ROOT:-}" ] || die "BC_MODEL_ROOT not set" 2
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then die "$OUT is not empty" 2; fi
mkdir -p "$OUT" || die "cannot create $OUT" 2
OUT=$(cd "$OUT" && pwd)
RUN=$OUT/run
BUILD=$(cd "$BUILD" && pwd -P)

OWNER_PID=""
STARTED=0
PROF_PIDS=()
FDS=()
stop_session() {
  [ "$STARTED" = 1 ] || return 0
  local rc=0
  if "$BC_SERVE" stop --run-dir "$RUN" >>"$OUT/owner.log" 2>&1; then
    wait "$OWNER_PID" || { echo "fn-profile: owner exited with error (see $OUT/owner.log)" >&2; rc=1; }
  else
    echo "fn-profile: bc-serve stop failed (see $OUT/owner.log)" >&2; rc=1
  fi
  STARTED=0
  return "$rc"
}
trap '' PIPE   # a profiler that exited early must not end this script when its detach pipe is written
close_pipes() { local fd; for fd in "${FDS[@]}"; do exec {fd}>&-; done; FDS=(); }
trap 'close_pipes; stop_session' EXIT
trap 'close_pipes; stop_session; exit 130' INT TERM HUP
fail() { echo "fn-profile: $*" >&2; close_pipes; stop_session; exit 1; }

python3 "$here/arm_yaml.py" --base "$CONFIG" --out "$OUT/native-input.yaml" --set "server.port=$PORT" ||
  fail "cannot write the arm YAML"
"$BC_SERVE" serve "$NAME" --run-dir "$RUN" --model-root "$BC_MODEL_ROOT" --engine-binary "$BUILD/bin/radiance" \
  --work-root "$BC_WORK_ROOT" --native-config "$OUT/native-input.yaml" \
  >"$OUT/owner.log" 2>&1 &
OWNER_PID=$!
STARTED=1

# Ready: the owner reports state running. If the owner exits first it refused or failed.
until "$BC_SERVE" status --run-dir "$RUN" --json 2>/dev/null |
    python3 -c 'import json,sys
try:
    assert json.load(sys.stdin).get("state") == "running"
except (ValueError, AssertionError):
    sys.exit(1)' 2>/dev/null; do
  kill -0 "$OWNER_PID" 2>/dev/null || fail "native bc-serve refused or exited (see $OUT/owner.log)"
  sleep 1
done

# The processes the session record names for this owner: the engine and its helpers.
mapfile -t TARGETS < <(python3 - "$RUN/session.json" "$OWNER_PID" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    record = json.load(f)
if record["owner_pid"] != int(sys.argv[2]):
    sys.exit(1)
for p in record["processes"]:
    if p["role"] == "engine" or p["role"].startswith("helper"):
        print(p["role"], p["pid"])
PY
)
[ "${#TARGETS[@]}" -gt 0 ] || fail "the session record names no engine or helper process"

# One rocprofv3 per named process, attached by PID; each detaches when it reads a line from its own pipe.
mkdir -p "$OUT/trace"
for t in "${TARGETS[@]}"; do
  role=${t% *}; pid=${t##* }
  fifo="$OUT/detach.$role"
  mkfifo "$fifo" || fail "cannot create $fifo"
  rocprofv3 --attach "$pid" --kernel-trace --memory-copy-trace --output-format csv \
    -d "$OUT/trace/$role" -o run <"$fifo" >"$OUT/rocprof.$role.log" 2>&1 &
  PROF_PIDS+=($!)
  exec {fd}>"$fifo" || fail "cannot open $fifo"
  FDS+=("$fd")
done

REQ_RC=0
PYTHONPATH="$repo/tools" python3 -m bigcherry stream-bench --url "http://127.0.0.1:$PORT" --streams 1 \
  --prompt-tokens "$TOKENS" --max-tokens 1 --reps 1 --temperature 0 --output "$OUT/request.json" \
  >"$OUT/request.log" 2>&1 || REQ_RC=$?

for fd in "${FDS[@]}"; do echo >&"$fd"; done 2>/dev/null
close_pipes
PROF_RC=0
for i in "${!PROF_PIDS[@]}"; do
  wait "${PROF_PIDS[$i]}" || { echo "fn-profile: rocprofv3 for ${TARGETS[$i]% *} failed (see $OUT/rocprof.${TARGETS[$i]% *}.log)" >&2; PROF_RC=1; }
done
STOP_RC=0
stop_session || STOP_RC=1

[ "$REQ_RC" = 0 ] || die "profiled request failed (see $OUT/request.log)"
[ "$PROF_RC" = 0 ] || exit 1
[ "$STOP_RC" = 0 ] || exit 1
trap - EXIT INT TERM HUP
find "$OUT/trace" \( -name "*kernel_trace.csv" -o -name "*memory_copy_trace.csv" \) | sort
