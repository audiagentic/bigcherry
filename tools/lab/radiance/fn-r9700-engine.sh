#!/bin/bash
# RAD02 first proof: the R9700 runs Flash-Next alone (tp 1, libr4d) and one XTX is the expert helper, computing the
# experts it owns at full width (R3_MOE_HELPER_ENGINE_RANKS=1). HELPER=0 runs the same R9700 engine without the
# helper: the baseline for this layout.
# Usage: B=<lane build dir> MODEL=<.rad> tools/lab/radiance/fn-r9700-engine.sh <out-dir>
#   ENGINE_DEV (2, the R9700), HELPER_DEV (0, an XTX), HELPER_MIB (20480), HOST_POOL_MIB (45056), N (256),
#   PROMPT (text file), EXTRA_ARGS
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
out=${1:?out dir}
mkdir -p "$out"
model=${MODEL:?set MODEL to the Flash-Next .rad container}
B=${B:?set B to the lane build directory}
rad=$B/bin/radiance
home=$B/radiance_home
helper=$home/helpers/r3_moe_helper
recipes=$repo/engines/radiance/recipes.toml
port=${PORT:-18437}

[ -x "$rad" ] || { echo "NO_RADIANCE $rad" >&2; exit 1; }
[ -x "$helper" ] || { echo "NO_R3_MOE_HELPER $helper" >&2; exit 1; }
[ -s "$model" ] || { echo "NO_MODEL $model" >&2; exit 1; }

# The flashnext-2xtx profile at one rank: what the R9700 cannot keep in VRAM sits in a larger host pool (each XTX
# rank holds ~31 GiB of experts between VRAM and its pool; one rank needs all of them).
mapfile -t engine_args < <(python3 - "$recipes" "${HOST_POOL_MIB:-45056}" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    a = tomllib.load(f)["runtime-profile"]["flashnext-2xtx"]["server-args"]
out, i = [], 0
while i < len(a):
    if a[i] in ("--tp", "--tp-wire", "--host-pool-mib"):
        i += 2
        continue
    out.append(a[i]); i += 1
print("\n".join(out + ["--tp", "1", "--host-pool-mib", sys.argv[2]]))
PY
)
ctx=$(python3 - "$recipes" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    print(tomllib.load(f)["runtime-profile"]["flashnext-2xtx"]["production-context"])
PY
)

helper_pid=
server_pid=
cleanup() {
    set +e
    if [ -n "${server_pid:-}" ] && kill -0 "$server_pid" 2>/dev/null; then kill -INT "$server_pid"; wait "$server_pid"; fi
    if [ -n "${helper_pid:-}" ] && kill -0 "$helper_pid" 2>/dev/null; then kill -TERM "$helper_pid"; wait "$helper_pid"; fi
}
trap cleanup EXIT INT TERM

pgrep -x r3_moe_helper >/dev/null && { pkill -TERM -x r3_moe_helper; sleep 3; pkill -KILL -x r3_moe_helper; }
rm -f /dev/shm/r3-moe-helper
if [ "${HELPER:-1}" != 0 ]; then
    echo "== helper: HIP device ${HELPER_DEV:-0} (XTX), full expert width"
    HIP_VISIBLE_DEVICES=${HELPER_DEV:-0} R3_MOE_HELPER_ENGINE_RANKS=1 R3_MOE_HELPER_MIB=${HELPER_MIB:-20480} \
        RADIANCE_HOME="$home" "$helper" "$model" >"$out/helper.log" 2>&1 &
    helper_pid=$!
    deadline=$((SECONDS + 900))
    until grep -qE '^r3_moe_helper: (ready |waiting for engine runtime expert maps)' "$out/helper.log"; do
        kill -0 "$helper_pid" 2>/dev/null || { echo "HELPER_EXITED" >&2; tail -40 "$out/helper.log" >&2; exit 1; }
        [ "$SECONDS" -lt "$deadline" ] || { echo "HELPER_READY_TIMEOUT" >&2; exit 1; }
        sleep 1
    done
    grep -E "^r3_moe_helper: (owns|selected|ready|waiting)" "$out/helper.log"
fi

echo "== radiance: HIP device ${ENGINE_DEV:-2} (R9700), tp 1: ${engine_args[*]}"
HIP_VISIBLE_DEVICES=${ENGINE_DEV:-2} R3_MOE_HELPER=${HELPER:-1} RADIANCE_HOME="$home" \
    "$rad" --model "$model" --radiance-home "$home" --kernels libr4d,libr3moe,libref \
    "${engine_args[@]}" ${EXTRA_ARGS:-} --max-model-len "$ctx" --host 127.0.0.1 --port "$port" \
    >"$out/server.log" 2>&1 &
server_pid=$!
deadline=$((SECONDS + ${LOAD_TIMEOUT:-1500}))
until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    kill -0 "$server_pid" 2>/dev/null || { echo "SERVER_EXITED_BEFORE_HEALTH" >&2; tail -60 "$out/server.log" >&2; exit 1; }
    if [ -n "$helper_pid" ] && ! kill -0 "$helper_pid" 2>/dev/null; then
        echo "HELPER_EXITED_DURING_LOAD" >&2; tail -60 "$out/helper.log" >&2; exit 1
    fi
    [ "$SECONDS" -lt "$deadline" ] || { echo "SERVER_HEALTH_TIMEOUT" >&2; tail -60 "$out/server.log" >&2; exit 1; }
    sleep 1
done

echo "== prefill/decode"
python3 "$here/fn-bench-once.py" "http://127.0.0.1:$port" "${PROMPT:-/mnt/data/bigcherry-work/corpus/kld-docs.txt}" \
    "${N:-256}" "$out"

[ -z "$helper_pid" ] || kill -0 "$helper_pid" 2>/dev/null || { echo "HELPER_DIED" >&2; tail -60 "$out/helper.log" >&2; exit 1; }
kill -0 "$server_pid" 2>/dev/null || { echo "SERVER_DIED" >&2; tail -60 "$out/server.log" >&2; exit 1; }
grep -E 'MoE remote hook active|R3_MOE_HELPER|error|abort|assert|fault' "$out/server.log" | tail -20 || true

kill -INT "$server_pid"
wait "$server_pid" || { echo "SERVER_STOP_FAILED" >&2; tail -60 "$out/server.log" >&2; exit 1; }
server_pid=
if [ -n "$helper_pid" ]; then
    kill -TERM "$helper_pid"
    wait "$helper_pid" || { echo "HELPER_STOP_FAILED" >&2; tail -60 "$out/helper.log" >&2; exit 1; }
    helper_pid=
    grep -E "served|large-lane" "$out/helper.log" || true
fi
trap - EXIT INT TERM
echo "FN_R9700_ENGINE_DONE"
