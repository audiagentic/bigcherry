#!/bin/bash
# Flash-Next helper smoke/perf: R9700 owns cold experts, two XTX ranks serve the model.
# Usage: MODEL=/path/to/model.rad tools/lab/radiance/fn-helper.sh [out-dir]
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
out=${1:-/tmp/r3-fn-helper}
mkdir -p "$out"
model=${MODEL:?set MODEL to the Flash-Next .rad container}
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
rad=${RAD_BIN:-$src/build/bin/radiance}
home=${RADIANCE_HOME:-$src/build/radiance_home}
runtime=${RADIANCE_RUNTIME_HOME:-${home%%:*}}
helper=${R3_MOE_WORKER_BIN:-$runtime/helpers/r3_moe_worker}
recipes=$repo/engines/radiance/recipes.toml
port=${PORT:-18437}

[ -x "$rad" ] || { echo "NO_RADIANCE $rad" >&2; exit 1; }
[ -x "$helper" ] || { echo "NO_R3_MOE_WORKER $helper" >&2; exit 1; }
[ -s "$model" ] || { echo "NO_MODEL $model" >&2; exit 1; }
[ -r "$recipes" ] || { echo "NO_RECIPES $recipes" >&2; exit 1; }

mapfile -t profile_args < <(python3 - "$recipes" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    p = tomllib.load(f)["runtime-profile"]["flashnext-2xtx"]
for x in p["server-args"]:
    print(x)
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
    if [ -n "${server_pid:-}" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill -INT "$server_pid"; wait "$server_pid"
    fi
    if [ -n "${helper_pid:-}" ] && kill -0 "$helper_pid" 2>/dev/null; then
        kill -TERM "$helper_pid"; wait "$helper_pid"
    fi
}
trap cleanup EXIT INT TERM

# A helper left by an interrupted run keeps its 28 GiB on the R9700 and the next one runs out of memory.
pgrep -x r3_moe_worker >/dev/null && { pkill -TERM -x r3_moe_worker; sleep 3; pkill -KILL -x r3_moe_worker; }
rm -f /dev/shm/r3-moe-helper
if [ "${HELPER:-1}" != 0 ]; then
echo "== helper: HIP device 2"
HIP_VISIBLE_DEVICES=2 RADIANCE_HOME="$home" \
    "$helper" "$model" >"$out/helper.log" 2>&1 &
helper_pid=$!
deadline=$((SECONDS + ${HELPER_TIMEOUT:-900}))
until grep -qE '^r3_moe_worker: (ready |waiting for engine runtime expert maps)' "$out/helper.log"; do
    if ! kill -0 "$helper_pid" 2>/dev/null; then
        echo "HELPER_EXITED" >&2; tail -40 "$out/helper.log" >&2; exit 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "HELPER_READY_TIMEOUT" >&2; tail -40 "$out/helper.log" >&2; exit 1
    fi
    sleep 1
done
grep -E "^r3_moe_worker: (owns|selected|ready|waiting)" "$out/helper.log"
fi

echo "== radiance: HIP devices 0,1; runtime-profile flashnext-2xtx"
HIP_VISIBLE_DEVICES=0,1 R3_MOE_WORKER=${HOOK:-${HELPER:-1}} RADIANCE_HOME="$home" \
    "$rad" --model "$model" --radiance-home "$home" --kernels libr3,libr3moe,libref \
    "${profile_args[@]}" ${EXTRA_ARGS:-} --max-model-len "$ctx" --host 127.0.0.1 --port "$port" \
    >"$out/server.log" 2>&1 &
server_pid=$!
deadline=$((SECONDS + ${LOAD_TIMEOUT:-900}))
until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "SERVER_EXITED_BEFORE_HEALTH" >&2; tail -60 "$out/server.log" >&2; exit 1
    fi
    if [ -n "$helper_pid" ] && ! kill -0 "$helper_pid" 2>/dev/null; then
        echo "HELPER_EXITED_DURING_LOAD" >&2; tail -60 "$out/helper.log" >&2; exit 1
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "SERVER_HEALTH_TIMEOUT" >&2; tail -60 "$out/server.log" >&2; exit 1
    fi
    sleep 1
done

echo "== prefill/decode"
python3 - "http://127.0.0.1:$port" "${CORPUS:-/mnt/data/bigcherry-work/corpus/kld-docs.txt}" "${N:-256}" "$out" <<'PY'
import json, sys, time, urllib.request
base, corpus_path, n, out_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
try:
    text = open(corpus_path, encoding="utf-8", errors="replace").read()[:24000]
except OSError:
    text = ("Radiance routes mixture-of-experts tokens across accelerators. "
            "This paragraph is repeated to create a deterministic prefill workload. ") * 180
body = {"prompt": text, "max_tokens": n, "temperature": 0, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True}}
req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t0 = time.monotonic()
first = last = None
usage = {}
pieces = 0
texts = []
with urllib.request.urlopen(req, timeout=1800) as resp:
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        d = json.loads(data)
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices") or []:
            s = ch.get("text") or ""
            if s:
                texts.append(s)
                now = time.monotonic()
                first = first or now
                last = now
                pieces += 1
# The greedy completion, so two arms can be checked for identical output.
open(f"{out_dir}/completion.txt", "w", encoding="utf-8").write("".join(texts))
prompt = int(usage.get("prompt_tokens") or 0)
gen = int(usage.get("completion_tokens") or 0)
if not first or prompt <= 0 or gen <= 1:
    raise SystemExit(f"BENCH_INCOMPLETE prompt={prompt} gen={gen} chunks={pieces}")
prefill = prompt / (first - t0)
span = max(last - first, 1e-9)
decode = (gen - 1) / span
print(f"prefill {prefill:.1f} tok/s ({prompt} tokens)")
print(f"decode {decode:.1f} tok/s ({gen} tokens)")
PY

[ -z "$helper_pid" ] || kill -0 "$helper_pid" 2>/dev/null || { echo "HELPER_DIED" >&2; tail -60 "$out/helper.log" >&2; exit 1; }
kill -0 "$server_pid" 2>/dev/null || { echo "SERVER_DIED" >&2; tail -60 "$out/server.log" >&2; exit 1; }

grep -E 'MoE remote hook active|R3_MOE_WORKER|error|abort|assert|fault' "$out/server.log" | tail -20 || true

kill -INT "$server_pid"
if ! wait "$server_pid"; then
    echo "SERVER_STOP_FAILED" >&2; tail -60 "$out/server.log" >&2; exit 1
fi
server_pid=
if [ -n "$helper_pid" ]; then
    kill -TERM "$helper_pid"
    if ! wait "$helper_pid"; then
        echo "HELPER_STOP_FAILED" >&2; tail -60 "$out/helper.log" >&2; exit 1
    fi
fi
helper_pid=
trap - EXIT INT TERM
echo "FN_HELPER_DONE"
