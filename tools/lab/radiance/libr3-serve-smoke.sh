#!/bin/bash
# RR01: does radiance serve a model on one RX 7900 XTX through libr3? Starts the engine with libr3 as the only device
# kernel library, waits for /health, sends one greedy completion and one chat request, prints the text and the
# speed the server reports, and stops the server with SIGINT. Speculation is off, so the reply is the target's own.
# libr3 must have been built first (libr3-build.sh).
# Run as a queue SCRIPT job (one card):
#   VIS=0 SCRIPT r3-serve tools/lab/radiance/libr3-serve-smoke.sh @<any build run> <out-dir>
# The first argument (a llama-server path from the queue) is ignored.
# Usage: libr3-serve-smoke.sh <ignored> <out-dir>
# env: MODEL (.rad container), GPU (HIP index, 0), TARGET (gfx1100), PORT (18431), CTX (8192), KV (bf16), SPEC (0),
#      N (64 tokens), LOAD_TIMEOUT (900 s), RADIANCE_SRC, WORK
set -u
out=$2
mkdir -p "$out"
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
work=${WORK:-/mnt/data/bigcherry-work/engines/radiance-rdna3}
target=${TARGET:-gfx1100}
model=${MODEL:-/mnt/data/llm-models/radiance/StillDeadcode/qwen3.8-27b-mxfp4/qwen3.8-27b-mxfp4.rad}
port=${PORT:-18431}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm-7.2.4}
export PATH="$ROCM_PATH/bin:$PATH"
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
r3so=$(find "$work/libr3-$target-build" -name 'libr3.so' 2> /dev/null | head -1)
[ -n "$r3so" ] || { echo "NO_LIBR3: run libr3-build.sh first"; exit 1; }
home="$(dirname "$(dirname "$r3so")"):$src/build/radiance_home"
echo "radiance $(git -C "$src" rev-parse --short HEAD); model $(basename "$model") ($(du -h "$model" | cut -f1)); HIP device ${GPU:-0}; kernels libr3"

HIP_VISIBLE_DEVICES=${GPU:-0} "$src/build/bin/radiance" --model "$model" --radiance-home "$home" --kernels libr3,libref \
    --tp 1 --max-model-len "${CTX:-8192}" --max-num-seqs 1 --kv-cache-dtype "${KV:-bf16}" \
    --num-speculative-tokens "${SPEC:-0}" --host 127.0.0.1 --port "$port" > "$out/server.log" 2>&1 &
pid=$!
t0=$(date +%s)
until curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1; do
    if ! kill -0 $pid 2> /dev/null; then
        echo "SERVER_EXITED before /health after $(( $(date +%s) - t0 )) s"
        grep -vE "^\s*$" "$out/server.log" | tail -25 | cut -c1-240
        exit 1
    fi
    if [ $(( $(date +%s) - t0 )) -gt "${LOAD_TIMEOUT:-900}" ]; then
        echo "LOAD_TIMEOUT"; tail -15 "$out/server.log" | cut -c1-240; kill -INT $pid; wait $pid; exit 1
    fi
    sleep 2
done
echo "healthy after $(( $(date +%s) - t0 )) s"
grep -iE "kernel|libr3|placement|vram|refus|fallback|host kernel" "$out/server.log" | head -20 | cut -c1-220

ask() {  # <name> <path> <json>
    local t1; t1=$(date +%s.%N)
    curl -s --max-time "${REQ_TIMEOUT:-600}" "http://127.0.0.1:$port$2" -H 'Content-Type: application/json' -d "$3" > "$out/$1.json"
    echo "-- $1: curl exit $?, $(echo "$(date +%s.%N) - $t1" | bc | cut -c1-6) s"
    python3 - "$out/$1.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception as e:
    print("  not JSON:", open(sys.argv[1], errors="replace").read()[:300]); sys.exit(0)
c = (d.get("choices") or [{}])[0]
text = c.get("text") or (c.get("message") or {}).get("content") or ""
print("  text:", repr(text[:400]))
print("  usage:", d.get("usage"), "finish:", c.get("finish_reason"), "error:", d.get("error"))
PY
}
n=${N:-64}
ask completion /v1/completions "{\"prompt\": \"The capital of France is\", \"max_tokens\": $n, \"temperature\": 0}"
ask chat /v1/chat/completions "{\"messages\": [{\"role\": \"user\", \"content\": \"Reply with the numbers one to ten, separated by commas, and nothing else.\"}], \"max_tokens\": $n, \"temperature\": 0, \"reasoning_effort\": \"none\"}"
grep -iE "tok/s|t/s|tokens per|prefill|decode" "$out/server.log" | tail -6 | cut -c1-220
grep -iE "error|abort|assert|nan|fault" "$out/server.log" | head -8 | cut -c1-220
kill -INT $pid; wait $pid; echo "server exit $?"
exit 0
