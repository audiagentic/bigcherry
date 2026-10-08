#!/bin/bash
# Reference lane: the owner's existing `radiance-vllm` container on Brutus (vLLM + radiance + libr4d, Qwen3.8-27B MXFP4,
# DFlash2-FP8 drafter, one R9700), measured with the lab's own prompt corpus and depths. The container is started,
# measured and stopped again; nothing in its configuration is changed.
# Run it as a queue SCRIPT job so the GPU lock covers it:
#   VIS=0,1,2,3 SCRIPT ref-radiance tools/lab/reference-vllm/run-radiance.sh @<any build run> <out-dir> [depth...]
# The first argument (a llama-server path from the queue) is ignored.
# Usage: run-radiance.sh <ignored> <out-dir> [depth-tokens...]   env: CONTAINER (radiance-vllm), PORT (8081), DECODE (512)
set -u
out=$2; shift 2
mkdir -p "$out"
C=${CONTAINER:-radiance-vllm}
PORT=${PORT:-8081}
depths=("$@"); [ ${#depths[@]} -eq 0 ] && depths=(2048 8192 24576 98304)
was_running=$(docker inspect -f '{{.State.Running}}' "$C" 2> /dev/null)
[ -z "$was_running" ] && { echo "NO_CONTAINER $C"; exit 1; }
[ "$was_running" = true ] || docker start "$C" > /dev/null
ok=0
for _ in $(seq 360); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { ok=1; break; }
    [ "$(docker inspect -f '{{.State.Running}}' "$C")" = true ] || break
    sleep 5
done
if [ "$ok" != 1 ]; then
    echo "SERVER_FAILED"; docker logs --tail 20 "$C" 2>&1 | cut -c1-200
    [ "$was_running" = true ] || docker stop "$C" > /dev/null
    exit 1
fi
model=$(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
echo "model $model; image $(docker inspect -f '{{.Config.Image}}' "$C")"
rocm-smi --showmeminfo vram 2> /dev/null | grep "Total Used" > "$out/vram.txt"
python3 "$(cd "$(dirname "$0")" && pwd)/bench-openai.py" "http://127.0.0.1:$PORT" "$model" "$out/bench.json" "${depths[@]}" \
    --decode "${DECODE:-512}" --reps 2
curl -s "http://127.0.0.1:$PORT/metrics" 2> /dev/null | grep -E "^vllm:spec_decode.*(accepted|draft)_tokens" | cut -c1-160
docker logs --tail 400 "$C" > "$out/container.log" 2>&1
[ "$was_running" = true ] || docker stop "$C" > /dev/null
exit 0
