#!/bin/bash
# MEN02: serve a model from the source-built radiance on the R9700 alone (--tp 1; its kernel library covers no other
# card here) and measure it with the lab client, corpus, depths and repeats used for the vLLM container and for
# llama.cpp (tools/lab/reference-vllm/). One arm per drafter setting: spec = the container's own drafter operating
# point (--num-speculative-tokens auto), none = drafter off.
# Run as a queue SCRIPT job so the GPU lock covers it:
#   VIS=0,1,2,3 SCRIPT radiance-run tools/lab/radiance/run-radiance.sh @<any build run> <out-dir> [depth...]
# The first argument (a llama-server path from the queue) is ignored.
# Usage: run-radiance.sh <ignored> <out-dir> [depth-tokens...]
# env: RADIANCE_SRC (/mnt/data/bigcherry-work/engines/radiance), MODEL (.rad file, checkpoint dir or HF repo id;
#      default StillDeadcode/qwen3.8-27b-mxfp4, fetched once into DOWNLOAD_DIR), DOWNLOAD_DIR
#      (/mnt/data/llm-models/radiance), GPU (HIP index of the R9700, 2), CTX (180000), ARMS ("spec none"),
#      EXTRA (further serve flags), DECODE (512), PORT (18745)
set -u
out=$2; shift 2
mkdir -p "$out"
depths=("$@"); [ ${#depths[@]} -eq 0 ] && depths=(2048 8192 24576 98304)
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
MODEL=${MODEL:-StillDeadcode/qwen3.8-27b-mxfp4}
PORT=${PORT:-18745}
here=$(cd "$(dirname "$0")" && pwd)
bench=$here/bench-openai.py; [ -f "$bench" ] || bench=$here/../reference-vllm/bench-openai.py
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
export HIP_VISIBLE_DEVICES=${GPU:-2} RADIANCE_HOME=$src/build/radiance_home
docker stop radiance-vllm > /dev/null 2>&1
echo "radiance $(git -C "$src" rev-parse --short HEAD); model $MODEL"

run_arm() {  # <arm>
    local arm=$1 spec pid ok=0
    case "$arm" in
        spec) spec=auto ;;
        none) spec=0 ;;
        *) echo "$arm: unknown arm"; return 2 ;;
    esac
    # shellcheck disable=SC2086
    "$src/build/bin/radiance" --model "$MODEL" --download-dir "${DOWNLOAD_DIR:-/mnt/data/llm-models/radiance}" \
        --tp 1 --max-num-seqs 8 --max-model-len "${CTX:-180000}" --kv-cache-dtype fp8 \
        --num-speculative-tokens "$spec" --host 127.0.0.1 --port "$PORT" ${EXTRA:-} > "$out/$arm.server.log" 2>&1 &
    pid=$!
    for _ in $(seq 2400); do   # the first start may download the container
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { ok=1; break; }
        kill -0 "$pid" 2> /dev/null || break
        sleep 1
    done
    if [ "$ok" != 1 ]; then
        echo "$arm: SERVER_FAILED $(grep -vE "^\s*$" "$out/$arm.server.log" | tail -4 | cut -c1-220 | tr '\n' '|')"
        kill -9 "$pid" 2> /dev/null
        return 1
    fi
    echo "== $arm: $(grep -m3 -iE "device|kernels|elastic|context" "$out/$arm.server.log" | cut -c1-160 | tr '\n' '|')"
    rocm-smi --showmeminfo vram 2> /dev/null | grep "Total Used" > "$out/$arm.vram.txt"
    local model
    model=$(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
    python3 "$bench" "http://127.0.0.1:$PORT" "$model" "$out/$arm.bench.json" "${depths[@]}" \
        --decode "${DECODE:-512}" --reps 2 | sed "s/^/$arm /"
    curl -s "http://127.0.0.1:$PORT/metrics" 2> /dev/null | grep -iE "accept|draft|spec" | grep -v "^#" | head -8 | cut -c1-160 | sed "s/^/$arm /"
    kill -INT "$pid" 2> /dev/null
    for _ in $(seq 60); do kill -0 "$pid" 2> /dev/null || break; sleep 1; done
    kill -9 "$pid" 2> /dev/null
    grep -iE "accept" "$out/$arm.server.log" | tail -2 | cut -c1-200 | sed "s/^/$arm /"
    sleep 5
    return 0
}

for arm in ${ARMS:-spec none}; do run_arm "$arm"; done
exit 0
