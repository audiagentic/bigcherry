#!/bin/bash
# MEN02: serve a model from the source-built radiance on the R9700 alone (--tp 1; its kernel library covers no other
# card here) and measure it with `bigcherry engine-bench`, the same command, corpus, depths and repeats used for
# llama.cpp (tools/lab/reference-vllm/run-llamacpp-r9700.sh). One arm per drafter setting: spec = the container's own
# drafter operating point (--num-speculative-tokens auto), none = drafter off.
# Each arm writes <out-dir>/<arm>.engine-bench.json and <arm>.server.log.
# Run as a queue SCRIPT job so the GPU lock covers it:
#   VIS=0,1,2,3 SCRIPT radiance-run tools/lab/radiance/run-radiance.sh @<any build run> <out-dir> [depth...]
# The first argument (a llama-server path from the queue) is ignored.
# Usage: run-radiance.sh <ignored> <out-dir> [depth-tokens...]
# env: RADIANCE_SRC (/mnt/data/bigcherry-work/engines/radiance), MODEL (.rad file, checkpoint dir or HF repo id;
#      default StillDeadcode/qwen3.8-27b-mxfp4, fetched once into DOWNLOAD_DIR), DOWNLOAD_DIR
#      (/mnt/data/llm-models/radiance), GPU (HIP index of the R9700, 2), CTX (180000), ARMS ("spec none"),
#      EXTRA (further serve flags), DECODE (512)
set -u
out=$2; shift 2
depths=("$@"); [ ${#depths[@]} -eq 0 ] && depths=(2048 8192 24576 98304)
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
MODEL=${MODEL:-StillDeadcode/qwen3.8-27b-mxfp4}
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
docker stop radiance-vllm > /dev/null 2>&1
echo "radiance $(git -C "$src" rev-parse --short HEAD); model $MODEL"

for arm in ${ARMS:-spec none}; do
    case "$arm" in
        spec) spec=auto ;;
        none) spec=0 ;;
        *) echo "$arm: unknown arm"; continue ;;
    esac
    serve=(--download-dir "${DOWNLOAD_DIR:-/mnt/data/llm-models/radiance}" --tp 1 --max-num-seqs 8
           --max-model-len "${CTX:-180000}" --kv-cache-dtype fp8 --num-speculative-tokens "$spec")
    # shellcheck disable=SC2206
    serve+=(${EXTRA:-})
    PYTHONPATH=tools python3 -m bigcherry engine-bench --engine radiance --binary "$src/build/bin/radiance" --model "$MODEL" --out "$out" --label "$arm" --depth "${depths[@]}" --decode "${DECODE:-512}" --reps 2 --env "HIP_VISIBLE_DEVICES=${GPU:-2}" -- "${serve[@]}" 2>&1 | sed "s/^/$arm /"
    sleep 5
done
exit 0
