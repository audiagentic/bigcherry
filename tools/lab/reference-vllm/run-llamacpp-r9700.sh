#!/bin/bash
# Like-for-like side of the reference lane: BigCherry llama-server on the R9700 ALONE with a 4-bit Qwen3.8-27B, measured
# with `bigcherry engine-bench`, the same command, corpus, depths and repeats as tools/lab/radiance/run-radiance.sh.
# No tensor split, so no AllReduce: this separates "another engine against llama.cpp" from "one card against a split"
# and "4-bit against Q8_0".
# One arm per drafter: mtp = the GGUF's built-in MTP head (--spec-draft-n-max 4, our production drafter),
# dflash = the DFlash2 drafter with 7 tokens (what the vLLM container uses), none = no drafter.
# KV is f16; q8_0 only if f16 does not fit (KV_FALLBACK=q8_0 is then tried once). Never q4.
# Each arm writes <out-dir>/<arm>-<kv>.engine-bench.json and <arm>-<kv>.server.log.
# Run as a queue SCRIPT job so the GPU lock covers it:
#   VIS=0,1,2,3 SCRIPT ref-llamacpp tools/lab/reference-vllm/run-llamacpp-r9700.sh @<build run> <out-dir> [depth...]
# Usage: run-llamacpp-r9700.sh <llama-server> <out-dir> [depth-tokens...]
# env: MODEL, DFLASH, GPU (HIP index of the R9700, 2), CTX (180000), ARMS ("mtp dflash none"), DECODE (512)
set -u
bin=$1 out=$2; shift 2
depths=("$@"); [ ${#depths[@]} -eq 0 ] && depths=(2048 8192 24576 98304)
MODEL=${MODEL:-/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/Qwen3.8-27B-UD-Q4_K_M.gguf}
DFLASH=${DFLASH:-/mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf}
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
# single-model reference run: no Flash-Next runtime flags or topology settings from the caller
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
docker stop radiance-vllm > /dev/null 2>&1

run_arm() {  # <arm> <kv type>
    local arm=$1 kv=$2 spec=()
    case "$arm" in
        mtp) spec=(--spec-type draft-mtp --spec-draft-n-max 4) ;;
        dflash) spec=(--spec-type draft-dflash -md "$DFLASH" -devd ROCm0 --spec-draft-n-max 7) ;;  # ROCm0 = the one visible card
        none) spec=() ;;
        *) echo "$arm: unknown arm"; return 2 ;;
    esac
    local serve=(-ngl 99 -c "${CTX:-180000}" -ub 512 -b 2048 -fa on --parallel 1 --threads 8 -ctk "$kv" -ctv "$kv" "${spec[@]}")
    PYTHONPATH=tools python3 -m bigcherry engine-bench --engine llamacpp --binary "$bin" --model "$MODEL" --out "$out" --label "$arm-$kv" --depth "${depths[@]}" --decode "${DECODE:-512}" --reps 2 --health-timeout 600 --env "HIP_VISIBLE_DEVICES=${GPU:-2}" -- "${serve[@]}" 2>&1 | sed "s/^/$arm kv=$kv /"
    local rc=${PIPESTATUS[0]}
    sleep 5
    return "$rc"
}

echo "model $(basename "$MODEL"); binary $bin"
for arm in ${ARMS:-mtp dflash none}; do
    run_arm "$arm" f16 || { [ -n "${KV_FALLBACK:-q8_0}" ] && run_arm "$arm" "${KV_FALLBACK:-q8_0}"; }
done
exit 0
