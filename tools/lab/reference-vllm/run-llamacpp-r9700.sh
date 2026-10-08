#!/bin/bash
# Like-for-like side of the reference lane: BigCherry llama-server on the R9700 ALONE with a 4-bit Qwen3.8-27B, measured
# with the same client, corpus, depths and repeats as run-radiance.sh. No tensor split, so no AllReduce: this separates
# "vLLM against llama.cpp" from "one card against a split" and "4-bit against Q8_0".
# One arm per drafter: mtp = the GGUF's built-in MTP head (--spec-draft-n-max 4, our production drafter),
# dflash = the DFlash2 drafter with 7 tokens (what the vLLM container uses), none = no drafter.
# KV is f16; q8_0 only if f16 does not fit (KV_FALLBACK=q8_0 is then tried once). Never q4.
# Run as a queue SCRIPT job so the GPU lock covers it:
#   VIS=0,1,2,3 SCRIPT ref-llamacpp tools/lab/reference-vllm/run-llamacpp-r9700.sh @<build run> <out-dir> [depth...]
# Usage: run-llamacpp-r9700.sh <llama-server> <out-dir> [depth-tokens...]
# env: MODEL, DFLASH, GPU (HIP index of the R9700, 2), CTX (180000), ARMS ("mtp dflash none"), DECODE (512), PORT (18744)
set -u
bin=$1 out=$2; shift 2
mkdir -p "$out"
depths=("$@"); [ ${#depths[@]} -eq 0 ] && depths=(2048 8192 24576 98304)
MODEL=${MODEL:-/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/Qwen3.8-27B-UD-Q4_K_M.gguf}
DFLASH=${DFLASH:-/mnt/data/llm-models/qwen3.8-27b/gguf/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf}
PORT=${PORT:-18744}
here=$(cd "$(dirname "$0")" && pwd)
# single-model reference run: no Flash-Next runtime flags or topology settings from the caller
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
export HIP_VISIBLE_DEVICES=${GPU:-2}
docker stop radiance-vllm > /dev/null 2>&1

run_arm() {  # <arm> <kv type>
    local arm=$1 kv=$2 spec=() pid ok=0
    case "$arm" in
        mtp) spec=(--spec-type draft-mtp --spec-draft-n-max 4) ;;
        dflash) spec=(--spec-type draft-dflash -md "$DFLASH" -devd ROCm0 --spec-draft-n-max 7) ;;  # ROCm0 = the one visible card
        none) spec=() ;;
        *) echo "$arm: unknown arm"; return 2 ;;
    esac
    "$bin" -m "$MODEL" -ngl 99 -c "${CTX:-180000}" -ub 512 -b 2048 -fa on --parallel 1 --threads 8 \
        -ctk "$kv" -ctv "$kv" --port "$PORT" --host 127.0.0.1 "${spec[@]}" > "$out/$arm.server.log" 2>&1 &
    pid=$!
    for _ in $(seq 600); do
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { ok=1; break; }
        kill -0 "$pid" 2> /dev/null || break
        sleep 1
    done
    if [ "$ok" != 1 ]; then
        echo "$arm kv=$kv: SERVER_FAILED $(grep -E " E |error|failed|out of memory" "$out/$arm.server.log" | tail -2 | cut -c1-200 | tr '\n' ' ')"
        kill -9 "$pid" 2> /dev/null
        return 1
    fi
    echo "== $arm kv=$kv ctx=${CTX:-180000} device: $(grep -m1 -oE "using device [^)]*\)" "$out/$arm.server.log")"
    rocm-smi --showmeminfo vram 2> /dev/null | grep "Total Used" > "$out/$arm.vram.txt"
    local model
    model=$(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
    python3 "$here/bench-openai.py" "http://127.0.0.1:$PORT" "$model" "$out/$arm.bench.json" "${depths[@]}" \
        --decode "${DECODE:-512}" --reps 2 | sed "s/^/$arm /"
    kill -INT "$pid" 2> /dev/null
    for _ in $(seq 60); do kill -0 "$pid" 2> /dev/null || break; sleep 1; done
    kill -9 "$pid" 2> /dev/null
    grep -E "draft acceptance|n_draft|accepted" "$out/$arm.server.log" | tail -1 | cut -c1-200 | sed "s/^/$arm /"
    sleep 5
    return 0
}

echo "model $(basename "$MODEL"); binary $bin"
for arm in ${ARMS:-mtp dflash none}; do
    run_arm "$arm" f16 || { [ -n "${KV_FALLBACK:-q8_0}" ] && run_arm "$arm" "${KV_FALLBACK:-q8_0}"; }
done
exit 0
