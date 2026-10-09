#!/bin/bash
# MEN05: are the two 4-bit Qwen3.8-27B stacks equally close to a higher-precision reference? Same prompts, one token
# per probe with its top alternatives, no drafter anywhere (a drafter does not change a greedy first token, and off
# keeps the arms simple). Arms:
#   ref   llama.cpp, Q8_0 GGUF, f16 KV, the two RX 7900 XTX (production tensor split)   - the reference
#   ref2  the same again                                                                 - run-to-run floor
#   gguf4 llama.cpp, UD-Q4_K_M GGUF, f16 KV, the R9700 alone
#   rad4  radiance (source build), MXFP4 container, fp8 KV, the R9700 alone
# probes-compare.py then reports ref2, gguf4 and rad4 against ref, and gguf4 against rad4.
# Run as a queue SCRIPT job so the GPU lock covers it:
#   VIS=0,1,2,3 SCRIPT xq27 tools/lab/cross-engine/run-quality-27b.sh @<llama.cpp build run> <out-dir> [depth...]
# Usage: run-quality-27b.sh <llama-server> <out-dir> [depth-tokens...]
# env: ARMS ("ref ref2 gguf4 rad4"), PROBES (24), TOP (20), CTX (40960), RADIANCE_SRC, RAD_MODEL, REF_GGUF, Q4_GGUF,
#      R9700 (HIP index, 2), PORT (18746)
set -u
bin=$1 out=$2; shift 2
mkdir -p "$out"
depths=("$@"); [ ${#depths[@]} -eq 0 ] && depths=(2048 8192 24576)
here=$(cd "$(dirname "$0")" && pwd)
src=${RADIANCE_SRC:-/mnt/data/bigcherry-work/engines/radiance}
RAD_MODEL=${RAD_MODEL:-/mnt/data/llm-models/radiance/StillDeadcode/qwen3.8-27b-mxfp4/qwen3.8-27b-mxfp4.rad}
REF_GGUF=${REF_GGUF:-/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/Qwen3.8-27B-Q8_0.gguf}
Q4_GGUF=${Q4_GGUF:-/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/Qwen3.8-27B-UD-Q4_K_M.gguf}
PORT=${PORT:-18746}
CTX=${CTX:-40960}
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
docker stop radiance-vllm > /dev/null 2>&1

run_arm() {  # <arm>
    local arm=$1 pid ok=0 log="$out/$1.server.log"
    case "$arm" in
        ref|ref2)
            HIP_VISIBLE_DEVICES=0,1 "$bin" -m "$REF_GGUF" -ngl 99 -c "$CTX" -ub 512 -b 2048 -fa on --parallel 1 --threads 8 \
                -sm tensor -ts 1,1 --allreduce cpu-root -ctk f16 -ctv f16 --port "$PORT" --host 127.0.0.1 > "$log" 2>&1 & ;;
        gguf4)
            HIP_VISIBLE_DEVICES=${R9700:-2} "$bin" -m "$Q4_GGUF" -ngl 99 -c "$CTX" -ub 512 -b 2048 -fa on --parallel 1 --threads 8 \
                -ctk f16 -ctv f16 --port "$PORT" --host 127.0.0.1 > "$log" 2>&1 & ;;
        rad4)
            HIP_VISIBLE_DEVICES=${R9700:-2} RADIANCE_HOME=$src/build/radiance_home "$src/build/bin/radiance" --model "$RAD_MODEL" \
                --tp 1 --max-num-seqs 1 --max-model-len "$CTX" --kv-cache-dtype fp8 --num-speculative-tokens 0 \
                --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 & ;;
        *) echo "$arm: unknown arm"; return 2 ;;
    esac
    pid=$!
    for _ in $(seq 900); do
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && { ok=1; break; }
        kill -0 "$pid" 2> /dev/null || break
        sleep 1
    done
    if [ "$ok" != 1 ]; then
        echo "$arm: SERVER_FAILED $(grep -vE "^\s*$" "$log" | tail -3 | cut -c1-200 | tr '\n' '|')"
        kill -9 "$pid" 2> /dev/null
        return 1
    fi
    local model
    model=$(curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
    for d in "${depths[@]}"; do
        python3 "$here/probe-openai.py" "http://127.0.0.1:$PORT" "$model" "$out/$arm.d$d.json" "$d" \
            --probes "${PROBES:-24}" --top "${TOP:-20}" 2>&1 | sed "s/^/$arm /"
    done
    kill -INT "$pid" 2> /dev/null
    for _ in $(seq 60); do kill -0 "$pid" 2> /dev/null || break; sleep 1; done
    kill -9 "$pid" 2> /dev/null
    sleep 5
}

for arm in ${ARMS:-ref ref2 gguf4 rad4}; do run_arm "$arm"; done
for d in "${depths[@]}"; do
    files=()
    for arm in ref ref2 gguf4 rad4; do [ -f "$out/$arm.d$d.json" ] && files+=("$arm=$out/$arm.d$d.json"); done
    echo "== depth $d"
    [ ${#files[@]} -ge 2 ] && python3 "$here/probes-compare.py" "${files[@]}"
done
exit 0
