#!/bin/bash
# Functional smoke (not a benchmark): does the Flash-Next production-v3 build (1291-1310) still run other
# architectures correctly with the new env-gated paths ON? For each model: start llama-server, one short greedy
# completion with the v3 flags OFF and ON, compare the outputs byte-for-byte and show the first line. Flash-Next-only
# settings inherited from a calling queue (BIGCHERRY_ATTN_TS etc.) are cleared so each model runs its own config.
# Usage: smoke-models.sh <llama-server> <out-dir>
set -u
bin=$1 out=$2
mkdir -p "$out"
port=18731
ON="GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_ROLLBACK_NO_CONT=1 BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1 BIGCHERRY_SCALE_ACT_FUSE=1 BIGCHERRY_SCHED_ASYNC_INPUTS=1"
M=/mnt/data/llm-models
PROMPT='Write three short sentences about why the sky is blue.'

run() {  # name devices split extra-args model
    local name=$1 devs=$2 sm=$3 extra=$4 model=$5
    for mode in off on; do
        local envs="HIP_VISIBLE_DEVICES=$devs"
        [ "$mode" = on ] && envs="$envs $ON"
        local log="$out/$name.$mode.log"
        env -u BIGCHERRY_ATTN_TS -u BIGCHERRY_ATTN_ROTATE -u BIGCHERRY_FFN_TS -u BIGCHERRY_DRAFT_VOCAB_N -u GGML_HIP_Q8_1_CACHE_MODE -u BIGCHERRY_ROLLBACK_NO_CONT -u BIGCHERRY_RMS_Q81 -u BIGCHERRY_ACT_Q81 -u BIGCHERRY_HC_Q81 -u BIGCHERRY_SCALE_ACT_FUSE -u BIGCHERRY_SCHED_ASYNC_INPUTS -u BIGCHERRY_QSA_HOST_REMAP $envs "$bin" -m "$model" -ngl 99 -c 4096 --port $port --host 127.0.0.1 -fa on $sm $extra > "$log" 2>&1 &
        local pid=$!
        local ok=0
        for _ in $(seq 240); do
            curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }
            kill -0 $pid 2>/dev/null || break
            sleep 1
        done
        if [ $ok = 1 ]; then
            curl -s "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
                -d "{\"prompt\": \"$PROMPT\", \"n_predict\": 64, \"temperature\": 0, \"seed\": 1, \"cache_prompt\": false}" \
                | python3 -c 'import json,sys; print(json.load(sys.stdin).get("content",""))' > "$out/$name.$mode.txt"
        else
            echo "SERVER_FAILED" > "$out/$name.$mode.txt"
            grep -E " E |error|assert|ABORT" "$log" | head -3
        fi
        kill -INT $pid 2>/dev/null
        for _ in $(seq 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done
        kill -9 $pid 2>/dev/null
        wait $pid 2>/dev/null
    done
    local verdict=DIFFERENT
    cmp -s "$out/$name.off.txt" "$out/$name.on.txt" && verdict=IDENTICAL
    grep -q SERVER_FAILED "$out/$name.on.txt" "$out/$name.off.txt" && verdict=FAILED
    echo "== $name: $verdict | on: $(head -c 120 "$out/$name.on.txt" | tr '\n' ' ')"
}

# dense, 2x XTX tensor split with cpu-root AllReduce
run qwen27b-q8-tp2 0,1 "-sm tensor -ts 1,1" "--allreduce cpu-root" $M/qwen3.8-27b/gguf/unsloth/Qwen3.8-27B-Q8_0.gguf
# MoE Gemma across the three tensor-split GPUs
run gemma26b-a4b-tp3 0,1,2 "-sm tensor -ts 1,1,1" "--allreduce cpu-root" $M/gemma-4-26B-A4B/gguf/gemma-4-26B-A4B-it-UD-Q5_K_S.gguf
# a different MoE architecture on a single GPU (R9700)
run gptoss20b-r9700 2 "" "" $M/gpt-oss-20B/gguf/gpt-oss-20b-UD-Q6_K_XL.gguf
echo ALL_JOBS_DONE
