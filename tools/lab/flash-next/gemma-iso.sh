#!/bin/bash
# Isolate the garbage Gemma-4-26B-A4B output seen when the Flash-Next env leaked into the smoke run: run Gemma on the
# three tensor-split GPUs with (a) nothing, (b) only BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0 (1303),
# (c) only the Q8_1 producer flags (1307-1311), (d) ATTN_TS rotated (=1). Prints each completion's first line.
# Usage: gemma-iso.sh <llama-server> <out-dir>
set -u
bin=$1 out=$2
mkdir -p "$out"
port=18751
model=/mnt/data/llm-models/gemma-4-26B-A4B/gguf/gemma-4-26B-A4B-it-UD-Q5_K_S.gguf
PROMPT='Write three short sentences about why the sky is blue.'
CLEAR="-u BIGCHERRY_ATTN_TS -u BIGCHERRY_ATTN_ROTATE -u BIGCHERRY_FFN_TS -u GGML_HIP_Q8_1_CACHE_MODE -u BIGCHERRY_ROLLBACK_NO_CONT -u BIGCHERRY_RMS_Q81 -u BIGCHERRY_ACT_Q81 -u BIGCHERRY_HC_Q81"
for arm in plain attn_ts q81 attn_ts_rot; do
    case $arm in
        plain)       envs="" ;;
        attn_ts)     envs="BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0" ;;
        q81)         envs="GGML_HIP_Q8_1_CACHE_MODE=on BIGCHERRY_RMS_Q81=1 BIGCHERRY_ACT_Q81=1 BIGCHERRY_HC_Q81=1" ;;
        attn_ts_rot) envs="BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=1" ;;
    esac
    env $CLEAR HIP_VISIBLE_DEVICES=0,1,2 $envs "$bin" -m "$model" -ngl 99 -c 4096 --port $port --host 127.0.0.1 -fa on \
        -sm tensor -ts 1,1,1 --allreduce cpu-root > "$out/$arm.log" 2>&1 &
    pid=$!
    ok=0
    for _ in $(seq 240); do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=1; break; }
        kill -0 $pid 2>/dev/null || break
        sleep 1
    done
    if [ $ok = 1 ]; then
        txt=$(curl -s "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
            -d "{\"prompt\": \"$PROMPT\", \"n_predict\": 48, \"temperature\": 0, \"seed\": 1, \"cache_prompt\": false}" \
            | python3 -c 'import json,sys; print(json.load(sys.stdin).get("content","").replace("\n"," "))')
        echo "== gemma $arm: ${txt:0:140}"
    else
        echo "== gemma $arm: SERVER_FAILED $(grep -m1 -E ' E |error' $out/$arm.log)"
    fi
    kill -INT $pid 2>/dev/null
    for _ in $(seq 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done
    kill -9 $pid 2>/dev/null
    wait $pid 2>/dev/null
done
