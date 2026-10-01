#!/bin/bash
# Reproduce the 3-GPU --allreduce host segfault (libamdhip64, first request; 27B BF16 on
# 2x XTX + R9700, 2026-10-01 ab-3g-bf16-nway) under gdb and save the backtrace.
# Usage: host3-crash-repro.sh <llama-server> <out-dir>
set -u
bin=$1 out=$2
mkdir -p "$out"
model=/mnt/data/llm-models/qwen3.8-27b/gguf/unsloth/BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf
port=$((42000 + RANDOM % 2000))
export HIP_VISIBLE_DEVICES=0,1,2 ROCR_VISIBLE_DEVICES=0,1,2 BIGCHERRY_PATCH_TRACE=1
gdb -batch -ex "set pagination off" -ex run -ex "thread apply all bt 30" --args \
  "$bin" -m "$model" -sm tensor -ngl 99 --fit off -c 8192 --flash-attn on --parallel 1 \
  --allreduce host --port "$port" > "$out/gdb.log" 2>&1 &
gpid=$!
for _ in $(seq 180); do curl -sf "http://127.0.0.1:$port/health" >/dev/null && break; sleep 2; done
curl -s -m 300 "http://127.0.0.1:$port/completion" -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello, how are you today?","n_predict":8}' > "$out/response.json" 2>&1
echo "CURL_EXIT=$?"
sleep 5
# If the server survived, stop it cleanly (SIGINT to the inferior via its PID).
spid=$(pgrep -P "$gpid" -f llama-server | head -1)
[ -n "$spid" ] && kill -INT "$spid"
wait "$gpid"
grep -nE "SIGSEGV|signal|^#[0-9]+ " "$out/gdb.log" | head -60 > "$out/backtrace.txt"
cat "$out/backtrace.txt"
echo REPRO_DONE
