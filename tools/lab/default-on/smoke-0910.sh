#!/bin/bash
# Startup smoke for 0910 (profile loader + init hooks): help lists the shipped profiles and exits 0; a real model
# load reaches /health with and without BIGCHERRY_FEATURES (the af459ed9 build deadlocked in ggml_init).
# Usage: smoke-0910.sh <llama-server> <out-dir>
set -u
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
bin=$1 out=$2
mkdir -p "$out"
ls "$(dirname "$bin")/profile/" > "$out/profile-files.txt" 2>&1
echo "smoke profile files: $(tr '\n' ' ' < "$out/profile-files.txt")"
BIGCHERRY_FEATURES=help timeout 60 "$bin" > "$out/help.txt" 2>&1
echo "help rc=$? profiles: $(grep -cE '^  [a-z0-9-]+( |$)' "$out/help.txt") flashnext: $(grep -c '^  flashnext' "$out/help.txt")"
model=/mnt/data/llm-models/qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf
for feat in "" hip-q81 flashnext; do
  port=$((46000 + RANDOM % 1000))
  BIGCHERRY_FEATURES=$feat HIP_VISIBLE_DEVICES=0 timeout 180 "$bin" -m "$model" -ngl 99 -c 4096 --port $port > "$out/load-${feat:-none}.log" 2>&1 &
  pid=$!
  ok=FAIL
  for _ in $(seq 120); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { ok=ok; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 1
  done
  echo "$ok load features='${feat:-none}' $(grep -h 'BIGCHERRY_FEATURES' "$out/load-${feat:-none}.log" | head -1 | cut -c1-160)"
  kill -INT $pid 2>/dev/null; wait $pid 2>/dev/null
done
