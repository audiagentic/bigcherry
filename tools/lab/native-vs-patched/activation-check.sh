#!/usr/bin/env bash
# Usage: activation-check.sh BINARY MODEL PATTERN [extra server args...]
# Launches llama-server on GPUs ${BC_GPUS:-0,1} with BIGCHERRY_PATCH_TRACE=1, sends one short completion, greps the log for PATTERN.
set -u
BIN=$1; MODEL=$2; PAT=$3; shift 3
LOG=$(mktemp -p "$HOME/bc-runs" activation.XXXXXX.log)
HIP_VISIBLE_DEVICES=${BC_GPUS:-0,1} ROCR_VISIBLE_DEVICES=${BC_GPUS:-0,1} BIGCHERRY_PATCH_TRACE=1 \
  "$BIN" -m "$MODEL" -sm tensor -ngl 99 --fit off -c 4096 --flash-attn on --parallel 1 --port 18092 "$@" >"$LOG" 2>&1 &
PID=$!
for i in $(seq 1 180); do curl -sf http://127.0.0.1:18092/health >/dev/null && break; sleep 2; done
curl -s http://127.0.0.1:18092/completion -d '{"prompt":"The capital of France is","n_predict":16,"temperature":0}' >/dev/null
kill -INT $PID; wait $PID 2>/dev/null
echo "log: $LOG"
grep -c "$PAT" "$LOG" | sed 's/^/hits: /'
grep -m3 "$PAT" "$LOG"
