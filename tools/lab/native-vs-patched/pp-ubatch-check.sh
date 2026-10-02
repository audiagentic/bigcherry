#!/bin/bash
# Prefill check for Qwen3.8-27B Q8_0 on dual XTX (-sm tensor): llama-bench pp1665/pp4096 at ubatch 512 vs
# 2048 for each given build (llama-bench from the same bin dir as each llama-server). Explains the ~560 t/s
# prefill seen in the server-based draft probe against the ~1249 t/s pp4096 baseline.
# Usage: pp-ubatch-check.sh <out-dir> <llama-server> [<llama-server> ...]
set -u
out=$1; shift
mkdir -p "$out"
model=/mnt/data/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
for srv in "$@"; do
  bench=$(dirname "$srv")/llama-bench
  tag=$(basename "$(dirname "$(dirname "$srv")")")
  echo "== $tag ($bench)"
  HIP_VISIBLE_DEVICES=0,1 ROCR_VISIBLE_DEVICES=0,1 "$bench" -m "$model" -sm tensor -ngl 99 -fa 1 \
    -p 1665,4096 -n 0 -ub 512,2048 -b 2048 -r 3 -o md 2>"$out/$tag.stderr" | tee "$out/$tag.md"
done
echo PP_CHECK_DONE
