#!/bin/bash
# DFlash2 / DSpark draft acceptance collapsed to 1-2% at pin b11474 (it was 44-59% at b11402): is that upstream or
# ours? Single-device target (Qwen3.8-27B Q4_K_M on one XTX, so no tensor split and no borrowed-tensor copy, patch
# 1286 not involved), draft on the same card and on the 6900 XT. Run the same layouts on native llama.cpp and on
# BigCherry builds and compare acceptance. Reuses probe() from probe-27b.sh.
# Usage: dflash-accept-iso.sh <llama-server> <out-dir> [layout regex] [reps, default 3]     env: REPO=<lab tree>
set -u
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
bin=$1 out=$2 only=${3:-} reps=${4:-3}
mkdir -p "$out"
G=/mnt/data/llm-models/qwen3.8-27b/gguf
model=$G/mtp/Qwen3.8-27B-Q4_K_M.gguf
common=(-m "$model" -ngl 99 --fit off -c 8192 --flash-attn on --parallel 1 --threads 8 -ub 2048 -b 2048 -lv 4)
eval "$(sed -n '/^probe() {$/,/^}$/p' "$REPO/tools/lab/dflash/probe-27b.sh")"

T="0,1,2,3 -dev ROCm0"
probe plain $T
probe mtp4 $T --spec-type draft-mtp --spec-draft-n-max 4 -ctkd q8_0 -ctvd q8_0
probe dflash-q8-n4-same $T -devd ROCm0 -md $G/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 4
probe dflash-q8-n7-same $T -devd ROCm0 -md $G/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 7
probe dflash-q8-n4-d6900 $T -devd ROCm3 -md $G/dflash/Qwen3.8-27B-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 4
probe dspark-q8-n6-same $T -devd ROCm0 -md $G/dspark/Qwen3.8-27B-DSpark-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 6
for f in "$out"/*.server.log; do
    e=$(grep -hE "\.cpp:[0-9]+: |failed to" "$f" | sed -E 's#.*/(ggml|src|common)/#\1/#' | head -1 | cut -c1-160)
    [ -n "$e" ] && echo "  $(basename "$f" .server.log): $e"
done
