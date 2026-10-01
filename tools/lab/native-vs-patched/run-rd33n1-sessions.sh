#!/bin/bash
# 4 independent contract-campaign sessions for patch 1241 (RD33, ne1==1 gate) on Brutus dual gfx1100.
# Cooldown between sessions (tools/lab/plan-qualification/cooldown.sh: >= 30 s, then until edge <= 55 C, max 300 s).
set -u
prefix=${1:?usage: run-rd33n1-sessions.sh <run-name-prefix, e.g. t-1241n2-gfx1100>}
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
source tools/lab/plan-qualification/cooldown.sh
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
M=/mnt/vault/llm-models
for s in 1 2 3 4; do
  jobs=$(mktemp)
  echo "VIS=0,1 MODEL=$M/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf 1241_rd33_mmvq_q8_0_f32_decode 1241_rd33_mmvq_q8_0_f32_decode/rd33 gfx1100 0,1 $prefix-s$s --producer-input control_model=$M/qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf --producer-corpus tools/bigcherry/bench/corpora/mtp-27b-v1.jsonl" > "$jobs"
  bash tools/lab/plan-qualification/queue.sh "$jobs"
  echo "SESSION_${s}_EXIT=$? $(date -Is)"
  rm -f "$jobs"
  [ "$s" -lt 4 ] && gpu_cooldown
done
echo ALL_SESSIONS_DONE
