#!/bin/bash
# 4 independent contract-campaign sessions for patch 1241 (RD33, ne1==1 gate) on Brutus dual gfx1100.
# 5-minute cooldown between sessions (PA35: back-to-back sessions without a gap cause clock instability).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
M=/mnt/vault/llm-models
for s in 1 2 3 4; do
  jobs=$(mktemp)
  echo "VIS=0,1 MODEL=$M/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf 1241_rd33_mmvq_q8_0_f32_decode 1241_rd33_mmvq_q8_0_f32_decode/rd33 gfx1100 0,1 t-1241n1-gfx1100-s$s --producer-input control_model=$M/qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf --producer-corpus tools/bigcherry/bench/corpora/mtp-27b-v1.jsonl" > "$jobs"
  bash tools/lab/plan-qualification/queue.sh "$jobs"
  echo "SESSION_${s}_EXIT=$? $(date -Is)"
  rm -f "$jobs"
  [ "$s" -lt 4 ] && sleep 300
done
echo ALL_SESSIONS_DONE
