#!/bin/bash
# GP11: winners are hardware-keyed, so every (card, model) pair needs its own
# tune run -- a gfx1100 cache says nothing about gfx1201 or gfx1030.
# See README.md.
cd ~/rd73-requal-clone
export PYTHONPATH=tools ROCM_PATH=/home/audumla/rocm-shim HIP_PATH=/home/audumla/rocm-shim
export PATH=/home/audumla/rocm-shim/bin:$PATH
M27=/mnt/vault/llm-models/qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
M9=/mnt/vault/llm-models/qwen3.5-9B/gguf/mtp/Qwen3.5-9B-Q6_K.gguf
tune () {
  local label=$1 model=$2 devs=$3 prof=$4
  echo "=== TUNE $label START $(date -Is) ==="
  python3 -m bigcherry tune-campaign --source bigcherry --platform linux-multi \
    --model "$model" --devices "$devs" --runtime-profile "$prof" 2>&1 \
    | grep -aE "promoted:|replay coverage|receipt:|Traceback|Error|winner" | tail -6
  echo "=== TUNE $label END $(date -Is) ==="
}
# gfx1201 can hold the 27B single-card (32.6GB); gfx1030 (16GiB) cannot, so it
# is tuned on the 9B -- which is also the only model it can ever serve.
tune "gfx1201-27b" "$M27" "2" production-safe-single
tune "gfx1030-9b"  "$M9"  "3" production-safe-single
tune "gfx1201-9b"  "$M9"  "2" production-safe-single
echo TUNE_REST_DONE
