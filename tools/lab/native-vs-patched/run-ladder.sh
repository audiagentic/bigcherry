#!/bin/bash
# Native (stock) vs patched reference ladder + activation probes on Brutus.
# Usage: run-ladder.sh STOCK_BIN_DIR PATCHED_BIN_DIR OUT_DIR
set -u
source /home/audumla/bc-pytest-venv/bin/activate
export PYTHONPATH=tools ROCM_PATH=/opt/rocm
cd /mnt/vault/development/projects/bigcherry/workspaces/main
source tools/lab/plan-qualification/cooldown.sh
STOCK=$1; PATCHED=$2; O=$3
mkdir -p "$O"
MR=/mnt/vault/llm-models
ladder() { python -m bigcherry reference-ladder --arm stock="$STOCK" --arm patched="$PATCHED" --model-id "$1" --devices "$2" --output "$O/$3" --rounds-per-arm 3; }
probe() { # model-path devices tag extra-args
  mkdir -p "$O/activation"
  HIP_VISIBLE_DEVICES=$2 BIGCHERRY_PATCH_HIT=1 BIGCHERRY_PATCH_TRACE=1 "$PATCHED/llama-bench" -m "$MR/$1" -p 512 -n 8 -r 1 -ngl 99 -v $4 > "$O/activation/$3.log" 2>&1
  grep -a "BIGCHERRY_PATCH_HIT" "$O/activation/$3.log" | sed 's/^.*BIGCHERRY_PATCH_HIT/BIGCHERRY_PATCH_HIT/' | sort | uniq -c > "$O/activation/$3.hits"
}
A=qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf
B=qwen3.5-9B/gguf/mtp/Qwen3.5-9B-Q6_K.gguf
L=qwen3.8-27b/gguf/mtp/Qwen3.8-27B-Q8_0.gguf
probe $A 2 tierA-gfx1201 ""
ladder tierA-qwen4b-q6k 2 gfx1201; sleep 120
probe $A 0 tierA-gfx1100 ""
ladder tierA-qwen4b-q6k 0 gfx1100; sleep 120
probe $B 0 tierB-gfx1100 ""
ladder tierB-qwen9b-q6k 0 gfx1100; gpu_cooldown
probe $L 0,1 tierL-tensor2 "-sm tensor"
ladder tierL-qwen27b-q8 0,1 tensor2
echo LADDER-DONE
