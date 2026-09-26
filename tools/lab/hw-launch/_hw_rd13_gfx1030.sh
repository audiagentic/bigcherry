#!/bin/bash
# PA36 #3 hardware leg: RD13/1206 migrated producer on Brutus gfx1030 (GPU[3]).
# standard_campaign=run -> 5-build scaffold (bigcherry-tuning) + isolated llama-server pair.
# trace_probe=run -> scaffold two-probe trace-marker activation (positive + fusion-disabled negative).
set -x
cd /home/audumla/bc-pa-work
export PYTHONPATH=tools
export ROCM_PATH=/mnt/vault/tmp/bc-rocm
export HIP_PATH=/mnt/vault/tmp/bc-rocm
export PATH=/mnt/vault/tmp/bc-rocm/bin:$PATH
python3 -m bigcherry.patch.validation_campaign \
  --patch 1206_rd13_mul_mat_add_view_fusion \
  --validation-producer 1206_rd13_mul_mat_add_view_fusion/rd13 \
  --baseline-source bigcherry-tuning \
  --amdgpu-targets gfx1030 \
  --device-map gfx1030=3 \
  --model /mnt/vault/llm-models/qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf \
  --hip-path /mnt/vault/tmp/bc-rocm \
  --workdir /mnt/vault/tmp/bc-runs/rd13-gfx1030-run1 \
  --worktree-root /mnt/vault/tmp/bc-worktrees
echo "RD13_EXIT=$?"
