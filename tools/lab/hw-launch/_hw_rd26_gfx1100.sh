#!/bin/bash
# PA36 #4 hardware leg: RD26/1210 migrated producer on Brutus gfx1100 (GPU[0]).
# standard_campaign=run -> 5-build scaffold (bigcherry-tuning) + isolated llama-results pair.
# trace_probe=skip -> RD26 declares no activation check (determinism/correctness only).
# Bit-identity: decode vs verify raw-F32-logit byte-identity (fixed-length content).
set -x
cd /home/audumla/bc-pa-work
export PYTHONPATH=tools
export ROCM_PATH=/mnt/vault/tmp/bc-rocm
export HIP_PATH=/mnt/vault/tmp/bc-rocm
export PATH=/mnt/vault/tmp/bc-rocm/bin:$PATH
python3 -m bigcherry.patch.validation_campaign \
  --patch 1210_rd26_bitidentical_decode_verify_standalone \
  --validation-producer 1210_rd26_bitidentical_decode_verify_standalone/rd26 \
  --baseline-source bigcherry-tuning \
  --amdgpu-targets gfx1100 \
  --device-map gfx1100=0 \
  --model /mnt/vault/llm-models/qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf \
  --hip-path /mnt/vault/tmp/bc-rocm \
  --workdir /mnt/vault/tmp/bc-runs/rd26-gfx1100-run1 \
  --worktree-root /mnt/vault/tmp/bc-worktrees
echo "RD26_EXIT=$?"
