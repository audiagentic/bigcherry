#!/bin/bash
# PA36 RD04/1202 DEFERRED hardware receipt: the CONTRACT model (tierA-qwen4b-q6k)
# on gfx1100 (GPU[0]). The earlier 0.8B smoke-model run was GPT-REJECTED as the
# wrong model; this is the contract-equivalent receipt.
# standard_campaign=run -> 5-build scaffold (bigcherry-tuning) + isolated PPL pair
# (primary_target=llama-perplexity, bigcherry baseline, fat-3, forced -fa on bf16)
# + paired benchmark reusing the scaffold's llama-bench pair. trace_probe=skip.
set -x
cd /home/audumla/bc-pa-work
export PYTHONPATH=tools
export ROCM_PATH=/mnt/vault/tmp/bc-rocm
export HIP_PATH=/mnt/vault/tmp/bc-rocm
export PATH=/mnt/vault/tmp/bc-rocm/bin:$PATH
python3 -m bigcherry.patch.validation_campaign \
	--patch 1202_rd04_bf16_flash_attn_tile \
	--validation-producer 1202_rd04_bf16_flash_attn_tile/rd04 \
	--baseline-source bigcherry-tuning \
	--amdgpu-targets gfx1100 \
	--device-map gfx1100=0 \
	--model /mnt/vault/llm-models/qwen3.5-4B/gguf/mtp/Qwen3.5-4B-UD-Q6_K_XL.gguf \
	--producer-corpus /home/audumla/perf-sweep-20260911/corpus/wikitext-2-raw/wiki.test.raw \
	--hip-path /mnt/vault/tmp/bc-rocm \
	--workdir /mnt/vault/tmp/bc-runs/rd04-gfx1100-contract-run2 \
	--worktree-root /mnt/vault/tmp/bc-worktrees
echo "RD04_EXIT=$?"
