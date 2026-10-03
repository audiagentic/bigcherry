#!/bin/bash
# 1303 max-context search: f16-K/q8_0-V, ub512, with the attention/KV split (BIGCHERRY_ATTN_TS, unrotated)
# decoupled from the expert-weight -ts that maxctx-search.py adapts. Same deployment draft settings.
# Usage: attn-maxctx.sh <llama-server> <out-root> <attn-ts> [rotate=0] [lo] [hi]
set -u
bin=$1 root=$2 attn=$3 rot=${4:-0} lo=${5:-147456} hi=${6:-212992}
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTKD=f16 CTVD=f16 B=512 EXTRA_OT='^token_embd\.weight$=CPU'
export BIGCHERRY_ATTN_TS=$attn BIGCHERRY_ATTN_ROTATE=$rot
python3 tools/lab/flash-next/maxctx-search.py "$bin" "$root" f16 q8_0 512 "$lo" "$hi"
grep -h "BIGCHERRY_PATCH_HIT attn_ts" "$root"/*/timing.server.log 2>/dev/null | head -1
