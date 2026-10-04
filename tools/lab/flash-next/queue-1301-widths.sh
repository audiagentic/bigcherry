#!/bin/bash
# QFP21 1301 width sweep on the 27B: MAXCOLS 2 and 3 vs 1 (MAXCOLS 5 regressed -6..-12%).
set -u
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export BC_HIP_PATH=/mnt/vault/tmp/bc-rocm
export BC_MODEL=/mnt/data/llm-models/qwen3.8-flash-next/gguf/mtp/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export DRAFT=/mnt/data/llm-models/qwen3.8-flash-next/gguf/unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q5_K_M-qsa4.gguf
export BIGCHERRY_DRAFT_VOCAB_N=65536 CTK=f16 CTV=f16 CTKD=f16 CTVD=f16 CTX=245760 TS=0.31,0.27,0.42 B=512 DECODE_N=512
export EXTRA_OT='^token_embd\.weight$=CPU' BIGCHERRY_ATTN_TS=1,1,0 BIGCHERRY_ATTN_ROTATE=0
export BIGCHERRY_FEATURES=flashnext
until grep -q ALL_JOBS_DONE /mnt/data/bigcherry-work/runs/queue-default-on.log 2>/dev/null; do sleep 30; done
docker stop radiance-vllm >/dev/null 2>&1
jobs=$(mktemp)
R=/mnt/data/bigcherry-work/runs
cat > "$jobs" <<JOBS
VIS=0,1,2,3 BUILD b-r1301b bigcherry:stock:linux-multi retest-1301 gfx1100,gfx1201,gfx1030
VIS=0,1 SCRIPT q8w2-27b tools/lab/flash-next/prod27b-ab.sh @b-r1301b @b-r1301b $R/qfp21-1301-27b-w2 BIGCHERRY_Q8_F32_MAXCOLS=2
VIS=0,1 SCRIPT q8w3-27b tools/lab/flash-next/prod27b-ab.sh @b-r1301b @b-r1301b $R/qfp21-1301-27b-w3 BIGCHERRY_Q8_F32_MAXCOLS=3
JOBS
for v in $(env | grep -oE "^(BIGCHERRY_[A-Z0-9_]+|GGML_HIP_[A-Z0-9_]+)"); do unset "$v"; done
bash tools/lab/plan-qualification/queue.sh "$jobs"
echo "QUEUE_EXIT=$? $(date -Is)"
rm -f "$jobs"
for w in 2 3; do echo "== MAXCOLS $w (A = 1, B = $w)"; grep -E "^d[0-9]|SERVER_FAILED|^ +[0-9]" $R/q8w$w-27b.log; done
echo ALL_JOBS_DONE
